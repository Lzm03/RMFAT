import os
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from torchvision.utils import save_image
from data.dataset_video_train import DataLoaderTurbVideo
from model.TMT_DC import TMT_MS
from utils.general import create_log_folder
from utils.losses import CharbonnierLoss, dwtLoss
from ultralytics import YOLO

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics.utils.ops import non_max_suppression
import lpips 
import time 

def visualize_flow(flow):
    """
    输入:
        flow: numpy array or torch.Tensor, shape [2, H, W] or [B, 2, H, W]
    输出:
        RGB flow visualization (np.array)
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().squeeze(0).numpy()  # remove batch dim if needed

    if flow.shape[0] == 2:  # shape: [2, H, W]
        u, v = flow[0], flow[1]
    else:
        raise ValueError("Flow must have shape [2, H, W]")

    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2               # angle (0~360 → 0~180 for OpenCV hue)
    hsv[..., 1] = 255                   # saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value (magnitude)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

def get_args():
    parser = argparse.ArgumentParser(description="Recursive 2-frame restoration training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--run_name", type=str, default="Recursive2Frame")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tmt_dims", type=int, default=16)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--quick_val", action="store_true")
    parser.add_argument("--lambda_dwt", type=float, default=0.02)
    parser.add_argument("--train_label_dir", type=str, required=True)
    parser.add_argument("--val_label_dir", type=str, required=True)
    parser.add_argument("--eval_only", action="store_true", help="Only run validation without training")

    return parser.parse_args()


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    box1: [N, 4], box2: [M, 4]
    Each box: (x1, y1, x2, y2)
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]).clamp(0) * (box[:, 3] - box[:, 1]).clamp(0)

    area1 = box_area(box1)
    area2 = box_area(box2)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)



def person_detection_loss(pred, gt):
    """
    pred: shape [N, 6] from YOLO: [x1, y1, x2, y2, conf, cls]
    gt:   shape [M, 5]: [class, cx, cy, w, h] normalized
    """
    gt = gt[gt[:, 1] == 0]  # class == person

    if len(pred) == 0 or len(gt) == 0:
        return torch.tensor(0.0, requires_grad=True).to(pred.device)

    gt_xy = gt[:, 2:4]
    gt_wh = gt[:, 4:6]
    gt_xy1 = gt_xy - gt_wh / 2
    gt_xy2 = gt_xy + gt_wh / 2
    gt_boxes = torch.cat([gt_xy1, gt_xy2], dim=1)

    pred_boxes = pred[:, :4] / 640  # normalized
    ious = box_iou(pred_boxes, gt_boxes)
    max_iou, _ = ious.max(dim=1)

    conf_loss = F.binary_cross_entropy(pred[:, 4], max_iou.detach())
    iou_loss = 1 - max_iou.mean()

    return conf_loss + iou_loss

def load_yolo_labels(label_path, img_shape, device):
    if not os.path.exists(label_path):
        return torch.zeros((0, 6)).to(device)

    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls, x, y, w, h = map(float, parts)
            labels.append([0, cls, x, y, w, h])

    return torch.tensor(labels, dtype=torch.float32).to(device)

def flow_warp(img, flow):
    """
    Warp img using flow.
    img: [B,C,H,W]
    flow: [B,2,H,W], flow vectors (dx,dy) for each pixel
    """
    assert flow.dim() == 4, f"flow must be [B, 2, H, W], but got {flow.shape}"
    B, C, H, W = img.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(img.device)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]

    # normalize grid to [-1,1]
    vgrid = grid + flow.permute(0, 2, 3, 1)
    vgrid_x = 2.0 * vgrid[..., 0] / (W - 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / (H - 1) - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=-1)

    output = F.grid_sample(img, vgrid_norm, align_corners=True)
    return output

def collate_fn(batch):
    return tuple(zip(*batch))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def blend_prev_frame(O_prev, T_t, alpha_t):
    return alpha_t * O_prev + (1 - alpha_t) * T_t.detach()

def main():
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    args = get_args()
    device = get_device()
    
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_loss_fn.eval()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = args.run_name + "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    result_img_path, path_ckpt = create_log_folder(run_path)
    logging.basicConfig(filename=os.path.join(run_path, "training.log"), level=logging.INFO)
    writer = SummaryWriter(log_dir=run_path)
    
    yolo_model = YOLO("/root/autodl-tmp/MitigationDetection/yolov8n.pt").to(device)
    yolo_model.eval()

    train_dataset = DataLoaderTurbVideo(
        root_dir=args.train_path,
        label_dir=args.train_label_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=True,
        max_clips_per_video=1
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

    val_dataset = DataLoaderTurbVideo(
        root_dir=args.val_path,
        label_dir=args.val_label_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

    model = TMT_MS(
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        warp_mode="all",
        n_frames=2,
        dim=args.tmt_dims
    ).to(device)

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}, strict=False)
        print(f"✅ Loaded checkpoint: {args.resume_ckpt}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = CharbonnierLoss()
    wavelet_criterion = dwtLoss(loss_weight=args.lambda_dwt, wkd_level=1, wkd_basis='haar').to(device)
    temporal_criterion = nn.L1Loss()

    best_score = 0
    best_model_path = os.path.join(path_ckpt, "best_model.pth")
    latest_model_path = os.path.join(path_ckpt, "latest_model.pth")

    patience = 5
    no_improve_count = 0
    output_history = []
    charb_losses = []
    dwt_losses = []
    flow_losses = []
    yolo_losses = []
    total_losses = []
    
    
    if args.eval_only:
        model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []
        total_restore_time = 0  
        frame_count = 0         

        with torch.no_grad():
            for val_data_idx, val_data in enumerate(tqdm(val_loader)):
                full_seq, target_seq = val_data[0][0].to(device), val_data[1][0].to(device)
                O_prev = full_seq[0]
                for t in range(1, full_seq.shape[0]):
                    I_t = full_seq[t]
                    T_t = target_seq[t]

                    input_cat = torch.stack([O_prev, I_t], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)

                    start = time.time()  
                    output, _ = model(input_cat)
                    end = time.time()   

                    total_restore_time += end - start
                    frame_count += 1

                    output = output.squeeze(0)
                    if output.dim() == 4:
                        output = output[:, -1, :, :]
                    O_prev = output

                    out_np = output.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    gt_np = T_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

                    psnr_list.append(compare_psnr(gt_np, out_np, data_range=1.0))
                    ssim_list.append(compare_ssim(gt_np, out_np, data_range=1.0, channel_axis=-1))
                    
                    out_norm = (output * 2 - 1).clamp(-1, 1)
                    gt_norm = (T_t * 2 - 1).clamp(-1, 1)
                    lpips_val = lpips_loss_fn(out_norm, gt_norm).mean().item()
                    lpips_list.append(lpips_val)

                    if val_data_idx == 0:
                        output_dir_epoch = os.path.join(result_img_path, f"eval_only")
                        os.makedirs(output_dir_epoch, exist_ok=True)
                        save_image(I_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_input.png"))
                        save_image(output, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_output.png"))
                        save_image(T_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_gt.png"))

        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        mean_lpips = np.mean(lpips_list)
        avg_restore_time = total_restore_time / frame_count

        print(f"[EVAL ONLY] PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        print(f"[EVAL ONLY] Avg Restore Time: {avg_restore_time:.4f} sec/frame")
        
        logging.info(f"[EVAL ONLY] PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        logging.info(f"[EVAL ONLY] Avg Restore Time: {avg_restore_time:.4f} sec/frame")
        return

    for epoch in range(args.epochs):
        model.train()
        train_loss_list = []

        for data in tqdm(train_loader):
            full_seq, target_seq, meta = data[0][0].to(device), data[1][0].to(device), data[2][0]
            O_prev = full_seq[0]

            for t in range(1, full_seq.shape[0]):
                I_t = full_seq[t]
                T_t = target_seq[t]
                
                # max_t = full_seq.shape[0] - 1
                # alpha_t = 0.9
                # O_prev_blend = blend_prev_frame(O_prev, T_t, alpha_t)

                input_cat = torch.stack([O_prev, I_t], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
                output, flows = model(input_cat)
                output = output.squeeze(0)
                if output.dim() == 4:
                    output = output[:, -1, :, :]

                # if len(flows) > 0:
                #     flow_l1 = flows[0][:, :, 0, :, :]
                #     flow_l1 = F.interpolate(flow_l1, size=O_prev.shape[-2:], mode='bilinear', align_corners=True)
                #     warped_prev = flow_warp(O_prev.unsqueeze(0), flow_l1)
                #     flow_loss = temporal_criterion(output.unsqueeze(0), warped_prev)
                # else:
                #     flow_loss = 0.0
                
                if t > 1:
                    flow_loss = 0.0
                    for k in range(1, min(len(output_history)+1, len(flows)+1)):
                        prev_out = output_history[-k]  
                        flow_k = flows[k-1][:, :, 0, :, :]
                        flow_k = F.interpolate(flow_k, size=output.shape[-2:], mode='bilinear', align_corners=True)
                        warped = flow_warp(prev_out.unsqueeze(0), flow_k)
                        weight = 0.5 ** k
                        flow_loss += weight * F.l1_loss(output.unsqueeze(0), warped)
                else:
                    flow_loss = 0.0

                output_history.append(output.detach())
                if len(output_history) > 3:
                    output_history.pop(0)
                    
                lambda_flow = 0.1 + 0.2 * (epoch / args.epochs)

                charb_loss = criterion(output, T_t)
                dwt_loss = wavelet_criterion(output.unsqueeze(0), T_t.unsqueeze(0))
                    
                if t == full_seq.shape[0]-1:
                    output_resized = F.interpolate(output.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)
                    output_resized = output_resized.clamp(0, 1)
                    
                    video_id = meta["image_id"][0]
                    label_filename = f"{video_id:012d}.txt"
                    label_path = os.path.join(args.train_label_dir, label_filename)

                    if not os.path.exists(label_path):
                        print(f"[WARN] YOLO label not found: {label_path}")
                    batch_size = output_resized.shape[0]
                    targets = load_yolo_labels(label_path, output.shape[-2:], device)

                    with torch.no_grad():
                        pred_results = yolo_model.predict(output_resized, verbose=False)[0]

                    if pred_results.boxes is None or pred_results.boxes.xyxy is None:
                        pred = torch.zeros((0, 6)).to(device)
                    else:
                        pred_boxes = pred_results.boxes.xyxy  
                        pred_confs = pred_results.boxes.conf 
                        pred_cls = pred_results.boxes.cls 
            
                        pred = torch.cat([pred_boxes, pred_confs.unsqueeze(1), pred_cls.unsqueeze(1)], dim=1).to(device)
                    if pred is None:
                        pred = torch.zeros((0, 6)).to(device)

                    yolo_loss = person_detection_loss(pred, targets)
                    writer.add_scalar("Train/YOLO_Loss", yolo_loss.item(), epoch)
                    lambda_det = min(0.2 + 0.02 * epoch, 0.4)
                    loss = charb_loss + dwt_loss + lambda_flow * flow_loss + 0 * yolo_loss
                else:
                    loss = charb_loss + dwt_loss + lambda_flow * flow_loss

                train_loss_list.append(loss.item())
                charb_losses.append(charb_loss.item())
                dwt_losses.append(dwt_loss.item())
                flow_losses.append(flow_loss if isinstance(flow_loss, float) else flow_loss.item())
                if t == full_seq.shape[0]-1:
                    yolo_losses.append(yolo_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                N = 3

                if t % N == 0:
                    O_prev = I_t.detach()
                else:
                    O_prev = output.detach()

            if args.quick_val:
                break
            
        mean_train_loss = np.mean(train_loss_list)
        mean_charb = np.mean(charb_losses)
        mean_dwt = np.mean(dwt_losses)
        mean_flow = np.mean(flow_losses)
        mean_yolo = np.mean(yolo_losses) if yolo_losses else 0.0
        print(f"[Train@Epoch {epoch}] Avg charb: {mean_charb:.4f}, dwt: {mean_dwt:.4f}, flow: {mean_flow:.4f}, "f"yolo: {mean_yolo:.4f}, total: {mean_train_loss:.4f}")
        logging.info(f"[Train@Epoch {epoch}] Mean Loss: {mean_train_loss:.6f}")
        writer.add_scalar("Train/Loss", mean_train_loss, epoch)
        scheduler.step()


        model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []
        with torch.no_grad():
            for val_data_idx, val_data in enumerate(tqdm(val_loader)):
                full_seq, target_seq = val_data[0][0].to(device), val_data[1][0].to(device)
                O_prev = full_seq[0]
                for t in range(1, full_seq.shape[0]):
                    I_t = full_seq[t]
                    T_t = target_seq[t]

                    input_cat = torch.stack([O_prev, I_t], dim=0)  # [2, 3, H, W]
                    input_cat = input_cat.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, 2, H, W]
                    output, _ = model(input_cat)
                    output = output.squeeze(0)
                    if output.dim() == 4:
                        output = output[:, -1, :, :]
                    O_prev = output

                    out_np = output.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    gt_np = T_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

                    psnr_list.append(compare_psnr(gt_np, out_np, data_range=1.0))
                    ssim_list.append(compare_ssim(gt_np, out_np, data_range=1.0, channel_axis=-1))
                    
                    out_norm = (output * 2 - 1).clamp(-1, 1)  # [0,1] → [-1,1]
                    gt_norm = (T_t * 2 - 1).clamp(-1, 1)

                    lpips_val = lpips_loss_fn(out_norm, gt_norm).mean().item()
                    lpips_list.append(lpips_val)

                    if val_data_idx == 0:
                        output_dir_epoch = os.path.join(result_img_path, f"epoch{epoch}")
                        os.makedirs(output_dir_epoch, exist_ok=True)

                        save_image(I_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_input.png"))
                        save_image(output, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_output.png"))
                        save_image(T_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_gt.png"))

        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        mean_lpips = np.mean(lpips_list)
        current_score = mean_psnr + mean_ssim

        print(f"[EVAL ONLY] PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        logging.info(f"[EVAL ONLY] PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        writer.add_scalar("Val/PSNR", mean_psnr, epoch)
        writer.add_scalar("Val/SSIM", mean_ssim, epoch)
        writer.add_scalar("Val/LPIPS", mean_lpips, 0)

        if current_score > best_score:
            best_score = current_score
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, best_model_path)
            print(f"✅ Best model saved at epoch {epoch}")
            logging.info(f"✅ Best model saved at epoch {epoch}")
            
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, latest_model_path)
        logging.info(f"💾 Latest model saved for epoch {epoch}")

if __name__ == "__main__":
    main()
