import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from model.TMT_DC import TMT_MS
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import subprocess
import lpips
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="/root/autodl-tmp/MitigationDetection/car_distorted_frames_old/car_distorted.mp4")
parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/MitigationDetection/car_distorted_frames")
parser.add_argument("--restoration_ckpt", type=str, default="/root/autodl-tmp/MitigationDetection/log/RecursiveTrain_WithYOLO_07-12-2025-01-10-09/checkpoints/best_model1.pth")
parser.add_argument("--save_raw", action="store_true", help="Save original frames")
parser.add_argument("--resize_hw", type=int, nargs=2, default=(512, 512))

parser.add_argument("--tmt_dim", type=int, default=16, help="Feature dimension for TMT_MS model")
parser.add_argument("--warp_mode", type=str, default="all", help="Warp mode for TMT_MS (e.g., 'all', 'enc', etc.)")
parser.add_argument("--n_frames", type=int, default=2, help="Number of input frames for the model")
args = parser.parse_args()


def pad_to_multiple(img, multiple=8):
    h, w, _ = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded


os.makedirs(args.output_dir, exist_ok=True)
if args.save_raw:
    os.makedirs(os.path.join(args.output_dir, "original"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
psnr_list, ssim_list, lpips_list = [], [], []
frame_count = 0
total_restore_time = 0

lpips_loss_fn = lpips.LPIPS(net='alex').to(device)


restoration_model = TMT_MS(
    num_blocks=[2, 3, 3, 4],
    num_refinement_blocks=2,
    warp_mode=args.warp_mode,
    n_frames=args.n_frames,
    dim=args.tmt_dim
).to(device)

ckpt = torch.load(args.restoration_ckpt, map_location=device)
restoration_model.load_state_dict(ckpt["state_dict"])
restoration_model.eval()


cap = cv2.VideoCapture(args.video_path)
frames = []
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = pad_to_multiple(frame, multiple=8)
    frame = frame.astype(np.float32) / 255.0
    frames.append(frame)
cap.release()

print(f" total {len(frames)} frames , start two frames recursive restored...")

first_frame = frames[0]
input_pair = [first_frame, first_frame]
clip_tensor = torch.from_numpy(np.stack(input_pair)).permute(0, 3, 1, 2).unsqueeze(0).to(device)

with torch.no_grad():
    output, _ = restoration_model(clip_tensor.permute(0, 2, 1, 3, 4))
    output = output.permute(0, 2, 1, 3, 4)

restored = output[0, 1].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
prev_restored = restored

Image.fromarray((restored * 255).astype(np.uint8)).save(os.path.join(args.output_dir, "restored_0000.jpg"))
if args.save_raw:
    Image.fromarray((first_frame * 255).astype(np.uint8)).save(os.path.join(args.output_dir, "original", "original_0000.jpg"))


for i in tqdm(range(1, len(frames))):
    current_distorted = frames[i]
    alpha_t = 1.0
    blended_prev = alpha_t * prev_restored + (1 - alpha_t) * current_distorted
    input_pair = [blended_prev, current_distorted]
    clip_tensor = torch.from_numpy(np.stack(input_pair)).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        output, _ = restoration_model(clip_tensor.permute(0, 2, 1, 3, 4))
        output = output.permute(0, 2, 1, 3, 4)
        total_restore_time += time.time() - start

    restored = output[0, 1].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    prev_restored = restored 

    restored_tensor = torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(current_distorted.transpose(2, 0, 1)).unsqueeze(0).to(device)
    restored_tensor = (restored_tensor * 2 - 1).clamp(-1, 1)
    gt_tensor = (gt_tensor * 2 - 1).clamp(-1, 1)

    lpips_list.append(lpips_loss_fn(restored_tensor, gt_tensor).mean().item())

    Image.fromarray((restored * 255).astype(np.uint8)).save(os.path.join(args.output_dir, f"restored_{i:04d}.jpg"))
    if args.save_raw:
        Image.fromarray((current_distorted * 255).astype(np.uint8)).save(os.path.join(args.output_dir, "original", f"original_{i:04d}.jpg"))

    psnr_list.append(compare_psnr(current_distorted, restored, data_range=1.0))
    ssim_list.append(compare_ssim(current_distorted, restored, data_range=1.0, channel_axis=-1))
    frame_count += 1


print(f"\n✅ Done. Restored {frame_count + 1} frames (including first frame).")
print(f"⏱️ Avg restore time: {total_restore_time / frame_count:.4f} sec/frame")
print(f"📈 Mean PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.4f}, LPIPS: {np.mean(lpips_list):.4f}")

output_video = os.path.join(args.output_dir, "car_distorted.mp4")
subprocess.run([
    "ffmpeg", "-framerate", "25",
    "-i", os.path.join(args.output_dir, "restored_%04d.jpg"),
    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
    output_video
])
print(f"🎬 restoration video saved to: {output_video}")
