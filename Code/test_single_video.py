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


def pad_to_multiple(img, multiple=8):
    h, w, _ = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded

video_path = "/root/autodl-tmp/MitigationDetection/car_distorted_frames_old/car_distorted.mp4"
output_dir = "/root/autodl-tmp/MitigationDetection/car_distorted_frames"
save_raw = True

os.makedirs(output_dir, exist_ok=True)
if save_raw:
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_hw = (512, 512)
psnr_list, ssim_list, lpips_list = [], [], []
frame_count = 0
total_restore_time = 0

restoration_ckpt = "/root/autodl-tmp/MitigationDetection/log/RecursiveTrain_WithYOLO_07-12-2025-01-10-09/checkpoints/best_model1.pth"
lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

restoration_model = TMT_MS(
    num_blocks=[2, 3, 3, 4],
    num_refinement_blocks=2,
    warp_mode="all",
    n_frames=2,
    dim=16
).to(device)

ckpt = torch.load(restoration_ckpt, map_location=device)
restoration_model.load_state_dict(ckpt["state_dict"])
restoration_model.eval()

cap = cv2.VideoCapture(video_path)
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


restored_uint8 = (restored * 255).astype(np.uint8)
Image.fromarray(restored_uint8).save(os.path.join(output_dir, "restored_0000.jpg"))

if save_raw:
    original_uint8 = (first_frame * 255).astype(np.uint8)
    Image.fromarray(original_uint8).save(os.path.join(output_dir, "original", "original_0000.jpg"))

for i in tqdm(range(1, len(frames))):
    current_distorted = frames[i]

    # alpha_t = max(1.0 - i / len(frames), 0.1)
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
    

    restored_tensor = torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_tensor = torch.from_numpy(current_distorted.transpose(2, 0, 1)).unsqueeze(0).to(device)

    restored_tensor = (restored_tensor * 2 - 1).clamp(-1, 1)
    gt_tensor = (gt_tensor * 2 - 1).clamp(-1, 1)

    lpips_val = lpips_loss_fn(restored_tensor, gt_tensor).mean().item()
    lpips_list.append(lpips_val)


    restored_uint8 = (restored * 255).astype(np.uint8)
    filename = f"restored_{i:04d}.jpg"
    Image.fromarray(restored_uint8).save(os.path.join(output_dir, filename))

    if save_raw:
        original_uint8 = (current_distorted * 255).astype(np.uint8)
        rawname = f"original_{i:04d}.jpg"
        Image.fromarray(original_uint8).save(os.path.join(output_dir, "original", rawname))

    psnr_list.append(compare_psnr(current_distorted, restored, data_range=1.0))
    ssim_list.append(compare_ssim(current_distorted, restored, data_range=1.0, channel_axis=-1))
    frame_count += 1
    
    
print(f"\n✅ Done. Restored {frame_count + 1} frames (including first frame).")
print(f"⏱️ Avg restore time: {total_restore_time / frame_count:.4f} sec/frame")
print(f"📈 Mean PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.4f}, LPIPS: {np.mean(lpips_list):.4f}")

output_video = os.path.join(output_dir, "car_distorted.mp4")
ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "25",
    "-i", os.path.join(output_dir, "restored_%04d.jpg"),
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    output_video
]

subprocess.run(ffmpeg_cmd)
print(f"🎬 restoration video save to: {output_video}")
