#!/bin/bash

# python /root/autodl-tmp/MitigationDetection/joint_distill.py \
#   --batch_size 1 \
#   --patch_size 256 \
#   --train_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/train2017carperson" \
#   --val_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/val2017all" \
#   --log_path "/root/autodl-tmp/MitigationDetection/log" \
#   --teacher_ckpt "/root/autodl-tmp/MitigationDetection/TMT_default_mamba_12-13-2024-13-35-51/checkpoints/turb_model_best.pth" \
#   --yolo_teacher "/root/autodl-tmp/MitigationDetection/yolo11l.pt" \
#   --yolo_student "/root/autodl-tmp/MitigationDetection/yolo11n.pt" \
#   --train_label_dir "/root/autodl-tmp/MitigationDetection/labels/train" \
#   --train_im_dir "/root/autodl-tmp/MitigationDetection/images/train2017carperson" \
#   --val_label_dir "/root/autodl-tmp/MitigationDetection/labels/val" \
#   --val_im_dir "/root/autodl-tmp/MitigationDetection/images/val2017all" \
#   --eval

  # python recursive_train.py \
  # --train_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/traindyn" \
  # --val_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/valdyn" \
  # --batch_size 1 \
  # --patch_size 256 \
  # --num_frames 20 \
  # --tmt_dims 16 \
  # --log_path "/root/autodl-tmp/MitigationDetection/log" \
  # --run_name RecursiveTrain \
  # --resume_ckpt "/root/autodl-tmp/MitigationDetection/log/RecursiveTrain_05-29-2025-11-48-17/checkpoints/best_model.pth" \
  # # --quick_val

CUDA_VISIBLE_DEVICES=0 python recursive_train.py \
  --train_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/traindyn" \
  --val_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/valdyn" \
  --train_label_dir "/root/autodl-tmp/MitigationDetection/labels/train" \
  --val_label_dir "/root/autodl-tmp/MitigationDetection/labels/val" \
  --batch_size 1 \
  --patch_size 256 \
  --num_frames 10 \
  --tmt_dims 16 \
  --log_path "/root/autodl-tmp/MitigationDetection/test" \
  --run_name test \
  --resume_ckpt "/root/autodl-tmp/MitigationDetection/log/RecursiveTrain_WithYOLO_07-12-2025-01-10-09/checkpoints/best_model1.pth" \
  --eval_only

python joint_distill_recursive.py \
  --train_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/traindyn" \
  --val_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/valdyn" \
  --teacher_ckpt "/root/autodl-tmp/MitigationDetection/log/RecursiveTrain_GPU0_06-02-2025-11-10-27/checkpoints/best_model.pth" \
  --batch_size 1 \
  --patch_size 256 \
  --num_frames 10 \
  --tmt_dims 16 \
  --log_path "/root/autodl-tmp/MitigationDetection/log" \
  --run_name RecursiveTrain_student \
  # --quick_val


CUDA_VISIBLE_DEVICES=0 python single_train.py \
  --train_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/traindyn" \
  --val_path "/root/autodl-tmp/MitigationDetection/STATIC_GT_P2S_MP4S/valdyn" \
  --train_label_dir "/root/autodl-tmp/MitigationDetection/labels/train" \
  --val_label_dir "/root/autodl-tmp/MitigationDetection/labels/val" \
  --batch_size 1 \
  --patch_size 256 \
  --num_frames 10 \
  --tmt_dims 16 \
  --log_path "/root/autodl-tmp/MitigationDetection/test" \
  --run_name test \
  --resume_ckpt "/root/autodl-tmp/MitigationDetection/test/test_07-27-2025-22-40-39/checkpoints/checkpoints/best_model.pth" \
  --eval_only