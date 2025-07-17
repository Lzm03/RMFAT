# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            # normalize in channel dimension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)
        return loss

class MGDLoss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(t_chan, t_chan, kernel_size=3, padding=1)
            ).to(device) for s_chan, t_chan in zip(student_channels, teacher_channels)
        ])


    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss
    
class KLLoss(nn.Module):
    """
    KL Divergence Loss between teacher and student feature maps with channel alignment.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(KLLoss, self).__init__()
        self.tau = tau
        self.align = nn.ModuleList()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for idx, (s_c, t_c) in enumerate(zip(channels_s, channels_t)):
            print(f"[KLLoss] Align Layer {idx}: Conv({t_c} → {s_c})")
            self.align.append(nn.Conv2d(t_c, s_c, kernel_size=1).to(device))

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t), f"[KLLoss] Feature count mismatch: student={len(y_s)}, teacher={len(y_t)}"
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            t = self.align[idx](t)  # ✅ 将 teacher 特征对齐到 student 的通道数

            s_log = F.log_softmax(s.view(s.size(0), s.size(1), -1) / self.tau, dim=2)
            t_soft = F.softmax(t.view(t.size(0), t.size(1), -1) / self.tau, dim=2)

            assert s_log.shape == t_soft.shape, f"[KLLoss] Shape mismatch: s_log={s_log.shape}, t_soft={t_soft.shape}"

            kl = F.kl_div(s_log, t_soft, reduction='batchmean') * (self.tau ** 2)
            losses.append(kl)

        return sum(losses)




class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        
        # Move all modules to same precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Convert to ModuleList and ensure consistent dtype
        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        
        # Create alignment modules
        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)
            
        # Create normalization layers
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
            
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))

        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        elif distiller == 'kl':
            self.feature_loss = KLLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        min_len = min(len(y_s), len(y_t))
        y_s = y_s[:min_len]
        y_t = y_t[:min_len]

        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # Match input dtype to module dtype
            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)
            
            if self.distiller in ["cwd", "kl"]:
                # Apply alignment and normalization
                s = self.align_module[idx](s)
                stu_feats.append(s)
                tea_feats.append(t.detach())
            else:
                # Apply normalization
                t = self.norm[idx](t)
                stu_feats.append(s)
                tea_feats.append(t.detach())

        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss
    
    
# class FeatureLoss(nn.Module):
#     def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
#         super(FeatureLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.distiller = distiller
        
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         # ✅ 只有 MGDLoss 需要通道对齐模块
#         if distiller == 'mgd':
#             self.align_module = nn.ModuleList()
#             for s_chan, t_chan in zip(channels_s, channels_t):
#                 align = nn.Sequential(
#                     nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
#                     nn.BatchNorm2d(t_chan, affine=False)
#                 ).to(device)
#                 self.align_module.append(align)

#             self.norm = nn.ModuleList()
#             for t_chan in channels_t:
#                 self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
        
#         # ✅ CWD/KL 不需要额外 align_module，它们内部自己处理
#         if distiller == 'mgd':
#             self.feature_loss = MGDLoss(channels_s, channels_t)
#         elif distiller == 'cwd':
#             self.feature_loss = CWDLoss(channels_s, channels_t)
#         elif distiller == 'kl':
#             self.feature_loss = KLLoss(channels_s, channels_t)
#         else:
#             raise NotImplementedError(f"Unknown distiller: {distiller}")

#     def forward(self, y_s, y_t):
#         min_len = min(len(y_s), len(y_t))
#         y_s = y_s[:min_len]
#         y_t = y_t[:min_len]

#         tea_feats = []
#         stu_feats = []

#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             if self.distiller == "mgd":
#                 s = s.type(next(self.align_module[idx].parameters()).dtype)
#                 t = t.type(next(self.align_module[idx].parameters()).dtype)
#                 t = self.norm[idx](t)
#                 stu_feats.append(s)
#                 tea_feats.append(t.detach())
#             else:
#                 # CWD or KL
#                 # 不对齐通道，不做归一化，由各自 loss 内部处理
#                 stu_feats.append(s)
#                 tea_feats.append(t.detach())

#         loss = self.feature_loss(stu_feats, tea_feats)
#         return self.loss_weight * loss



class DistillationLoss:
    def __init__(self, models, modelt, distiller="CWDLoss"):
        self.distiller = distiller
        self.layers = ["6", "8", "13", "16", "19", "22"]
        self.models = models 
        self.modelt = modelt

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ini warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            _ = self.models(dummy_input.to(device))
            _ = self.modelt(dummy_input.to(device))
        
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []
        
        self._find_layers()
        
        self.distill_loss_fn = FeatureLoss(
            channels_s=self.channels_s, 
            channels_t=self.channels_t, 
            distiller=distiller[:3]
        )
        
    def _find_layers(self):

        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        
        for name, ml in self.modelt.named_modules():
            if name is not None:
                name = name.split(".")
                # print(name)
                
                if name[0] != "model":
                    continue
                if len(name) >= 3:
                    if name[1] in self.layers:
                        if "cv2" in name[2]:
                            if hasattr(ml, 'conv'):
                                self.channels_t.append(ml.conv.out_channels)
                                self.teacher_module_pairs.append(ml)
        # print()
        for name, ml in self.models.named_modules():
            if name is not None:
                name = name.split(".")
                # print(name)
                if name[0] != "model":
                    continue
                if len(name) >= 3:
                    if name[1] in self.layers:
                        if "cv2" in name[2]:
                            if hasattr(ml, 'conv'):
                                self.channels_s.append(ml.conv.out_channels)
                                self.student_module_pairs.append(ml)

        nl = min(len(self.channels_s), len(self.channels_t))
        self.channels_s = self.channels_s[-nl:]
        self.channels_t = self.channels_t[-nl:]
        self.teacher_module_pairs = self.teacher_module_pairs[-nl:]
        self.student_module_pairs = self.student_module_pairs[-nl:]

    def register_hook(self):
        # Remove the existing hook if they exist
        self.remove_handle_()
        
        self.teacher_outputs = []
        self.student_outputs = []

        def make_student_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    out = output.clone()  # Clone to ensure we don't modify the original
                    l.append(out)
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.detach().clone())  # Detach and clone teacher outputs
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))

    def get_loss(self):
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, requires_grad=True)
        
        if len(self.teacher_outputs) != len(self.student_outputs):
            print(f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.student_outputs)}")
            return torch.tensor(0.0, requires_grad=True)
        
        quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs)
        
        if self.distiller != 'cwd':
            quant_loss *= 0.3

        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()