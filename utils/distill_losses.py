import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops



class SimpleDetectionLoss(nn.Module):
    def __init__(self, box_weight=0.05, obj_weight=1.0, cls_weight=0.5, num_classes=80):
        super().__init__()
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, yolo_result, gt_boxes, gt_labels):
        if len(yolo_result.boxes) == 0 or len(gt_boxes) == 0:
            return torch.tensor(0.0, device=gt_boxes.device)

        pred_boxes = yolo_result.boxes.xyxy  # [N, 4]
        pred_obj = yolo_result.boxes.conf.unsqueeze(1)  # [N, 1]
        pred_cls = yolo_result.boxes.cls.long()  # [N]

        # IoU matching
        ious = ops.box_iou(pred_boxes, gt_boxes)  # [N, M]
        best_iou, best_gt_idx = ious.max(dim=1)  # [N]

        # Box loss
        box_loss = (1.0 - best_iou).mean()

        # Objectness loss
        obj_target = (best_iou > 0.5).float().unsqueeze(1)
        obj_loss = self.bce(pred_obj.to(gt_boxes.device), obj_target)

        # Class loss (simulated logits)
        matched_gt_labels = gt_labels[best_gt_idx].long()
        fake_logits = F.one_hot(pred_cls, num_classes=self.num_classes).float()
        fake_logits = fake_logits + 1e-6  # prevent log(0)
        cls_loss = self.ce(fake_logits.log(), matched_gt_labels)

        total_loss = (
            self.box_weight * box_loss +
            self.obj_weight * obj_loss +
            self.cls_weight * cls_loss
        )

        return total_loss

def kl_div_loss(student, teacher, T=4.0):
    s_log_prob = F.log_softmax(student / T, dim=1)
    t_prob = F.softmax(teacher / T, dim=1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)

class MGDLoss(nn.Module):
    """
    Mask Guided Distillation Loss
    Reference: "Mask Guided Knowledge Distillation" (CVPR 2020)
    """
    def __init__(self, alpha=0.00002):
        super(MGDLoss, self).__init__()
        self.alpha = alpha

    def forward(self, student_feat, teacher_feat):
        # Feature shape: (B, T, C, H, W)
        # We reshape into (B*T, C, H, W)
        B, T, C, H, W = student_feat.shape
        student_feat = student_feat.reshape(B*T, C, H, W)
        teacher_feat = teacher_feat.reshape(B*T, C, H, W)

        # Create spatial mask based on teacher features
        with torch.no_grad():
            mask = torch.norm(teacher_feat, dim=1, keepdim=True)  # (B*T, 1, H, W)
            mask = mask / (mask.max() + 1e-8)  # normalize to [0,1]

        loss = self.alpha * torch.mean((student_feat - teacher_feat) ** 2 * mask)
        return loss


class CWDLoss(nn.Module):
    """
    Channel-wise Distillation Loss
    Reference: "Distilling the Knowledge in a Neural Network" (Hinton et al.)
    Modified for feature map channel-wise matching.
    """
    def __init__(self, temperature=1.0):
        super(CWDLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_feat, teacher_feat):
        B, T, C, H, W = student_feat.shape
        student_feat = student_feat.reshape(B*T, C, -1)  # (B*T, C, H*W)
        teacher_feat = teacher_feat.reshape(B*T, C, -1)

        # Normalize across spatial dimension
        student_log_softmax = F.log_softmax(student_feat / self.temperature, dim=-1)
        teacher_softmax = F.softmax(teacher_feat / self.temperature, dim=-1)

        loss = self.kl_div(student_log_softmax, teacher_softmax) * (self.temperature ** 2)
        return loss