# import os
# import random

# import cv2
# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms.functional as TF
# from PIL import Image
# from torch.utils.data import Dataset
# from .coco import CocoDetection


# class DataLoaderTurbVideo(Dataset):
#     """Data loader for turbulent video mitigation

#     Args:
#         Dataset (Abstract dataloader): torch.utils.data dataloader base class
#     """

#     def __init__(
#         self, root_dir, ann_file, im_dir, num_frames=12, patch_size=None, noise=None, is_train=True
#     ):
#         super(DataLoaderTurbVideo, self).__init__()
#         self.dataset = CocoDetection(im_dir, ann_file, is_train)
        
#         self.num_frames = num_frames
#         self.turb_list = []
#         self.blur_list = []
#         self.gt_list = []
#         for v in os.listdir(os.path.join(root_dir, "gt")):
#             self.gt_list.append(os.path.join(root_dir, "gt", v))
#             self.turb_list.append(os.path.join(root_dir, "turb", v))

#         self.gt_list.sort()
#         self.turb_list.sort()
#         self.ps = patch_size
#         self.sizex = len(self.gt_list)  # get the size of target
#         self.train = is_train
#         self.noise = noise

#         for idx in range(self.sizex):
#             this_id = self.dataset[idx][1]['image_id'][0]
#             _, file_name_gt = os.path.split(self.gt_list[idx])                                                                                                                                                                                                                                                 
#             file_number_gt = int(file_name_gt[:-4])
#             _, file_name_turb = os.path.split(self.turb_list[idx])                                                                                                                                                                                                                                                 
#             file_number_turb = int(file_name_turb[:-4])
#             if not (this_id == file_number_gt == file_number_turb):
#                 print("datasets don't match")
#                 exit()


#     def __len__(self):
#         return self.sizex

#     def _inject_noise(self, img, noise):
#         noise = (noise**0.5) * torch.randn(img.shape)
#         out = img + noise
#         return out.clamp(0, 1)

#     def _fetch_chunk_val(self, idx):
                                                                                                                                             
#         _, this_target = self.dataset[idx] 

#         turb_vid = cv2.VideoCapture(self.turb_list[idx])
#         gt_vid = cv2.VideoCapture(self.gt_list[idx])
#         total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total_frames < self.num_frames:
#             print("no enough frame in video " + self.gt_list[idx])
#         start_frame_id = (total_frames - self.num_frames) // 2

#         # load frames from video
#         gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
#         turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
#         tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
#         turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
#         turb_vid.release()
#         gt_vid.release()

#         tar_imgs = [
#             Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs
#         ]
#         turb_imgs = [
#             Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs
#         ]

#         turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
#         tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

#         if self.noise:
#             noise_level = self.noise * random.random()
#             turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
#         return turb_imgs, tar_imgs, this_target 

#     def _fetch_chunk_train(self, idx):
#         _, this_target = self.dataset[idx] 
#         ps = self.ps
#         turb_vid = cv2.VideoCapture(self.turb_list[idx])
#         gt_vid = cv2.VideoCapture(self.gt_list[idx])
#         h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
#         total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total_frames < self.num_frames:
#             print("no enough frame in video " + self.gt_list[idx])
#             # return self._fetch_chunk_train(idx + 1)
#         start_frame_id = random.randint(0, total_frames - self.num_frames)

#         # load frames from video
#         gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
#         turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
#         tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
#         turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
#         if tar_imgs[0] is None:
#             print(self.gt_list[idx])

#         tar_imgs = [
#             Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs
#         ]
#         turb_imgs = [
#             Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs
#         ]
#         padw = ps - w if w < ps else 0
#         padh = ps - h if h < ps else 0
#         if padw != 0 or padh != 0:
#             turb_imgs = [
#                 TF.pad(img, (0, 0, padw, padh), padding_mode="reflect")
#                 for img in turb_imgs
#             ]
#             tar_imgs = [
#                 TF.pad(img, (0, 0, padw, padh), padding_mode="reflect")
#                 for img in tar_imgs
#             ]

#         aug = random.randint(0, 2)
#         if aug == 1:
#             turb_imgs = [TF.adjust_gamma(img, 1) for img in turb_imgs]
#             tar_imgs = [TF.adjust_gamma(img, 1) for img in tar_imgs]

#         aug = random.randint(0, 2)
#         if aug == 1:
#             sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
#             turb_imgs = [TF.adjust_saturation(img, sat_factor) for img in turb_imgs]
#             tar_imgs = [TF.adjust_saturation(img, sat_factor) for img in tar_imgs]

#         turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
#         tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

#         if self.noise:
#             noise_level = self.noise * random.random()
#             turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]

#         return turb_imgs, tar_imgs, this_target 

#     def __getitem__(self, index):
#         index_ = index % self.sizex
#         if self.train:
#             turb_imgs, tar_imgs, this_target = self._fetch_chunk_train(index_)
#         else:
#             turb_imgs, tar_imgs, this_target = self._fetch_chunk_val(index_)
#         return torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), this_target


# from torch.utils.data import Dataset
# import os
# import torch
# import torchvision.transforms.functional as TF
# from PIL import Image
# import numpy as np
# import random
# import cv2


# class DataLoaderTurbVideo(Dataset):
#     def __init__(self, root_dir, label_dir, num_frames=12, patch_size=None, noise=None, is_train=True):
#         super().__init__()
#         self.num_frames = num_frames
#         self.patch_size = patch_size
#         self.noise = noise
#         self.train = is_train

#         self.gt_list = sorted([os.path.join(root_dir, "gt", v) for v in os.listdir(os.path.join(root_dir, "gt"))])
#         self.turb_list = sorted([os.path.join(root_dir, "turb", v) for v in os.listdir(os.path.join(root_dir, "turb"))])
#         self.label_dir = label_dir

#         self.image_ids = []
#         for gt_path in self.gt_list:
#             filename = os.path.basename(gt_path)
#             image_id = int(filename.split('.')[0])
#             label_path = os.path.join(label_dir, f"{image_id:012d}.txt")
#             if os.path.exists(label_path):
#                 self.image_ids.append(image_id)
#             else:
#                 print(f"⚠️ Warning: Label file missing for {image_id}, skipping.")
        
#         self.gt_list = [os.path.join(root_dir, "gt", f"{img_id:012d}.mp4") for img_id in self.image_ids]
#         self.turb_list = [os.path.join(root_dir, "turb", f"{img_id:012d}.mp4") for img_id in self.image_ids]

#     def __len__(self):
#         return len(self.image_ids)

#     def _inject_noise(self, img, noise_level):
#         noise = (noise_level**0.5) * torch.randn(img.shape)
#         return (img + noise).clamp(0, 1)

#     def _load_video_frames(self, path, start_frame, num_frames):
#         cap = cv2.VideoCapture(path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#         frames = [cap.read()[1] for _ in range(num_frames)]
#         cap.release()
#         return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

#     def _load_labels(self, image_id):
#         label_path = os.path.join(self.label_dir, f"{image_id:012d}.txt")
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 lines = f.readlines()
#             targets = []
#             for line in lines:
#                 cls, xc, yc, w, h = map(float, line.strip().split())
#                 targets.append([0, cls, xc, yc, w, h])  # batch_idx = 0
#             return torch.tensor(targets, dtype=torch.float32)
#         return torch.empty((0, 6), dtype=torch.float32)

#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
#         turb_path = self.turb_list[idx]
#         gt_path = self.gt_list[idx]

#         cap = cv2.VideoCapture(gt_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         h, w = int(cap.get(4)), int(cap.get(3))
#         cap.release()

#         if self.train:
#             start_frame = random.randint(0, total_frames - self.num_frames)
#         else:
#             start_frame = (total_frames - self.num_frames) // 2

#         turb_imgs = self._load_video_frames(turb_path, start_frame, self.num_frames)
#         gt_imgs = self._load_video_frames(gt_path, start_frame, self.num_frames)

#         # Resize/Padding
#         if self.patch_size:
#             padw = self.patch_size - w if w < self.patch_size else 0
#             padh = self.patch_size - h if h < self.patch_size else 0
#             if padw != 0 or padh != 0:
#                 turb_imgs = [TF.pad(im, (0, 0, padw, padh), padding_mode="reflect") for im in turb_imgs]
#                 gt_imgs = [TF.pad(im, (0, 0, padw, padh), padding_mode="reflect") for im in gt_imgs]

#         # Augmentation
#         if self.train:
#             if random.random() < 0.5:
#                 turb_imgs = [TF.adjust_gamma(im, gamma=1.2) for im in turb_imgs]
#                 gt_imgs = [TF.adjust_gamma(im, gamma=1.2) for im in gt_imgs]
#             if random.random() < 0.5:
#                 sat = 1 + (0.2 - 0.4 * np.random.rand())
#                 turb_imgs = [TF.adjust_saturation(im, sat) for im in turb_imgs]
#                 gt_imgs = [TF.adjust_saturation(im, sat) for im in gt_imgs]

#         turb_imgs = [TF.to_tensor(im) for im in turb_imgs]
#         gt_imgs = [TF.to_tensor(im) for im in gt_imgs]

#         if self.noise:
#             noise_level = self.noise * random.random()
#             turb_imgs = [self._inject_noise(im, noise_level) for im in turb_imgs]

#         targets = self._load_labels(image_id)

#         return torch.stack(turb_imgs), torch.stack(gt_imgs), {'image_id': [image_id], 'labels': targets}



import os
import cv2
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class DataLoaderTurbVideo(Dataset):
    def __init__(
        self, root_dir, label_dir=None, num_frames=10, patch_size=None,
        noise=None, is_train=True, clip_stride=1, max_clips_per_video=None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.noise = noise
        self.train = is_train
        self.clip_stride = clip_stride
        self.max_clips_per_video = max_clips_per_video

        self.turb_list = sorted([os.path.join(root_dir, "turb", f) for f in os.listdir(os.path.join(root_dir, "turb")) if f.endswith(".mp4")])
        self.gt_list = sorted([os.path.join(root_dir, "gt", f) for f in os.listdir(os.path.join(root_dir, "gt")) if f.endswith(".mp4")])

        self.indices = []  # list of (video_index, start_frame)

        for video_idx, turb_path in enumerate(self.turb_list):
            cap = cv2.VideoCapture(turb_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            num_possible_clips = max(0, total_frames - num_frames + 1)
            start_indices = list(range(0, num_possible_clips, clip_stride))
            if not self.train:
                start_indices = [0]
            elif self.max_clips_per_video is not None:
                start_indices = start_indices[:self.max_clips_per_video]
            for start in start_indices:
                self.indices.append((video_idx, start))
                
        if len(self.indices) == 0:
            raise RuntimeError(f"No valid clips found! Check num_frames ({self.num_frames}) and video lengths.")

    def __len__(self):
        return len(self.indices)

    def _inject_noise(self, img, noise_level):
        noise = (noise_level ** 0.5) * torch.randn_like(img)
        return (img + noise).clamp(0, 1)

    def _load_video_clip(self, path, start_frame):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
        cap.release()
        return frames

    def __getitem__(self, index):
        video_idx, start_frame = self.indices[index]
        turb_path = self.turb_list[video_idx]
        gt_path = self.gt_list[video_idx]
        image_id = int(os.path.basename(turb_path).split(".")[0])

        turb_imgs = self._load_video_clip(turb_path, start_frame)
        gt_imgs = self._load_video_clip(gt_path, start_frame)

        # Resize / Padding
        if self.patch_size:
            w, h = turb_imgs[0].size
            padw = self.patch_size - w if w < self.patch_size else 0
            padh = self.patch_size - h if h < self.patch_size else 0
            if padw or padh:
                turb_imgs = [TF.pad(im, (0, 0, padw, padh), padding_mode="reflect") for im in turb_imgs]
                gt_imgs = [TF.pad(im, (0, 0, padw, padh), padding_mode="reflect") for im in gt_imgs]

        # Transform to tensor
        turb_tensors = [TF.to_tensor(img) for img in turb_imgs]
        gt_tensors = [TF.to_tensor(img) for img in gt_imgs]

        # Noise
        if self.noise:
            noise_level = self.noise * random.random()
            turb_tensors = [self._inject_noise(img, noise_level) for img in turb_tensors]

        turb_tensor = torch.stack(turb_tensors, dim=0)  # [T, C, H, W]
        gt_tensor = torch.stack(gt_tensors, dim=0)      # [T, C, H, W]

        meta = {
            "image_id": [image_id],
            "center_idx": start_frame + self.num_frames // 2,
            "video_idx": video_idx,
            "start_frame": start_frame
        }
        return turb_tensor, gt_tensor, meta