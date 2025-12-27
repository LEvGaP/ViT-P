import torch
import os

# from torchvision.io import read_imagez
from PIL import Image 
import torchvision
import torchvision.transforms.v2 as transforms
# from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import pandas as pd
import cv2
from numpy import random
import glob
import numpy as np
import pickle
import json
from dotenv import load_dotenv


class ADE20k_P(Dataset):
    def __init__(self, split: str, n_points: int, image_size: int):
        super().__init__()

        self.root = "/home6/m_imm_freedata/Segmentation/Projects/ADEChallengeData2016"

        if not os.path.isdir(self.root):
            raise RuntimeError(f"ADE20K_ROOT does not exist: {self.root}")

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.split = "training" if split == "train" else "validation"

        images_dir = os.path.join(self.root, "images", self.split)
        annotations_dir = os.path.join(self.root, "annotations", self.split)

        if not os.path.isdir(images_dir):
            raise RuntimeError(f"Images directory not found: {images_dir}")

        if not os.path.isdir(annotations_dir):
            raise RuntimeError(f"Annotations directory not found: {annotations_dir}")

        if split == "train":
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                    antialias=True,
                ),
                transforms.RandomRotation((-60, 60)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augmentation = None
        
        all_images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        if len(all_images) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

        valid_images = []
        empty_masks = 0

        for img_pth in all_images:
            fname = os.path.basename(img_pth).replace(".jpg", ".png")
            mask_pth = os.path.join(annotations_dir, fname)

            if not os.path.isfile(mask_pth):
                continue

            mask = np.array(Image.open(mask_pth))

            if np.all(mask == 0):
                empty_masks += 1
                continue

            valid_images.append(img_pth)

        if len(valid_images) == 0:
            raise RuntimeError("All images were filtered out due to empty masks")

        print(
            f"[ADE20k_P] Split: {self.split} | "
            f"Total images: {len(all_images)} | "
            f"Removed empty masks: {empty_masks} | "
            f"Remaining: {len(valid_images)}"
        )

        self.root_dir = valid_images
        self.n_points = n_points

        self.transform = torchvision.transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        img_pth = self.root_dir[idx]
        image = Image.open(img_pth).convert('RGB')
        splited = img_pth.split('/')
        splited = splited[-1].split('.')
        
        msk_pth = os.path.join(
            self.root,
            "annotations",
            self.split,
            splited[0] + ".png"
        )
        
        # mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
        mask_np = Image.open(msk_pth)
        
        if self.augmentation:
            image0,mask_np0 = self.augmentation(image, mask_np)
            
            mask_np0 = np.array(mask_np0)
            object_numbers0 = np.unique(mask_np0.reshape(-1), axis=0)
    
            if len(object_numbers0)==1 and (0 in object_numbers0):
                mask_np = np.array(mask_np)
            else:
                image = image0
                mask_np = mask_np0

        else:
            mask_np = np.array(mask_np)


        
        object_numbers = np.unique(mask_np.reshape(-1), axis=0)
        if self.transform:
            image = self.transform(image)
        

        # if len(object_numbers) > 1 :
        if 0 in object_numbers:
            object_numbers = np.delete(object_numbers,np.where(object_numbers==0))    
        
        # try:
        x = np.random.choice(object_numbers, size=self.n_points, replace=True)

        # except Exception as e:
        #     print("\n===== DATASET DEBUG INFO =====")
        #     print(f"Index: {idx}")
        #     print(f"Image path: {img_pth}")
        #     print(f"Mask path: {msk_pth}")
        #     print(f"Split: {self.split}")
        #     print(f"Unique objects (after aug): {object_numbers}")
        #     print(f"Mask shape: {mask_np.shape}")
        #     print(f"Mask dtype: {mask_np.dtype}")
        #     print(f"Mask min/max: {mask_np.min()} / {mask_np.max()}")

        #     # создаём директорию для дебага
        #     debug_dir = "./debug_dataset"
        #     os.makedirs(debug_dir, exist_ok=True)

        #     # сохраняем изображение
        #     debug_img_path = os.path.join(debug_dir, f"img_{idx}.png")
        #     Image.fromarray(
        #         (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        #         if isinstance(image, torch.Tensor) else np.array(image)
        #     ).save(debug_img_path)

        #     # сохраняем маску
        #     debug_mask_path = os.path.join(debug_dir, f"mask_{idx}.png")
        #     Image.fromarray(mask_np.astype(np.uint8)).save(debug_mask_path)

        #     print(f"Saved image to: {debug_img_path}")
        #     print(f"Saved mask to: {debug_mask_path}")
        #     print("================================\n")

        #     raise e

        
        points = np.zeros((self.n_points,2))
        
        label = np.zeros((self.n_points))
        
        h,w =mask_np.shape
        j=0
        for i in x:
            ori = np.where( mask_np == i)
            rand = random.randint(ori[0].shape[0])
            points[j] = (ori[0][rand]/h, ori[1][rand]/w)
            label[j] = i - 1
            j+=1
        
        points = 2 * points - 1

        return {"image" : image,
                "points" : torch.from_numpy(points).to(dtype=torch.float32),
                "label" : torch.from_numpy(label).to(dtype=torch.float32),
               }

