import torch
import numpy as np
from detectron2.data import detection_utils as utils
# from .semantic_oneformer_custom_dataset_mapper import SemanticOneFormerCustomDatasetMapper
from datasets.custom_datasets.semantic_oneformer_custom_dataset_mapper import (
    SemanticOneFormerCustomDatasetMapper
)

import copy
import logging
import os

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from oneformer.utils.box_ops import masks_to_boxes
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize

class LandcoverTreesMapper(SemanticOneFormerCustomDatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        # Читаем маску
        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("uint8")
        
        # Создаем новую маску: 
        # Пусть 0 будет "деревья", а 1 - "всё остальное"
        new_sem_seg = np.ones(sem_seg_gt.shape, dtype="uint8") # Заполняем единицами (фон)
        new_sem_seg[sem_seg_gt == 192] = 0 # Ставим нули там, где деревья
        
        sem_seg_gt = new_sem_seg # Теперь у нас только классы 0 и 1

        # Аугментации
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt

        image_shape = (image.shape[0], image.shape[1])
        instances = Instances(image_shape)
        
        # Находим уникальные классы (теперь это [0, 1])
        classes = np.unique(sem_seg_gt.numpy())
        # Убираем игнорируемые, если они вдруг остались
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        masks = []
        for class_id in classes:
            masks.append(sem_seg_gt == class_id)

        if len(masks) == 0:
            instances.gt_masks = torch.zeros((0, image_shape[0], image_shape[1]))
        else:
            instances.gt_masks = torch.stack(masks)

        dataset_dict["instances"] = instances
        dataset_dict["task"] = "The task is semantic"
        
        dataset_dict["text"] = ["a photo with a trees", "a photo with a background"]
        
        return dataset_dict