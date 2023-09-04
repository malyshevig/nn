#from generate import generator

import torchvision
import torchvision.transforms as T
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import logging
import json


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transforms(train):
    trans = [T.ToTensor()]
    if train:
        trans.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(trans)


def read_images_desc() -> []:
    with open("../data/images.json", "rt") as fd:
        images = json.load(fd)

    # with open("../data_2/images.json", "rt") as fd:
    #    images2 = json.load(fd)

    # images += images2
    return images


def read_image(image_desc: dict) -> (Image, Image, []):
    image_file = image_desc["image"]
    image_mask_file = image_desc["image_mask"]
    image_boxes_file = image_desc["image_boxes"]

    image: Image = Image.open(image_file)
    image.load()
    image_mask: Image = Image.open(image_mask_file)
    image_mask.load()

    with open(image_boxes_file, "rt") as fd:
        boxes = json.load(fd)

    return image, image_mask, boxes


class ModelsDataset(torch.utils.data.Dataset):
    def __init__(self, models):
        print ("New DataSet Instance")
        self.transforms = get_transforms(True)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images_desc:[dict] = read_images_desc()
        print (f"DataSet len ={len(self.images_desc)}")
        self.c = 0

    def __len__(self):
        return len(self.images_desc)

    def __getitem__(self, idx):
        logging.info(f'DataSet getItem ({idx})')
 #       print (f'{self.c} DataSet getItem ({idx})')
        self.c += 1

        # load images and masks
        image_desc: dict = self.images_desc[idx]
        image, image_mask, boxes = read_image(image_desc)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = image_mask
        # convert the PIL Image into a numpy array
        mask = np.array(mask)

        obj_ids = list(range(1, len(boxes)+1))

        masks = []
        new_boxes = []
        new_labels =[]
        w,h,_ = mask.shape
        for obj in obj_ids:
            m = np.zeros((w,h))
            for i,j in range (w,h):
                m [i][j] = mask[i,j,2] == obj
            masks.append(m)

        for b in boxes:
            new_boxes.append(b[0:4])
            new_labels.append(b[4])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
        num_objs = len(boxes)
        # there is only one class
       #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(torch.asarray(new_labels), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
       # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image), target

        logging.info("getItem Ok")
        return image, target
