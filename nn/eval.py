import torch
from util import get_model_instance_segmentation
from util import ModelsDataset, get_transforms
from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as pl

import torchvision.transforms as T

num_classes = 3


model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("../model/model.save"))



img = cv2.imread ("../eval/eval_1.jpg")
#img = cv2.imread ("../data/image_125.jpg")

img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, [800,800], interpolation=cv2.INTER_AREA)

#pl.imshow(img)

#cv2.waitKey(0)
imt = T.ToTensor()(img)



new_image: Image = Image.new("RGB", (800, 800), (255, 255, 255))
image_draw: ImageDraw = ImageDraw.Draw(new_image, "RGB")

model.eval()
with torch.no_grad():
    out_data: [dict] = model([imt])

    r_dict: dict = out_data[0]
    scores = r_dict["scores"]
    boxes = r_dict["boxes"]
    masks = r_dict["masks"]
    labels = r_dict["labels"]

    print (scores)
    boxes_list = [boxes[idx] for idx, score in enumerate(scores) if score > 0.5]
    #print (boxes_list)
    #print (masks)
    for n,box in enumerate(boxes_list):
        label = labels[n]
        if label == 1:
            image_draw.rectangle((box[0], box[1], box[2], box[3]), outline="Blue", width=2)
        else:
            image_draw.rectangle((box[0], box[1], box[2], box[3]), outline="Black", width=2)

    #pil.show("Image 1")
    #new_image = new_image.resize(original_size)
    new_image.show("Image 2")


