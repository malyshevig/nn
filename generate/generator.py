
from PIL import Image, ImageDraw
import unittest
import random
import numpy as np
import json
import graph.bez2 as bez
from graph.bez2 import BezìerFactory
from graph.bez2 import BezierRect
import im_utils as utils
from im_utils import CvUtils
from numpy import ndarray, asarray
import cv2
from matplotlib import pyplot as pl




def is_crossed(box1: (), box2: ()) -> bool:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    if b1_x1 > b2_x2:
        return False

    if b1_x2 < b2_x1:
        return False

    if b1_y1 > b2_y2:
        return False

    if b1_y2 < b2_y1:
        return False

    return True


def generate_boxes() -> []:
    box_num = random.randint(2, 6)
    boxes = []

    for i in range(box_num):
        not_checked = True

        while not_checked:
            x1 = random.randint(10, 3000)
            y1 = random.randint(10, 3000)
            x2 = random.randint(x1 + 10, 3990)
            y2 = random.randint(y1 + 10, 3990)

            if (abs(y2 - y1) < 40) or (abs(x2 - x1) < 40):
                not_checked = True
                continue

            box = (x1, y1, x2, y2)
            not_checked = False
            for b in boxes:
                if is_crossed(box, b):
                    not_checked = True
                    break

        boxes.append(box)
    return boxes


def generate_lines() -> []:
    pass


def generate_image() -> (Image, Image, list):
    image = Image.new("RGB", (800, 800), (255, 255, 255))
    image_mask = Image.new("RGB", (800, 800), (0, 0, 0))

    image_draw: ImageDraw = ImageDraw.Draw(image, "RGB")
    image_mask_draw: ImageDraw = ImageDraw.Draw(image_mask, "RGB")
    boxes = generate_boxes()

    for n, b in enumerate(boxes):
        image_draw.rectangle(b, outline="Green", width=2)
        image_mask_draw.rectangle(b, fill=(100, 100, n + 1))

    return image, image_mask, boxes


bezierFactory = BezìerFactory()


def draw_points(draw: ImageDraw, points: [], color="Black"):
    for (x, y) in points:
        draw.point((x, y), fill=color)


def generate_warp_image() -> (Image, Image, list):
    image = Image.new("RGB", (800, 800), (255, 255, 255))
    image_mask = Image.new("RGB", (800, 800), (0, 0, 0))

    image_draw: ImageDraw = ImageDraw.Draw(image, "RGB")
    image_mask_draw: ImageDraw = ImageDraw.Draw(image_mask, "RGB")
    boxes = generate_boxes()

    new_boxes = []

    for n, b in enumerate(boxes):
        br = bezierFactory.get_rect((b[0], b[1]), (b[2], b[3]))

        mask, box = br.get_mask()
        points = br.get_points()

        new_boxes.append(box)
        draw_points(image_draw, points)
        draw_points(image_mask_draw, mask, color=(100, 100, n + 1))

    return image, image_mask, boxes


from os import listdir


class SamplesGenerator:
    sample_dir = "../data/sample/"

    def __init__(self):
        self.samples = []

        for fn in listdir(self.sample_dir):
            fname = self.sample_dir + fn

            pic_class = None
            if fn.startswith("arrow"):
                pic_class = "arrow"

            if fn.startswith("rect"):
                pic_class = "rect"

            if pic_class:
                img: ndarray = cv2.imread(fname)
                self.samples.append((pic_class, img))


    def generate_warp_image2(self) -> (Image, Image, list, list):
        image = Image.new("RGB", (4000, 4000), (255, 255, 255))
        image_mask = Image.new("RGB", (4000, 4000), (0, 0, 0))

        image_draw: ImageDraw = ImageDraw.Draw(image, "RGB")
        image_mask_draw: ImageDraw = ImageDraw.Draw(image_mask, "RGB")
        boxes = generate_boxes()

        new_boxes = []
        pic_classes = []

        for n, box in enumerate(boxes):
            pic_class, img = self.samples[random.randint(0, len(self.samples) - 1)]
            pic_classes.append(pic_class)

            x = box[0]
            y = box[1]

            img_new = img

            points_list = CvUtils.get_points_condition(img_new, lambda c: c[0] == 255)
            if pic_class == "rect":
                mask_points, n_box = CvUtils.fill(points_list)
            else:
                mask_points = points_list
                n_box = CvUtils.get_box_for_list(points_list)

            new_box = [x+n_box[0], y + n_box[1], x + n_box[2], y + n_box[3]]

            new_boxes.append(new_box)
            draw_points(image_draw, CvUtils.shift((box[0], box[1]), points_list), color="Black")
            draw_points(image_mask_draw, CvUtils.shift((box[0], box[1]), mask_points), color=(100, 100, n + 1))

            #b = new_boxes[0]
            #image_draw.rectangle(b, outline="Black", width=2)
            #cv2.rectangle (image, (b[0],b[1]), (b[2],b[3]), color="Black",thickness = 2)

 #           image.show()
 #           image_mask.show()



        return image, image_mask, new_boxes, pic_classes


def generate_warp_image2() -> (Image, Image, list):
    image = Image.new("RGB", (800, 800), (255, 255, 255))
    image_mask = Image.new("RGB", (800, 800), (0, 0, 0))

    image_draw: ImageDraw = ImageDraw.Draw(image, "RGB")
    image_mask_draw: ImageDraw = ImageDraw.Draw(image_mask, "RGB")
    boxes = generate_boxes()

    new_boxes = []

    for n, b in enumerate(boxes):
        br = bezierFactory.get_rect((b[0], b[1]), (b[2], b[3]))

        mask, box = br.get_mask()
        points = br.get_points()

        new_boxes.append(box)
        draw_points(image_draw, points)
        draw_points(image_mask_draw, mask, color=(100, 100, n + 1))

    return image, image_mask, boxes


def generate_images(num: int) -> None:
    images = []
    for i in range(num):
        image, image_mask, boxes = generate_warp_image()
        image_file = f"../data/image_{i}.jpg"
        image.save(image_file)

        image_mask_file = f"../data/image_mask_{i}.jpg"
        image_mask.save(image_mask_file)
        s = json.dumps(boxes)

        image_boxes_file = f"../data/image_boxes_{i}.json"
        with open(image_boxes_file, "wt") as fd:
            fd.write(s)

        images.append({"num": i, "image": image_file, "image_mask": image_mask_file, "image_boxes": image_boxes_file})

    with open("../data/images.json", "wt") as fd:
        fd.write(json.dumps(images))


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


class TestCalculator(unittest.TestCase):

    def setUp(self):
        pass

    def test_cross(self):
        assert (not is_crossed((10, 10, 20, 20), (100, 100, 200, 200)))
        assert (is_crossed((10, 10, 200, 200), (100, 100, 200, 200)))
        assert (is_crossed((100, 100, 500, 500), (80, 80, 200, 200)))

    def test_gen_boxes(self):
        assert (generate_boxes() is not None)

    def no_test_generation(self):
        image, image_mask, boxes = generate_image()
        image.save("/Users/im/tmp/1.jpg")
        image_mask.save("/Users/im/tmp/1_mask.jpg")


    def no_test_generate_images(self):
        generate_images(1000)

    pic_class_dict = {"rect":1, "arrow": 2}


    def test_generate_images2(self):
        images = []
        for n in range(1000):
            sg = SamplesGenerator()
            img, img_mask, boxes, pic_classes = sg.generate_warp_image2()

            img_arr = asarray(img)
            new_img_arr = CvUtils.image_resize(img_arr, [800,800])

            mask_arr = asarray(img_mask)
            new_mask_arr = cv2.cvtColor(CvUtils.image_resize(mask_arr, [800, 800]), cv2.COLOR_RGB2BGR)

            new_boxes = []
            for i,b in enumerate(boxes):
                new_boxes.append ([b[0] // 5, b[1] // 5,b[2] // 5 ,  b[3] // 5,
                                   __class__.pic_class_dict[pic_classes[i]]])

            image_file = f"../data/image_{n}.jpg"
            image_mask_file = f"../data/image_mask_{n}.jpg"
            cv2.imwrite(image_file, new_img_arr)
            cv2.imwrite(image_mask_file, new_mask_arr)

            s = json.dumps(new_boxes)
            image_boxes_file = f"../data/image_boxes_{n}.json"
            with open(image_boxes_file, "wt") as fd:
                fd.write(s)

            images.append({"num": n, "image": image_file, "image_mask": image_mask_file, "image_boxes": image_boxes_file})

        with open("../data/images.json", "wt") as fd:
            fd.write(json.dumps(images))




    def no_test_read_images_desc(self):
        read_images_desc()

    def no_test_read_image(self):
        read_image(read_images_desc()[2])


def main():
    image, image_mask, boxes = generate_image()
    image.save("/Users/im/tmp/1.jpg")
    image_mask.save("/Users/im/tmp/1_mask.jpg")


if __name__ == "__main__":
    main()
