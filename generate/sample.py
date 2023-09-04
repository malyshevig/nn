from sklearn.cluster import KMeans
import torch
from  PIL import Image, ImageDraw
import numpy as np
from numpy import asarray, uint8
import sklearn as sk
from graph.bez2 import get_box
from im_utils import CvUtils, CvClusterUtil
from matplotlib import pyplot as pl


import cv2


def generate_samples2(in_fname, out_fname, ratio):
    img = cv2.imread (in_fname)
    if img is None: raise Exception (f" can't read file {in_fname}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11,15,15)
   # factored = cv2.Canny(gray.copy(), 50, 200, 20)
    factored = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)[1]

    cv2.imwrite(f"/tmp/gray.jpg", gray)
    cv2.imwrite(f"/tmp/factored.jpg",factored)


    clusters = CvClusterUtil(factored).find_clusters()
    for n,c in enumerate(clusters):
        image = CvUtils.draw_points(c)
        image.save(f"{out_fname}_{n}.jpg")


  #  points = CvUtils.get_points_condition(factored, lambda c: c[0] == 255)






def generate_samples(in_fname, out_fname, ratio):
    image: Image = Image.open(in_fname)
    image.load()

    image_arr = asarray(image)
    w,h,_ = image_arr.shape
    if ratio < 1:
        image_arr = CvUtils.image_resize(image_arr, (round(w * ratio), round(h * ratio)))

    w,h,_ = image_arr.shape

    arr_factored = np.zeros((w, h), dtype=uint8)
    points = CvUtils.get_points_condition(image_arr, lambda c: c[0] < 150 and c[1] < 150 and c[2] < 150)

    CvUtils.draw_points(points).show()

    for x, y in points: arr_factored[x][y] = 1

    clusters = CvClusterUtil(arr_factored).find_clusters()
    for n,c in enumerate(clusters):
        image = CvUtils.draw_points(c)
        image.save(f"{out_fname}_{n}.jpg")

    return len(clusters)

print ("Generatings rectangles")
n = generate_samples2("../data_hw/rectangle.jpg","../data/sample/rect", 0.5)
print (f"Generated {n} rectangle samples")

print ("Generatings arrow")
n = generate_samples2("../data_hw/arrow2.jpg","../data/sample/arrow2", 1)
print (f"Generated {n} arrow samples")

