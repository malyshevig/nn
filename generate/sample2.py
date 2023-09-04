from sklearn.cluster import KMeans
import torch
from  PIL import Image, ImageDraw
import numpy as np
from numpy import asarray, uint8
import sklearn as sk
from graph.bez2 import get_box
from im_utils import CvUtils, CvClusterUtil


import cv2

def generate_samples(in_fname, out_fname):
    image: Image = Image.open(in_fname)
    image.load()

    image_arr = asarray(image)
    image_arr = CvUtils.image_resize(image_arr, (round(w * 0.50), round(h * 0.50)))

    arr_factored = np.zeros((w, h), dtype=uint8)
    points = CvUtils.get_points_condition(image_arr, lambda c: c[0] < 80 and c[1] < 80 and c[2] < 80)

    for x, y in points: arr_factored[x][y] = 1

    clusters = CvClusterUtil(arr_factored).find_clusters()




image: Image = Image.open("../data_hw/rectangle.jpg")
#image: Image = Image.open("../data_hw/arrow.jpg")
image.load()
arr = asarray(image)

w,h = image.size

arr = CvUtils.image_resize(arr,(round(w*0.50),round(h*0.50)))
print (f"initial size {w} {h}")
print (f"initial size {arr.shape[0]} {arr.shape[1]}")



w,h,c  = arr.shape
arr3 = np.zeros((w,h), dtype=uint8)
points = []


points = CvUtils.get_points_condition(arr, lambda c: c[0]<80 and c[1]<80 and c[2]<80)
for x,y in points:
    arr3[x][y] = 1




image3 = Image.new("RGB", (w,h), (0,0,0))
draw = ImageDraw.Draw(image3)
for p in points:
    draw.point(p, fill="White")

image3.show()

h,w = arr3.shape
clusters = []

clusters = CvClusterUtil(arr3).find_clusters()

for c in clusters:
    img = CvUtils.draw_points(c)
    img.show()


cur_cluster = None

dist = 20


class Stack:

    def __init__(self):
        self.vals = []

    def push(self, v):
        self.vals.append(v)

    def peek(self):
        return self.vals[-1]

    def pop(self):
        return self.vals.pop()

    def empty(self):
        return len(self.vals) == 0


def check_point (x,y) -> bool:
    if x < 0 or x>= arr3.shape[1] or y <0 or y>= arr3.shape[0]: return False

    if arr3[y][x] != 1: return False
    return True


def find_cluster2(x,y)->[]:
    cluster = []

    if not check_point(x,y): return cluster

    stack = Stack ()

    cluster.append((x, y))
    arr3[y][x] = 0
    stack.push((x,y))
    while not stack.empty():
        (x, y) = stack.pop()

        for dx in range(-dist, dist):
            for dy in range(-dist, dist):

                if dx == 0 and dy == 0: continue

                nx = x+dx
                ny = y+dy

                if check_point(nx,ny):
                    stack.push((nx,ny))
                    cluster.append((nx, ny))
                    arr3[ny][nx] = 0

    return cluster



def draw_points(points_list):
    minx, miny, maxx, maxy = CvUtils.get_box_for_list(points_list)
    w = maxx-minx
    h = maxy-miny


    image = Image.new("RGB", (w, h), (0,0,0))
    draw = ImageDraw.Draw(image)

    for p in points_list:
        draw.point((p[0]-minx,p[1]-miny), fill="White")

    return image


clusters = []
n = 0

for y in range(h):
    for x in range(w):
        cluster = find_cluster2(x,y)
        if cluster and len(cluster)>100:
            clusters.append(cluster)
            print (f"{len(cluster)}")
            im = draw_points(cluster)

            box = CvUtils.get_box_for_list(cluster)

            im.save(f"../data/sample/rect_{n}.jpg")
            #im.save(f"../data/sample/arrow_{n}.jpg")

            n+=1


print(len(clusters))




