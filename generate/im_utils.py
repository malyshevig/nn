import numpy as np
import time
import unittest
from PIL import Image, ImageDraw
from numpy import asarray, ndarray
import cv2


class CvUtils:

    @staticmethod
    def read_image_with_resizing(fname: str, dim: ()) -> ndarray:
        img = cv2.imread(fname)
        if img is None:
            raise Exception(f"Can't open file {fname}")

        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    @staticmethod
    def image_resize(in_img: ndarray, dim: ()) -> ndarray:

        return cv2.resize(in_img, dim, interpolation=cv2.INTER_AREA)


    @staticmethod
    def get_points_except_bg(image_array: ndarray, back=[255, 255, 255]) -> list:
        return CvUtils.get_points_condition(image_array, lambda c: c[0] == back[0] and c[1] == back[1] and c[2] == back[2])

    @staticmethod
    def get_points_condition(image_array: ndarray, condition) -> list:
        points = []
        w, h, c = image_array.shape

        for x in range(0, w):
            for y in range(0, h):
                cell: ndarray = image_array[x][y]

                if condition(cell):
                    points.append((x, y))

        return points

    @staticmethod
    def get_box_for_array (points: ndarray) -> ():

        l = np.max(points, axis=0)
        max_x = l[0]
        max_y = l[1]

        l = np.min(points, axis=0)
        min_x = l[0]
        min_y = l[1]

        return min_x, min_y, max_x, max_y

    @staticmethod
    def get_box_for_list(points_list: list) -> ():
        if not points_list: return None

        min_x, min_y, max_x, max_y = points_list[0][0],points_list[0][1],points_list[0][0],points_list[0][1]

        for x,y in points_list:
            if x > max_x: max_x = x

            if x < min_x: min_x = x

            if y > max_y: max_y = y

            if y < min_y: min_y = y

        return min_x, min_y, max_x, max_y

    @staticmethod
    def shift(xy: (), points: list) -> list:
        x, y = xy
        return [(x+p[0], y+p[1]) for p in points]

    @staticmethod
    def draw_points(points_list: list) -> Image:
        minx, miny, maxx, maxy = CvUtils.get_box_for_list(points_list)
        w = maxx - minx
        h = maxy - miny

        image = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(image)

        for p in points_list:
            draw.point((p[0] - minx, p[1] - miny), fill="White")

        return image


    @staticmethod
    def make_array_smooth(arr: ndarray):
        groups: [] = None
        tolerance = 10

        for v in arr:
            if v is not None:
                if groups is None:
                    groups = [(v, 1)]
                else:
                    add_new_group = True
                    for n, (g, c) in enumerate(groups):
                        if abs(g - v) < tolerance:
                            groups[n] = ((g * c + v) / (c + 1), c + 1)
                            add_new_group = False
                            break

                    if add_new_group:
                        groups.append((v, 1))

        leading_group = None
        for g in groups:
            if leading_group is None:
                leading_group = g
            else:
                if g[1] > leading_group[1]:
                    leading_group = g

        if len(leading_group) == 1:
            return arr

        leading_val = int(round(leading_group[0]))

        xp = []
        yp = []
        interp_x = []

        for n, v in enumerate(arr):
            if v is None or abs(leading_val - v) > tolerance:
                interp_x.append(n)
            else:
                xp.append(n)
                yp.append(v)

        arr2 = np.interp(interp_x, xp, yp)

        for idx, n in enumerate(interp_x):
            arr[n] = int(round(arr2[idx]))

        return arr

    @staticmethod
    def fill(points_list: list) -> (list, []):
        box = [int(round(d)) for d in CvUtils.get_box_for_list(points_list)]

        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1
        mask_points = []

        min_x = h * [None]
        max_x = h * [None]
        min_y = w * [None]
        max_y = w * [None]

        l = [(int(round(x)), int(round(y))) for (x, y) in points_list]
        for x, y in l:
            x = x - box[0]
            y = y - box[1]

            miny = min_y[x]
            maxy = max_y[x]
            if miny is None or miny > y:
                min_y[x] = y
            if maxy is None or maxy < y:
                max_y[x] = y

            minx = min_x[y]
            maxx = max_x[y]

            if minx is None or minx > x:
                min_x[y] = x
            if maxx is None or maxx < x:
                max_x[y] = x

        min_x = check_array(min_x)
        max_x = check_array(max_x)
        min_y = check_array(min_y)
        max_y = check_array(max_y)

        for x in range(w):
            for y in range(h):
                check_x = (x >= min_x[y]) and (x <= max_x[y])
                check_y = (y >= min_y[x]) and (y <= max_y[x])

                if check_x and check_y:
                    mask_points.append((x + box[0], y + box[1]))
                else:
                    pass

        return mask_points, box

    @staticmethod
    def draw_points(points_list: list):
        minx, miny, maxx, maxy = CvUtils.get_box_for_list(points_list)
        w = maxx-minx
        h = maxy-miny

        image = Image.new("RGB", (w, h), (0,0,0))
        draw = ImageDraw.Draw(image)

        for p in points_list:
            draw.point((p[0]-minx,p[1]-miny), fill="White")

        return image


class CvUtilsTest(unittest.TestCase):

    def test_read_message_with_resizing(self):
        CvUtils.read_image_with_resizing("./arrow.jpg", (800, 800))


class CvClusterUtil:
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

    def __init__(self, array: ndarray):
        self.array = array
        self.w = array.shape[0]
        self.h = array.shape[1]


    def __check_point(self, x, y) -> bool:
        if x < 0 or x >= self.w or y < 0 or y >= self.h: return False

        if self.array[x][y] == 0: return False
        return True

    def find_cluster(self, x, y) -> []:
        cluster = []

        if not self.__check_point(x, y): return cluster

        stack = self.Stack()

        cluster.append((x, y))
        self.array[x][y] = 0
        stack.push((x, y))

        while not stack.empty():
            (x, y) = stack.pop()

            for dx in range(-self.dist, self.dist):
                for dy in range(-self.dist, self.dist):

                    if dx == 0 and dy == 0: continue

                    nx = x + dx
                    ny = y + dy

                    if self.__check_point(nx, ny):
                        stack.push((nx, ny))
                        cluster.append((nx, ny))
                        self.array[nx][ny] = 0

        return cluster


    def find_clusters(self) ->[]:
        clusters = []
        n = 0

        for y in range(self.h):
            for x in range(self.w):
                cluster = self.find_cluster(x, y)
                if cluster and len(cluster) > 100:
                    print(f"add_cluster {cluster}")
                    clusters.append(cluster)

        return clusters


def check_array(arr):
    return make_array_smooth(arr)


# Make array smooth
def make_array_smooth(arr):
    groups: [] = None
    tolerance = 10

    tm1 = time.time_ns()

    for v in arr:
        if v is not None:
            if groups is None:
                groups = [(v, 1)]
            else:
                add_new_group = True
                for n, (g, c) in enumerate(groups):
                    if (g is None or v is None):
                        print(f"{groups} {arr}")

                    if abs(g - v) < tolerance:
                        groups[n] = ((g * c + v) / (c + 1), c + 1)
                        add_new_group = False
                        break

                if add_new_group:
                    groups.append((v, 1))

    tm2 = time.time_ns()

    leading_group = None
    for g in groups:
        if leading_group is None:
            leading_group = g
        else:
            if g[1] > leading_group[1]:
                leading_group = g

    if len(leading_group) == 1:
        return arr

    leading_val = int(round(leading_group[0]))

    xp = []
    yp = []
    interp_x = []

    for n, v in enumerate(arr):
        if v is None or abs(leading_val - v) > tolerance:
            interp_x.append(n)
        else:
            xp.append(n)
            yp.append(v)

    arr2 = np.interp(interp_x, xp, yp)

    for idx, n in enumerate(interp_x):
        arr[n] = int(round(arr2[idx]))

    return arr


def get_points_except_bg(img: Image, bg=[255, 255, 255]) -> list:
    image_array = asarray(img)
    points = []

    w, h, c = image_array.shape

    for x in range(0, w):
        for y in range(0, h):
            cell: ndarray = image_array[x][y]

            if not (cell[0] == bg[0] and cell[1] == bg[1] and cell[2] == bg[2]):
                points.append((x, y))

    return points


def draw_points(draw: ImageDraw, points: [], color="Blue"):
    for (x, y) in points:
        draw.point((x, y), fill=color)


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_fill(self):
        img = Image.new("RGB", (800, 800), (255, 255, 255))
        draw: ImageDraw = ImageDraw.Draw(img)

        draw.rectangle(((100, 100), (300, 300)), outline="Green", width=2)
        img_array = asarray(img)

        points = CvUtils.get_points_except_bg(img_array)
        mask_points, box = CvUtils.fill(points)
        draw_points(draw, mask_points)

        img.show()

