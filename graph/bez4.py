import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import unittest

print(f"cwd={os.getcwd()}")
file = "./data/ttf/Sergeyp-Regular.ttf"

import bezier as bz

font = ImageFont.truetype(font=file, size=20)

def voice(val, sigma=50):
    return np.random.normal(val, sigma)


def timer(func, *args):
    tm1 = time.time_ns()
    r = func(*args)

    tm2 = time.time_ns()
    print (f"{func} time {tm2-tm1}")
    return r


 # Make array smooth
def make_array_smooth (arr):
    groups: [] = None
    tolerance = 10

    tm1 = time.time_ns()

    for v in arr:
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

    tm2 = time.time_ns()
    print(f"point 1 = {tm2 - tm1} {len(groups)}")
    if (len(groups) > 3):
        print(f"{groups}")

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
        if abs(leading_val - v) > tolerance:
            interp_x.append(n)
        else:
            xp.append(n)
            yp.append(v)

    arr2 = np.interp(interp_x, xp, yp)

    for idx, n in enumerate(interp_x):
        arr[n] = int(round(arr2[idx]))

    return arr


#image = Image.new("RGB", (1000, 1000), (255, 255, 255))
#draw = ImageDraw.Draw(image)

def get_box2(x1, y1, x2, y2):
    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2)
    ]


def get_box(xy1,xy2):
    x1, y1 = xy1
    x2, y2 = xy2
    return get_box2(x1,y1,x2,y2)




class NNObject:
    def __init__(self, box, points):
        self.box = box
        self.masks = np.zeros ()

    def get_box (self):
        return self.box

    def get_mask(self):
        x1,x2,y1,y2 = box
        mask = np.zero([x2-x1+1, y2-y1+1], dtype=int)
        for x,y in points:
            mask [x,y] = 1

        return mask


class Drawer:
    def __init__(self, drawImage):
        self.drawImage = drawImage
        self.size = drawImage.im.size

    def drawObj(self, obj):
        if obj.points:
            self.draw(obj.points())
        if obj.label:
            self.drawLabel(obj.label)


    def draw (self,points):
        for p in points:
            x,y = p

            self.drawImage.point((x, self.size[1]-y), fill='red')

    def drawLabel (self, label):
        x,y = label.rect
        self.drawImage.text((x, self.size[1]-y), label.text, 'blue', font)



class Canvas:
    def parentCanvas(self):
        pass

    def draw(self, points):
        pass

    def drawText (self, x,y, text):
        pass

    def drawBox(self, box):
        pass


class ImageCanvas (Canvas):
    def __init__(self, image):
        self.image = image
        self.drawImage = ImageDraw.Draw(image)
        self.w, self.h = image.size

    def draw (self,points):
        for p in points:
            x, y = p
            self.drawImage.point((x, self.h - y), fill='black')

    def drawText(self,  x, y, text):
        text_width, text_height = font.getsize(text)

        self.drawImage.text((x, self.h-(y+text_height)), text, 'blue', font)

    def drawBox(self, box):
        x1,x2,y1,y2 = box
        y1 = self.h - y1
        y2 = self.h - y2

        self.drawImage.line([(x1,y1), (x1,y2)], fill="red", width = 2)
        self.drawImage.line([(x1,y1), (x2, y1)], fill="red", width = 2)
        self.drawImage.line([(x1, y2), (x2, y2)], fill="red", width = 2)
        self.drawImage.line([(x2, y1), (x2, y2)], fill="red", width = 2)



class RectCanvas(Canvas):
    def __init__(self, parentCanvas,xy):
        self.parentCanvas = parentCanvas
        self.x, self.y = xy


    def draw (self,points):
        new_points = ((x+self.x, y+self.y) for x,y in points)
        self.parentCanvas.draw(new_points)

    def drawText(self, x, y, text):
        self.parentCanvas.drawText(x+self.x, y+self.y, text)


class Bezier:
    def __init__(self, xp, yp):
        self.nodes = np.asfortranarray([xp, yp])
        self.curve = bz.Curve(self.nodes, degree=len(xp) - 1)
        super().__init__()

    def points(self, points):
        s_vals = np.linspace(0.0, 1.0, points)
        points = self.curve.evaluate_multi(s_vals)
        rp = [(points[0, i], points[1, i]) for i in range(len(points[0]))]
        return rp


class Line(Bezier):

    def sigma_factor (self,i,n):
        if (i == 0):
            return 0
        if (i == n):
            return 0

        return abs(n*0.5-i)*1.3

    def __init__(self, xy1, xy2):
        self.xy1 = xy1
        self.xy2 = xy2

        (x1, y1) = xy1
        (x2, y2) = xy2

        self.len = int (abs(x2 - x1) + abs(y2 - y1))
        self.n = round(self.len / 40)
        n = self.n
        if (n < 3):
            n = 3

        if (n>5):
            n = 5


        vx = abs(x2 - x1) / 30
        vy = abs(y2 - y1) / 30

        if (n >0):
            xp = [voice((x2 - x1) / n * i, vy*self.sigma_factor(i,n)) + x1 for i in range(n + 1)]
            yp = [voice((y2 - y1) / n * i, vx*self.sigma_factor(i,n)) + y1 for i in range(n + 1)]
        else:
            xp = [x1,x2]
            yp = [y1,y2]

        #print (f"{n} {vx} {vy}")

        super().__init__(xp, yp)

    def points(self):
        return super().points(self.len * 2)


class Arrow:
    def __init__(self, xy, direction="left"):

        x1,y1 = xy

        d = 15
        dd = d / 3

        x2, y2 = x1,y1
        x3, y3 = x1,y1

        if direction == "right":
            x2, y2 = x1-d, y1-dd
            x3, y3 = x1-d, y1+dd
        elif direction == "left":
            x2, y2 = x1+d, y1-dd
            x3, y3 = x1+d, y1+dd
        elif direction == "up":
            x2, y2 = x1-dd, y1-d
            x3, y3 = x1+dd, y1-d
        elif direction == "down":
            x2, y2 = x1-dd, y1+d
            x3, y3 = x1+dd, y1+d

        self.lines = []
        self.lines.append(Line( xy, (x2,y2)))
        self.lines.append(Line( xy, (x3,y3)))


    def points (self):
        r = []
        l: Line
        for l in self.lines:
            r.extend(l.points())

        return r



class ComplexLine:
    def __init__(self, xy1, xy2, direct=True, arrows=(False,False)):


        self.xy1 = xy1
        self.xy2 = xy2

        (x1, y1) = xy1
        (x2, y2) = xy2

        w = x2 - x1
        h = y2 - y1

        self.lines = []

        if direct:
            self.lines.append(Line(xy1, (x1+w,y1)))
            self.lines.append(Line((x1+w,y1),xy2))
        else:
            self.lines.append(Line(xy1, (x1, y1+h)))
            self.lines.append(Line((x1, y1+h), xy2))

        beginArrow, endArrow = arrows

        self.arrows = []
        if direct:
            if arrows[0]:
                if w>0:
                    self.arrows.append (Arrow(xy1,"left"))
                else:
                    self.arrows.append(Arrow(xy1, "right"))
            if arrows[1]:
                if h >0:
                    self.arrows.append(Arrow(xy2, "up"))
                else:
                    self.arrows.append(Arrow(xy2, "down"))
        else:
            if arrows[0]:
                if h>0:
                    self.arrows.append (Arrow(xy1,"down"))
                else:
                    self.arrows.append (Arrow(xy1, "up"))

            if arrows[1]:
                if w >0:
                    self.arrows.append(Arrow(xy2, "right"))
                else:
                    self.arrows.append(Arrow(xy2, "left"))

    def points(self):
        r = []
        l: Line
        for l in self.lines:
            r.extend(l.points())

        for arr in self.arrows:
            r.extend(arr.points())

        return r

    def get_box(self):
        return get_box(self.xy1, self.xy2)

class Label:
    def __init__(self, rect, text):
        self.text = text
        self.x, self.y = rect
        text_width, text_height = font.getsize(text)

    def getImage(self):
        return rect, label


class BezierRect:

    def __init__(self,  xy1, xy2, label=""):
        self.xy1 = xy1
        self.xy2 = xy2

        (x1, y1) = xy1
        x1 = voice (x1,5)
        y1 = voice (y1,5)

        (x2, y2) = xy2
        x2 = voice (x2,5)
        y2 = voice (y2,5)

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.label = Label((20,20), label)

        self.lines = []
        self.lines.append(Line(xy1, (x1, y2)))
        self.lines.append(Line(xy1, (x2, y1)))
        self.lines.append(Line((voice(x1,9), voice(y2,9)), (x2, y2)))
        self.lines.append(Line((x2, y1), (voice(x2,9), voice(y2,9))))
        self.points_list = self.points()

    def points(self):
        r = []
        l: Line
        for l in self.lines:
            r.extend(l.points())

        return r

    def get_points(self):
        return self.points_list

    def get_box(self):
        return get_box(self.xy1, self.xy2)

    def get_box2(self):
        points_list = np.array(self.points_list)

        l= np.max (points_list, axis=0)
        max_x =l[0]
        max_y =l[1]

        l = np.min(points_list, axis=0)
        min_x = l[0]
        min_y = l[1]

        return get_box((min_x, min_y), (max_x, max_y))

    def get_mask(self):
        points_list = np.array(self.points())
        box = [int(round(d)) for d in self.get_box2()]

        w = box[2]-box[0]+1
        h = box[3]-box[1]+1
        mask_points = []

        min_x = h*[None]
        max_x = h*[None]
        min_y = w*[None]
        max_y = w*[None]

        l = [(int(round(x)), int(round(y))) for (x, y) in points_list]
        for x,y in l:
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

        min_x = self.check_array(min_x)
        max_x = self.check_array(max_x)
        min_y = self.check_array(min_y)
        max_y = self.check_array(max_y)

    #    min_x = timer (self.check_array, min_x)
    #    max_x = timer (self.check_array, max_x)
    #    min_y = timer (self.check_array, min_y)
    #    max_y = timer (self.check_array, max_y)


       # min_x, max_x = self.check_arrays(min_x,max_x)
       # min_y, max_y = self.check_arrays(min_y,max_y)

        for x in range(w):
            for y in range(h):
                check_x = (x >= min_x[y]) and (x <= max_x[y])
                check_y = (y >= min_y[x]) and (y <= max_y[x])

                if check_x and check_y:
                    mask_points.append((x+box[0], y+box[1]))
                else:
                    pass

        return mask_points, box


    def check_array(self,  arr):
        return make_array_smooth(arr)


    def check_arrays(self, arr_min, arr_max):
        last_normal = None
        arr_len = len(arr_min)

        for n in range(arr_len):
            min_v = arr_min[n]
            max_v = arr_max[n]

            if min_v < max_v:
                last_normal = (min_v,max_v)

            if min_v >= max_v:
                if last_normal is not None:
                    arr_min[n], arr_max[n] = last_normal

        last_normal = None

        for n in range(arr_len,-1):
            min_v = arr_min[n]
            max_v = arr_max[n]

            if min_v < max_v:
                last_normal = (min_v, max_v)

            if min_v >= max_v:
                if last_normal is not None:
                    arr_min[n], arr_max[n] = last_normal

        return arr_min, arr_max


    def get_mask5(self):
        points_list = np.array(self.points())
        box = self.get_box2()
        mask_points = []

        l = [(round(x), round(y)) for (x, y) in points_list]

        points_list = list(set(l))
        box = [round(x) for x in box]

        for x in np.arange (box[0], box[2],1):
            p2 = [py for (px, py) in points_list if abs(px - x) < 0.3]


            for y in np.arange (box[1], box[3],1):
                if round(x) % 10 == 0:
                    print (f"{x} {y}")
                check_x: bool
                check_y: bool

                p1 = [px for (px,py) in points_list if abs (py - y)<0.3]
                check_x = (x >= min(p1)) and (x <= max(p1))
                check_y = (y >= min(p2)) and (y <= max(p2))

                if check_x and check_y:
                    mask_points.append((x,y))

        return mask_points



    def get_mask4(self):
        points_list = np.array(self.points())
        box = self.get_box2()
        mask_points = []

        #points_list = [x,y for (x,y) in points_list]

        for x in np.arange (box[0], box[2],1):
            p2 = [py for (px, py) in points_list if abs(px - x) < 0.3]

            for y in np.arange (box[1], box[3],1):
                if round(x) % 10 == 0:
                    print (f"{x} {y}")
                check_x: bool
                check_y: bool

                p1 = [px for (px,py) in points_list if abs (py - y)<0.3]
                check_x = (x >= min(p1)) and (x <= max(p1))
                check_y = (y >= min(p2)) and (y <= max(p2))

                if check_x and check_y:
                    mask_points.append((x,y))

        return mask_points


    def get_mask2(self):
        points_list = np.array(self.points())
        box = self.get_box2()
        mask_points = []

        for x in range (int(np.round(box[0])), int(np.round(box[2]))):
            for y in range (int(np.round(box[1])), int(np.round(box[3]))):
                if x % 10 == 0:
                    print (f"{x} {y}")
                check_x: bool
                check_y: bool

                p1 = [px for (px,py) in points_list if abs (py - y)<0.3]
                check_x = (x >= min(p1)) and (x <= max(p1))

                p2 = [py for (px, py) in points_list if abs(px - x)<0.3]
                check_y = (y >= min(p2)) and (y <= max(p2))

                if check_x and check_y:
                    mask_points.append((x,y))

        return mask_points

    def get_mask3(self):
        points_list = np.array(self.points())
        box = self.get_box2()
        mask_points = []

        for x in np.arange ((np.round(box[0])), (np.round(box[2])),1.0):
            p2 = [py for (px, py) in points_list if abs(px - x) < 0.3]

            for y in np.arange ((np.round(box[1])), (np.round(box[3])),1.0):
                if x % 10 == 0:
                    print (f"{x} {y}")
                check_x: bool
                check_y: bool

                p1 = [px for (px,py) in points_list if abs (py - y)<0.3]
                check_x = (x >= min(p1)) and (x <= max(p1))


                check_y = (y >= min(p2)) and (y <= max(p2))

                if check_x and check_y:
                    mask_points.append((x,y))

        return mask_points


class Rect:
    def __init__(self, canvas, xy1, w, h, label=""):
        self.canvas = canvas
        (x1, y1) = xy1

        x2 = x1 + w
        y2 = y1 + h

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.label = Label(self.subCanvas(), (10,10), label)

        self.lines = []
        self.lines.append(Line(self.canvas, xy1, (x1, y2)))
        self.lines.append(Line(self.canvas, xy1, (x2, y1)))
        self.lines.append(Line(self.canvas, (x1, y2), (x2, y2)))
        self.lines.append(Line(self.canvas, (x2, y1), (x2, y2)))

    def points(self):
        r = []
        l: Line
        for l in self.lines:
            r.extend(l.points())

        return r

    def subCanvas (self):
        return RectCanvas(self.canvas, (self.x1, self.y1))

    def get_box(self):
        return get_box2 (self.x1, self.y1, self.x2, self.y2)


class FigureTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_smooth(self):
        arr = [0,0,20,18,15,0,0,1,2,3,4,5,6,100,200, 200,200,200,200,201,20]

        arr2 = timer (make_array_smooth,arr)
        print (arr2)

    def test_rect(self):
        rect = BezierRect((100, 100), (300, 300))
        assert (rect is not None)

        image = Image.new ("RGB",(800,800), (255,255,255))
        mask = Image.new ("RGB",(800,800), (255,255,255))

        draw = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(image)

        for x,y in rect.points():
            draw.point((x,y), fill= "Blue")

        box: [] = rect.get_box2()
        draw.rectangle(box, outline="Green", width=2)

        mask,box = rect.get_mask()
        for x, y in mask:
            draw.point((x, y), fill="Red")

        image.show()


class BezìerFactory:

    def get_rect(self, xy1:(), xy2:())-> BezierRect:
        return BezierRect(xy1,xy2)

