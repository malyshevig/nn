
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Box:

    def __init__(self,xy1,xy2):
        self.xy1 = xy1
        self.xy2 = xy2

    def to_array(self):
        return [self.xy1[0],self.xy1[1],  self.xy2[0], self.xy2[1]]

    def __repr__(self):
        return f"({self.xy1[0]},{self.xy1[1]}  {self.xy2[0]},{self.xy2[1]})"

class Container:
    def __init__(self):
        self.objs = {}
        self.objs_list = []

    def put(self, figure: object):
        self.objs [figure.id] = figure
        self.objs_list.append(figure)
        return self

    def get_obj(self, id):
        return self.objs[id]

    def get_objs(self):
        return self.objs.values()

    def __call__(self,arg):
        return self.get_obj(arg)

    def __getitem__(self, idx):
        return self.objs_list[idx]

    def __len__(self):
        return len(self.objs_list)

class Figure(Container):
    def __init__(self, id: str):
        super().__init__()
        self.id = id

    def box(self): pass


class Rect(Figure):

    def __init__(self, id: str, xy, size):
        super().__init__(id)

        self.x, self.y = xy
        self.w, self.h = size

        self.x1, self.y1, self.x2, self.y2 = self.x,self.y, self.x+self.w, self.y+self.h

    def box(self):
        return Box((self.x,self.y), (self.x+self.w, self.y+self.h))




class Connect(Figure):
    def __init__(self, id: str, id1: str, id2: str):
        super().__init__(id)
        self.id1 = id1
        self.id2 = id2


class Label(Figure):
    def __init__(self, id: str, xy, label: str):
        super().__init__(id)
        self.str = str
        self.x, self.y = xy
        self.label = label






def is_left(r1:Rect, r2:Rect):
    return r1.x2 < r2.x1

def is_right(r1:Rect, r2:Rect):
    return r1.x1 > r2.x2

def is_upper(r1:Rect, r2:Rect):
    return r1.y1>r2.y1

def is_lower(r1:Rect, r2:Rect):
    return r1.y2<r2.y1

def connect(canvas, r1:Rect,r2:Rect):
    x1,y1 = 0,0
    x2,y2 = 0,0
    if is_left(r1,r2):
        x1 = r1.x2
        x2 = voice((r2.x1+r2.x2) /2, abs(r2.x1-r2.x2)/4)
        y1 = voice((r1.y1+r1.y2) /2, abs(r1.y1-r1.y2)/4)

        if is_upper(r1, r2):
            y2 = r2.y2

        if is_lower(r1, r2):
            y2 = r2.y1

    if is_right(r1,r2):
        x1 = r1.x1
        x2 = voice((r2.x1 + r2.x2) / 2, abs(r2.x1-r2.x2)/4)
        y1 = voice((r1.y1 + r1.y2) / 2, abs(r1.y1-r1.y2)/4)

        if is_upper(r1, r2):
            y2 = r2.y2

        if is_lower(r1, r2):
            y2 = r2.y1

    line = ComplexLine(canvas, (x1, y1), (x2, y2), True, (True, True))
    canvas.draw(line.points())
    return line


class Model2(Container):

    def __init__(self):
        super().__init__()

    def boxes(self):
        return [b.box() for b in self.get_objs_list()]

    def __repr__(self):
        return f"({self.x}, {self.y},) ({self.w}, {self.h})"

    def __str__(self):
        return self.__repr__()


def draw(obj):
    canvas = obj.canvas
    canvas.draw (obj.points())

    l = obj.label
    if l:
        l.canvas.drawText(l.x,l.y,l.text)


def drawBox (obj):
    box = obj.get_box()
    canvas = obj.canvas
    canvas . drawBox(box)


class Model:
    def __init__(self):
        self.objs = []
        pass

    def add (self, obj):
        self.objs.append (obj)


    def get_objs(self):
        return self.objs

    def get_boxes (self):
        return [obj.get_box().to_array() for obj in self.objs]


    def get_mask (self):
        mask = np.zeros ([1000,1000], dtype = np.uint8)
        for obj in self.objs:
            points = obj.points()
            for p in points:
                x,y = p
                x = int(np.round(x))
                y = int(np.round(y))
                mask[1000-y,x] = 128
        return mask




class Canvas:
    def __init__(self, x,y, size):
        self.w, self.h = size
        self.x, self.y = x,y


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




def model1 ():
    image = Image.new("RGB", (1000, 1000), (255, 255, 255))
    canvas = ImageCanvas (image)


    model = Model ()


    r1 = WryRect(canvas,(300, 300), (380,420), "CA")
    draw(r1)
    model.add (r1)


    r2 = Rect(canvas,(510, 600), 120, 100, "EКБД")
    draw(r2)
    model.add (r2)

    r3 = WryRect(canvas,(20, 20), (140, 120), "СЦК")
    draw(r3)
    model.add(r3)

    r4 = Rect(canvas,(750, 750), 120, 100, "EIF")
    draw(r4)

    model.add(r4)

    model.add(connect(canvas, r1, r2))
    model.add(connect(canvas, r1, r3))
    model.add(connect(canvas, r2, r3))
    model.add(connect(canvas, r2, r4))

    image.show()

    for obj in model.get_objs():
        drawBox(obj)

    image.show()

    with open("/Users/im/1.jpg", "w") as fd:
        image.save(fd)

    mask = model.get_mask()
    img = Image.fromarray(mask)
    img.show()

#model1()

