import sys
from PIL import Image, ImageDraw, ImageFont

print(sys.path)
from figures import Rect, Figure, Model2, Label
from graph.bez2 import voice
from bez2 import BezierRect, ComplexLine
from figures import Connect

file = "./data/ttf/Sergeyp-Regular.ttf"

font = ImageFont.truetype(font=file, size = 20)


def is_left(r1:Rect, r2:Rect):
    return r1.x2 < r2.x1


def is_right(r1:Rect, r2:Rect):
    return r1.x1 > r2.x2


def is_upper(r1:Rect, r2:Rect):
    return r1.y1>r2.y1


def is_lower(r1:Rect, r2:Rect):
    return r1.y2<r2.y1


def connect(r1:Rect,r2:Rect):
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

    line = ComplexLine((x1, y1), (x2, y2), True, (True, True))
    return line


class Canvas:

    def __init__(self, x,y, size, transformX, transformY):
        self.x = x
        self.y = y
        self.size = size
        self.transformX = transformX
        self.transformY = transformY


class ModelDrawer:

    def getGraphFigure (self, fig: Figure, mod):
        if isinstance(fig, Rect):
            x,y = fig.x, fig.y
            w,h = fig.w, fig.h

            return BezierRect((x, y), (x + w, y + h))


        if isinstance (fig,Connect):
            r1 = mod.get_obj(fig.id1)
            r2 = mod.get_obj(fig.id2)

            line = connect(r1,r2)
            return line

        raise RuntimeError(f'{fig.__class__} type is unknown')


    def __init__ (self, image: Image, mask: Image):
        self.image = image
        self.mask = mask
        self.drawImage = ImageDraw.Draw(image)
        self.drawMask = ImageDraw.Draw(mask)
        self.size = image.size


    def drawModel(self, mod: Model2):
        w,h = self.size
        canvas = Canvas (0,0,self.size,lambda x:x, lambda y: h - y)
        for obj in mod.get_objs():
            self.draw(obj,canvas,mod)


    def draw(self, obj, canvas, mod):
        gobj = self.getGraphFigure(obj, mod)

        points = [(canvas.transformX(x),canvas.transformY(y)) for x,y in gobj.points() ]
        self.drawImage.point(points, fill='black')
        self.drawMask.point(points,fill="grey")

        for sobj in obj.get_objs():
            sx = obj.x
            sy = obj.y
            scanvas = Canvas(sx, sy, (10,10), lambda x: canvas.transformX(sx + x),
                             lambda y: canvas.transformY(sy + y))

            if isinstance(sobj, Label):
                lx = scanvas.transformX(sobj.x)
                ly = scanvas.transformY(sobj.y)

                sz = font.getsize (sobj.label)

                lx = obj.w / 2 - sz [0] / 2
                ly = obj.h / 2

                lx = voice (lx,10)
                ly = voice (ly,10)

                lx = scanvas.transformX(lx)
                ly = scanvas.transformY(ly)

                self.drawImage.text((lx, ly), sobj.label,'blue', font)
            else:
                self.draw (self,sobj,scanvas, mod)








