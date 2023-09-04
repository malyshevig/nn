import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


file = "../../ttf/Sergeyp-Regular.ttf"

import bezier as bz

font = ImageFont.truetype(font=file, size = 30)

def voice(val, sigma=50):
    return np.random.normal(val, sigma)

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

            self.drawImage.point((x, self.size[1]-y), fill='black')

    def drawLabel (self, label):
        x,y = label.rect
        self.drawImage.text((x, self.size[1]-y), label.text, 'blue', font)

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
        self.lines.append(Line(xy, (x2,y2)))
        self.lines.append(Line(xy, (x3,y3)))


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
                    self.arrows.append(Arrow(xy1, "up"))

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

class Label:
    def __init__(self, rect, text):
        self.text = text
        self.rect = rect
        text_width, text_height = font.getsize(text)

    def getImage(self):
        return rect, label






class Rect:
    def __init__(self, xy1, w, h, label=""):
        (x1, y1) = xy1

        x2 = x1 + w
        y2 = y1 + h

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.label = Label((x1 + 20, (y2 + y1) / 2), label)

        self.lines = []
        self.lines.append(Line(xy1, (x1, y2)))
        self.lines.append(Line(xy1, (x2, y1)))
        self.lines.append(Line((x1, y2), (x2, y2)))
        self.lines.append(Line((x2, y1), (x2, y2)))

    def points(self):
        r = []
        l: Line
        for l in self.lines:
            r.extend(l.points())

        return r





np.random.seed()


def testRect ():
    image = Image.new("RGB", (1000, 1000), (255, 255, 255))
    drawer = Drawer(ImageDraw.Draw(image))

    r = Rect ((100,100), 200, 150, "Hello")
    drawer.draw(r.points())
    if r.label:
        drawer.drawLabel (r.label)


    r = Rect ((300,300), 60, 120, "EIF")
    drawer.drawObj(r)

    r = Rect ((510,330), 120, 60,"EKBD")
    drawer.drawObj(r)


    r = Rect ((700,400), 250, 250)
    drawer.draw(r.points())


    with open("/Users/im/2.jpg", "w") as fd:
        image.save(fd)

    image.show()


def testComplexLine():
    image = Image.new("RGB", (1000, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    drawer = Drawer(ImageDraw.Draw(image))

    r = ComplexLine((100, 100), (200, 200), True,(True,True))
    drawer.draw(r.points())

    r = ComplexLine((150, 220), (300, 500), False,(True,True))
    drawer.draw(r.points())

    r = ComplexLine((400, 600), (300, 450), False, (True, True))
    drawer.draw(r.points())

    image.show()








def drawline(xy1, xy2):
    (x1, y1) = xy1
    (x2, y2) = xy2
    n = 5
    v = 5

    xp = [(x2 - x1) / n * i + x1 for i in range(n + 1)]
    yp = [voice((y2 - y1) / n * i) + y1 for i in range(n + 1)]
    nodes = np.asfortranarray([xp, yp])
    curve1 = bz.Curve(nodes, degree=n)

    s_vals = np.linspace(0.0, 1.0, 100)
    points = curve1.evaluate_multi(s_vals)

    #    plt.plot(points[0, :], points[1, :], color="red", alpha=None)
    #    plt.show()
    image = Image.new("RGB", (1000, 1000), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    rp = [(points[0, i], points[1, i]) for i in range(len(points[0]))]

    for p in rp:
        draw.point(p, fill='white')

    with open("/Users/im/1.jpg", "w") as fd:
        image.save(fd)

def main():
    testComplexLine()
    testRect()


if __name__ == "__main__":
    main()

