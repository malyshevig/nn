from figures import *
from PIL import Image, ImageDraw, ImageFont
import draw
import random


sizes = [
    (40,60), (100, 130), (80,60), (120, 180),
    (70,90), (60,100),  (40,90), (50,50)
]

class ModelDesc:
    pass



class Generator:

    @staticmethod
    def generate(n: int) -> Model2:
        model = Model2()
        for i in range (n):
            id = f"box {i}"
            x = random.randint(100,900)
            y = random.randint(100,900)
            n = random.randint(0,len(sizes)-1)
            model.put(Rect(id, (x,y),sizes [n]))

        return model

class Loader:

    @staticmethod
    def read_models (path = "./data/models"):
        pass


def model2():

    model = Model2()
    model.put(Rect("b1",(10,10), (100,200)))
    model.put(Rect("b2", (300,20), (100, 200)))
    model.put(Rect("b3", (600,400), (35, 40)))
    model.put(Rect("b4", (600, 750), (100, 100)))
    model.put(Rect("b5", (100, 750), (100, 100)))
    model.put(Rect("b6", (500, 500), (200, 150)))
    model.put(Connect("c1", "b1", "b2"))
    model.put(Connect("c2", "b2", "b3"))
    model.put(Connect("c3", "b1", "b4"))
    model.put(Connect("c4", "b2", "b5"))
    model.put(Connect("c5", "b1", "b6"))

    model.get_obj("b1").put (Label("l1", (10,50),"Hello"))


    image = Image.new("RGB", (800, 800), (255, 255, 255))
    mask = Image.new("RGB", (800, 800), (0, 0, 0))


    dr = draw.ModelDrawer(image, mask)
    dr.drawModel (model)
    image.show()
    image.save("./data/images/model2.jpg")


def model3():

    model = Model2()
    model.put(Rect("b1",(10,10), (100,200)))
    model.put(Rect("b2", (300,20), (100, 200)))
    model.put(Rect("b3", (600,400), (35, 40)))
    model.put(Rect("b4", (600, 800), (100, 100)))
    model.put(Rect("b5", (100, 800), (100, 100)))
    model.put(Rect("b6", (500, 500), (200, 150)))

    image = Image.new("RGB", (800, 800), (255, 255, 255))
    mask = Image.new("RGB", (800, 800), (0, 0, 0))

    dr = draw.ModelDrawer(image, mask)
    dr.drawModel (model)
    image.show()


def main():
    g = Generator()
    model = g.generate(5)
    image = Image.new("RGB", (1000, 1000), (255, 255, 255))
    mask = Image.new("RGB", (1000, 1000), (0, 0, 0))

    dr = draw.ModelDrawer(image,mask)
    dr.drawModel (model)
    image.show()
    mask.show()
    print (f"{model.boxes()}")




if __name__ == "__main__":
    model2()