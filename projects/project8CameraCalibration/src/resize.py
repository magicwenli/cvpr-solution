import glob
import os

from PIL import Image


def ResizeImage(filein, fileout, width=800, height=800, type='jpeg'):
    img = Image.open(filein)
    out = img.resize((width, height), Image.ANTIALIAS)
    # resize image with high-quality
    out.save(fileout, type)


if __name__ == '__main__':
    width = 800
    height = 800

    images = glob.glob('../imgs' + os.sep + '**.jpg')
    for img in images:
        file_out=os.path.splitext(img)[0] + '_resize.jpg'
        ResizeImage(img,file_out)
