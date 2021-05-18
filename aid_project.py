import cv2
import matplotlib.pyplot as plt
from PIL import Image


def cvt(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


img1 = Image.open("pics/fig00001.png")
img2 = cv2.imread('pics/fig00003.png', 1)
img3 = cv2.imread('pics/fig00004.jpg', 1)
img4 = Image.open("pics/fig00002.png")

plt.figure("fig_1")
plt.subplot(2, 2, 1)
plt.imshow(img1)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cvt(img2))
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cvt(img3))
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img4)
plt.axis('off')

plt.show()
