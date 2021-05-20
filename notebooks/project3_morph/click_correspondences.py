# -*- coding: utf-8 -*-
# @Time         : 2021/5/18 12:34
# @Author       : magicwenli
# @FileName     : click_correspondences.py
# @GitHub       : https://github.com/magicwenli
# @Description  : "<Button-1>", add_point
#                 "<Button-2>", del_point


import random
import tkinter as tk

from PIL import Image, ImageTk
import json

img_1 = Image.open('../../pics/dog_1.png')
# img_1 = img_1.resize((200, 200), Image.ANTIALIAS)
img_2 = Image.open('../../pics/dog_2.png')
# img_2 = img_2.resize((200, 200), Image.ANTIALIAS)

BORDER = 50
COLOR = 'blue'
points_ids = []  # 存放圆圈id
next_img = 0  # 0:下次需要点击图片1， 1: 下次需要点击图片2
img1_to_img2 = dict()
xy_img1 = []  # 存放图片1的坐标


def add_point(event):
    global next_img, COLOR, img1_to_img2, xy_img1
    x = event.x
    y = event.y

    if not next_img:
        # 轮到图片 1
        COLOR = random_color()
        if event.x <= BORDER // 2 or event.x >= img_1.size[0] + BORDER // 2 \
                or event.y <= BORDER // 2 or event.y >= img_1.size[0] + BORDER // 2:
            pass
        else:
            x -= BORDER // 2 + 1
            y -= BORDER // 2 + 1
            print("clicked at img 1: ", (x, y))
            draw_circle(event.x, event.y, color=COLOR)
            xy_img1.append((x, y))
            next_img = 1

    else:
        # 轮到图片 2
        if event.x <= img_1.size[0] + BORDER or event.x >= img_1.size[0] + img_2.size[0] + BORDER \
                or event.y <= BORDER // 2 or event.y >= img_1.size[0] + BORDER // 2:
            pass
        else:
            x -= img_1.size[0] + BORDER + 1
            y -= BORDER // 2 + 1
            print("clicked at img 2: ", (x, y))
            draw_circle(event.x, event.y, color=COLOR)
            img1_to_img2[xy_img1[-1]] = (x, y)
            next_img = 0

        # print(img1_to_img2)


def del_point(event):
    global next_img, img1_to_img2, xy_img1
    try:
        if next_img:
            canvas.delete(points_ids.pop())
            print('delete img 1 point')
            next_img = 0
        else:
            canvas.delete(points_ids.pop())
            canvas.delete(points_ids.pop())
            xy = xy_img1.pop()
            del img1_to_img2[xy]
            print('delete img 1, img 2 point')
    except IndexError:
        print('canvas is empty')


def random_color():
    rand = lambda: random.randint(50, 255)
    return '#%02X%02X%02X' % (rand(), rand(), rand())


def draw_circle(x, y, color="blue"):
    r = 4
    id = canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, width=2)
    points_ids.append(id)


def save():
    with open('dict.txt', 'w') as f:
        f.write(str(img1_to_img2))
    print('dict saved')
    img_1.save('a.png', 'png')
    img_2.save('b.png', 'png')


if __name__ == '__main__':
    # print(1)
    assert img_1.size == img_2.size

    window = tk.Tk(className="Click Correspondences")

    canvas = tk.Canvas(window, width=img_1.size[0] + img_2.size[0] + BORDER * 3 // 2, height=img_1.size[1] + BORDER)
    canvas.pack()
    image_tk_1 = ImageTk.PhotoImage(img_1)
    image_tk_2 = ImageTk.PhotoImage(img_2)

    canvas.create_image(BORDER // 2, BORDER // 2, anchor='nw', image=image_tk_1)
    canvas.create_image(img_1.size[0] + BORDER, BORDER // 2, anchor='nw', image=image_tk_2)

    canvas.bind("<Button-1>", add_point)
    canvas.bind("<Button-2>", del_point)

    button = tk.Button(window, text="save", fg='black', command=save)
    button.pack()

    tk.mainloop()
