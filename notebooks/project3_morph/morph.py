# -*- coding: utf-8 -*-
# @Time         : 2021/5/18 20:23
# @Author       : magicwenli
# @FileName     : morph.py
# @GitHub       : https://github.com/magicwenli
# @Description  :

import ast
import cv2
from project_3_morph import Morph


def avg_img(c_1, c_2, dict_points, i):
    m = Morph(c_1, c_2, dict_points)
    m.set_t(i / 10)
    a = m.morph1d(target='src')
    b = m.morph1d(target='dst')
    return m.average(a, b, i / 10)


if __name__ == '__main__':

    img_1 = cv2.imread('a.png', 1)
    img_2 = cv2.imread('b.png', 1)

    points_dict = 'dict.txt'
    with open(points_dict, 'r') as f:
        dict_points = ast.literal_eval(f.read())

    b_1, g_1, r_1 = cv2.split(img_1)
    b_2, g_2, r_2 = cv2.split(img_2)

    for i in range(1, 10, 1):
        cb = avg_img(b_1, b_2, dict_points, i)
        cg = avg_img(g_1, g_2, dict_points, i)
        cr = avg_img(r_1, r_2, dict_points, i)
        c = cv2.merge([cb, cg, cr])
        cv2.imwrite('avg_{:.2f}.png'.format(i / 10), c)
