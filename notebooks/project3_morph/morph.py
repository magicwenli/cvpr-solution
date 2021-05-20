# -*- coding: utf-8 -*-
# @Time         : 2021/5/18 20:23
# @Author       : magicwenli
# @FileName     : morph.py
# @GitHub       : https://github.com/magicwenli
# @Description  :

import ast

from tools import *


def morph(img_src, img_dst, s_points, d_points, percent=0.5):
    shape = img_src.shape
    result_points = weighted_average_points(s_points, d_points, percent)
    dst_img1 = warp_image(img_src, s_points, result_points, shape)
    dst_img2 = warp_image(img_dst, d_points, result_points, shape)
    ave = weighted_average(dst_img1, dst_img2, percent)
    cv2.imwrite('avg_{:.2f}.png'.format(percent), ave)


if __name__ == '__main__':
    img_1 = cv2.imread('a.png')
    img_2 = cv2.imread('b.png')
    points_dict_path = 'dict.txt'
    with open(points_dict_path, 'r') as f:
        dict_points = ast.literal_eval(f.read())

    src_points = np.asarray(list(dict_points.keys()))
    dst_points = np.asarray(list(dict_points.values()))

    for i in np.linspace(0, 1, 30, True):
        morph(img_1, img_2, src_points, dst_points, i)
