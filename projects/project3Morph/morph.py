# -*- coding: utf-8 -*-
# @Time         : 2021/5/18 20:23
# @Author       : magicwenli
# @FileName     : morph.py
# @GitHub       : https://github.com/magicwenli
# @Description  :

import ast

from PIL import Image

from tools import *


def morph(img_src, img_dst, s_points, d_points, percent=0.5, save_flag=1):
    shape = img_src.shape
    result_points = weighted_average_points(s_points, d_points, percent)
    dst_img1 = warp_image(img_src, s_points, result_points, shape)
    dst_img2 = warp_image(img_dst, d_points, result_points, shape)
    ave = weighted_average(dst_img1, dst_img2, percent)
    if save_flag:
        cv2.imwrite(folder + 'avg_{:.2f}.png'.format(percent), ave)
        cv2.imwrite(folder + 'src_{:.2f}.png'.format(percent), dst_img1)
        cv2.imwrite(folder + 'dst_{:.2f}.png'.format(percent), dst_img2)
        print('Write {:.2f}'.format(percent))
    return dst_img1, dst_img2, ave


def make_gif(target_path, images, duration=80):
    images[0].save(target_path,
                   save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=1)


if __name__ == '__main__':
    # os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])

    folder = 'tmp/'
    img_1 = cv2.imread(folder + 'a.png')
    img_2 = cv2.imread(folder + 'b.png')
    points_dict_path = folder + 'dict.txt'
    with open(points_dict_path, 'r') as f:
        dict_points = ast.literal_eval(f.read())

    src_points = np.asarray(list(dict_points.keys()))
    dst_points = np.asarray(list(dict_points.values()))

    result = []
    for i in np.linspace(0, 1, 30, True):
        result.append(morph(img_1, img_2, src_points, dst_points, i))
    make_gif(folder + 'src.gif', [Image.fromarray(cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB)) for x in result])
    make_gif(folder + 'dst.gif', [Image.fromarray(cv2.cvtColor(x[1], cv2.COLOR_BGR2RGB)) for x in result])
    make_gif(folder + 'ave.gif', [Image.fromarray(cv2.cvtColor(x[2], cv2.COLOR_BGR2RGB)) for x in result])
