# -*- coding: utf-8 -*-
# @Time         : 2021/5/18 13:48
# @Author       : magicwenli
# @FileName     : project_3_morph.py
# @GitHub       : https://github.com/magicwenli
# @Description  :

import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt

class Morph:

    def __init__(self, src_img, dst_img, dict_points):
        self.tmp_dela_tri = []
        self.dst_dela_tri = []
        self.src_dela_tri = []

        self.tmp2src = dict()
        self.tmp2dst = dict()

        self.src_img = src_img
        self.dst_img = dst_img

        self.src_points = []
        self.dst_points = []
        self.tmp_points = []
        self.t = 0.5
        self.get_points(dict_points)
        self.get_tem_point()
        self.delaunay()

    def set_t(self, t):
        self.t = t
        self.get_tem_point()
        self.delaunay()

    def get_points(self, dict_points):
        for key in dict_points:
            self.src_points.append(key)
            self.dst_points.append(dict_points[key])

    def get_tem_point(self):
        if self.t < 0 or self.t > 1:
            raise NameError
        self.tmp_points = (np.dot((1 - self.t), self.src_points) + np.dot(self.t, self.dst_points)).astype(int)

        tmp = [tuple(x) for x in self.tmp_points]
        src = [tuple(x) for x in self.src_points]
        dst = [tuple(x) for x in self.dst_points]

        self.tmp2src = dict(zip(tmp, src))
        self.tmp2dst = dict(zip(tmp, dst))

    def delaunay(self):
        self.tmp_dela_tri = Delaunay(self.tmp_points)
        self.dst_dela_tri = Delaunay(self.dst_points)
        self.src_dela_tri = Delaunay(self.src_points)

    def get_matrix_T(self, tri_cords):
        return np.vstack((tri_cords[:, 0], tri_cords[:, 1], np.asarray([1, 1, 1])))

    def tri_plot(self):
        plt.triplot(self.tmp_points[:, 0], self.tmp_points[:, 1], self.tmp_dela_tri.simplices)


    def morph1d(self, target='src'):
        if target == 'src':
            dst_img = self.src_img
            src2dst = self.tmp2src
        elif target == 'dst':
            dst_img = self.dst_img
            src2dst = self.tmp2dst
        else:
            raise NameError

        new_img=np.zeros(dst_img.shape)
        # plt.triplot(src_points[:, 0], src_points[:, 1], src_tri.simplices)
        # plt.imshow(dst_img)
        # plt.show()
        # 后向变换 维护一个点与点对应的矩阵
        for x, col in enumerate(new_img):
            for y, val in enumerate(col):
                # tri_id 该坐标所在的Delaunay三角形的序号
                src_tri_id = self.tmp_dela_tri.find_simplex((x, y))
                if src_tri_id == -1:
                    new_img[x, y] = 255
                else:
                    # tri_cords 该坐标所在的三角形的顶点坐标
                    src_tri_cords = self.tmp_points[self.tmp_dela_tri.simplices[src_tri_id]]

                    # 取得对应三角形顶点坐标
                    dst_tri_cords = np.asarray([src2dst[tuple(x)] for x in src_tri_cords])
                    # print(dst_tri_cords)

                    T_src = self.get_matrix_T(src_tri_cords)
                    T_dst = self.get_matrix_T(dst_tri_cords)

                    X = np.asarray([[x], [y], [1]])
                    para = np.linalg.inv(T_src).dot(X)
                    new_cord = T_dst.dot(para).T
                    new_cord = np.round(new_cord).astype(int)

                    # print(new_cord)
                    old_val = dst_img[new_cord[0], new_cord[1]]
                    new_img[x, y] = old_val

        print("t = {}".format(self.t))
        return new_img

    def average(self, img_1, img_2, scale):
        avg_img = np.add(np.dot(img_2, scale), np.dot(img_1, 1 - scale))
        return avg_img

