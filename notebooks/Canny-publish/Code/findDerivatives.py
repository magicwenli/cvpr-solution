'''
  File name: findDerivatives.py
  Author: Tarmily Wen
  Date created: Dec. 8, 2019
'''

import numpy as np
from scipy import signal
import cv2

'''
  File clarification:
    Compute gradient put ginformation of the inrayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the derivatives along x-axis
    - Output Magy: H x W matrix represents the derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''


def findDerivatives(I_gray):
    # smoothing kernels
    gaussian = np.array(
        [[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159.0

    # kernel for x and y gradient
    dx = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    dy = np.asarray([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    ###############################################################################
    # Your code here: calculate the gradient magnitude and orientation
    ###############################################################################

    Gx= signal.convolve2d(gaussian, np.rot90(dx, 2), 'same')
    Gy= signal.convolve2d(gaussian, np.rot90(dy, 2), 'same')

    Magx = signal.convolve2d(I_gray, np.rot90(Gx, 2), 'same')
    Magy = signal.convolve2d(I_gray, np.rot90(Gy, 2), 'same')

    Mag = signal.convolve2d(Magx, np.rot90(dy, 2), 'same')

    Ori = np.arctan2(Magy, Magx)/np.pi*90

    return Mag, Magx, Magy, Ori


if __name__ == '__main__':
    I_g = np.arange(0, 81, 1)
    I_g = I_g.reshape(9, 9)
    findDerivatives(I_g)
