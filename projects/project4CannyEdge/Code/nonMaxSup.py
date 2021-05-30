'''
  File name: nonMaxSup.py
  Author: Tarmily Wen
  Date created: Dec. 8, 2019
'''

import numpy as np
from helpers import get_edge_angle

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''


def nonMaxSup(Mag, Ori, grad_Ori):
    ###############################################################################
    # Your code here: do the non maximum suppression
    ###############################################################################
    suppressed = np.copy(Mag)
    suppressed.fill(0)
    shape = suppressed.shape

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            g0 = Mag[i, j]
            edge_ori=get_edge_angle(Ori[i,j])

            if edge_ori == 0:
                if g0 >= Mag[i, j + 1] and g0 >= Mag[i, j - 1]:
                    suppressed[i, j] = 1
                else:
                    suppressed[i, j] = 0
            elif edge_ori == np.pi / 4:
                if g0 >= Mag[i + 1, j + 1] and g0 >= Mag[i - 1, j - 1]:
                    suppressed[i, j] = 1
                else:
                    suppressed[i, j] = 0
            elif edge_ori == np.pi / 2:
                if g0 >= Mag[i + 1, j] and g0 >= Mag[i - 1, j]:
                    suppressed[i, j] = 1
                else:
                    suppressed[i, j] = 0
            else:
                if g0 >= Mag[i + 1, j - 1] and g0 >= Mag[i - 1, j + 1]:
                    suppressed[i, j] = 1
                else:
                    suppressed[i, j] = 0

    return suppressed
