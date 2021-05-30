import numpy as np

img = np.matrix([[6, 7, 6, 3, 5, 3, 0, 5, 6],
                 [7, 1, 7, 5, 2, 3, 3, 7, 1],
                 [1, 7, 5, 1, 7, 5, 7, 7, 7],
                 [7, 7, 0, 5, 0, 5, 2, 4, 2],
                 [5, 3, 6, 0, 3, 6, 4, 1, 1],
                 [0, 6, 7, 2, 4, 2, 1, 1, 2],
                 [2, 1, 5, 0, 6, 5, 6, 2, 4],
                 [4, 3, 6, 0, 6, 5, 2, 6, 3],
                 [7, 7, 5, 6, 1, 1, 4, 2, 2]])

his1, bins = np.histogram(img.ravel(), 8, [0, 7])

ps = his1 / sum(his1)
cs = sum(ps)

# plt.bar([x for x in range(8)], cdf_normalised)
