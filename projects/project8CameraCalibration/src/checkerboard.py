import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def Checkerboard(N,n):
    """N: size of board; n=size of each square; N/(2*n) must be an integer """
    if (N%(2*n)):
        print('Error: N/(2*n) must be an integer')
        return False
    a = np.concatenate((np.zeros(n),np.ones(n)))
    b=np.pad(a,int((N**2)/2-n),'wrap').reshape((N,N))
    return (b+b.T==1).astype(int)


# 3+6n<2np
# n: 图像个数，20
# p: 点的个数，

colors = ['white', 'black']
cmap = mpl.colors.ListedColormap(colors)

B=Checkerboard(180,18)
plt.imshow(B,cmap=cmap)

plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("../checkerboard.png")
plt.show()
