# Morphing

## 实验结果

对以下两张源图片进行Morphing，

<table><tr>
<td align="center"><img src=wolf2leo/a.png border=0></td>
<td align="center"><img src=wolf2leo/b.png border=0></td>
</tr></table>

得到的结果为

<table><tr>
<td align="center"><img src=wolf2leo/src.gif border=0></td>
<td align="center"><img src=wolf2leo/ave.gif border=0></td>
<td align="center"><img src=wolf2leo/dst.gif border=0></td>
</tr></table>

图片序列在`wolf2leo`文件夹中。

## 实验过程

### 选取图片的对应点

在这里使用tkinker库中的canvas类编写了一个脚本`click_correspondences.py`用于选取对应点。

![windows](assets/windows.png)

具体方法很简单，左键增加一个点，右键撤销一个或一对点（在windows下无效）。轮流点击两幅图片的对应位置，点击`save`将点对保存为字典`dict.txt`。

### 计算平均形状

```python
def weighted_average_points(start_points, end_points, percent=0.5):
    """ Weighted average of two sets of supplied points
    :param start_points: *m* x 2 array of start face points.
    :param end_points: *m* x 2 array of end face points.
    :param percent: [0, 1] percentage weight on start_points
    :returns: *m* x 2 array of weighted average points
    """
    if percent <= 0:
        return end_points
    elif percent >= 1:
        return start_points
    else:
        return np.asarray(start_points * percent + end_points * (1 - percent), np.int32)
```

### 向平均形状变形

#### 计算仿射变换参数

```python
def triangular_affine_matrices(vertices, src_points, dest_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dest_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dest_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat
```

根据仿射变换，对于面片内的坐标$X$满足三角形顶点的矩阵$srcMat_{m\times m}$与仿射参数$\alpha Mat_{m\times 1}$的关系

$$\left\{\begin{matrix} srcMat\cdot\alpha Mat = X\\ dstMat\cdot\alpha Mat = X'  \end{matrix}\right.$$

要找到目标三角面片中的点$X'$对应在源三角面片中的坐标，即

$$X=srcMat\cdot dstMat^{-1}\cdot X'$$

所以通过`triangular_affine_matrices`函数，计算每个三角面片的$srcMat\cdot dstMat^{-1}$矩阵，后续通过这个矩阵，可以方便的求出源坐标。

#### 对目标图像的每个面片进行变形

```python
def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None
```

先计算目标区域的大小，即`dst_points`中x,y两个方向中最大和最小值所决定的矩形。

使用`delaunay.find_simplex(roi_coords)`遍历目标中每个点，寻找它们在哪个三角面片中，并返回面片的索引。

接下来先遍历面片`delaunay.simplices`，再遍历面片中的点`coords`。

这里用到了`numpy`的一些特性。`roi_tri_indices == simplex_index`，当`ndarray`类型和`int`类型进行比较时，相等位置会返回`True`，不等的位置会返回`False`。例如

```python
In[4]: a = np.asarray([3,2,1,0,1,2,3])
In[4]: a == 1
Out[5]: array([False, False,  True, False,  True, False, False])
```

对`ndarray`类型可以使用等长的布尔型的`array`进行索引。例如

```python
In[5]: b=np.asarray([True,False,False,True,False,True,True])
In[6]: a[b]
Out[6]: array([3, 0, 2, 3])
```

所以，`coords = roi_coords[roi_tri_indices == simplex_index]`这条语句将取出一个面片中所有坐标。

最后使用$X=srcMat\cdot dstMat^{-1}\cdot X'$计算源图像中的位置，得到源图像中的坐标后，通过双线性插值，取到源图像中的值并赋值到新图像中。

## 实验总结

实验效果比较明显，应该算成功了把。最开始写程序的时候因为对`numpy`不熟悉，写的第一个版本十分低效的，而且效果也不太好。后来借鉴了网上的一些代码，也明白了自己代码的问题。运行效率提高了非常多。

<table>
<tr><td align="center">edition 1</td><td align="center">edition 2</td></tr>
  <tr>
<td align="center"><img src=wolf2leo/bad_1.gif border=0></td>
<td align="center"><img src=wolf2leo/bad_2.gif border=0></td>
</tr></table>

