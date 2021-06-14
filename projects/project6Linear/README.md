# Proj6 Linear Regression & Perceptron

## Linear Regression

### 解析解

要使损失函数

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D%3D%5Csum_%7Bi%7D%5Cleft(%5Ctheta%20%5Ccdot%20x_%7Bi%7D%2B%5Cmathrm%7Bb%7D-%5Chat%7By%7D_%7Bi%7D%5Cright)%5E%7B2%7D%0A">

最小，即损失函数的梯度为0

<img src="https://render.githubusercontent.com/render/math?math=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bl%7D%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Ctheta%7D%3D%5Csum_%7Bi%7D%5Cleft(%5Ctheta%20%5Ccdot%20x_%7Bi%7D%2B%5Cmathrm%7Bb%7D-%5Chat%7By%7D_%7Bi%7D%5Cright)%20x_%7Bi%7D%3D0%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cmathrm%7Bb%7D%7D%3D%5Csum_%7Bi%7D%5Cleft(%5Ctheta%20%5Ccdot%20x_%7Bi%7D%2B%5Cmathrm%7Bb%7D-%5Chat%7By%7D_%7Bi%7D%5Cright)%3D0%0A%5Cend%7Barray%7D%5Cright.">

为了便于计算，令

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bw%7D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7D%20b%5C%5C%20%5Ctheta%20%5Cend%7Barray%7D%5Cright%5D%2C%5C%20%5Cmathbf%7BH%7D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7D%20ones(1%2C%5C%20len(%5Cmathbf%7BX%7D%5B0%5D))%5C%5C%20%5Cmathbf%7BX%7D%20%5Cend%7Barray%7D%5Cright%5D%0A">

损失函数的梯度即为

<img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bw%7D%7D%3D%5Csum_i%20(%5Cmathbf%7Bw%7D%20%5Ccdot%20h_i-%5Chat%7By%7D_i)h_i%20%3D%200">

解得

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bw%7D%20%3D%20(%5Cmathbf%7BH%7D%5E%5Cintercal%20%5Ccdot%20%5Cmathbf%7BH%7D%20)%5Ccdot%20%5Cmathbf%7BH%7D%5E%5Cintercal%20%5Ccdot%20%5Cmathbf%7BY%7D%0A">


`python`代码：

```python
def closed_form_solution(age, features):
    # Preprocess
    H = features
    ones = np.ones(len(H))
    H = np.column_stack((ones,H))  # 按列合并
    Y = age
    # Define parameter weights
    
    ##########################################################################
    # TODO: YOUR CODE HERE
    ########################################################################## 
    # calculate the closed form solution
    weights = None

    weights = np.linalg.inv(H.T.dot(H)).dot(H.T).dot(Y)

    bias    = weights[0]
    weights    = weights[1:]
    
    return weights, bias

```

### 梯度下降

梯度下降算法是通过优化来求解最佳权重和偏置的方法。首先可以通过 `numpy` 随机变量生成器 `random` 中高斯噪声生成器来生成一个初始的权重和偏置<img src="https://render.githubusercontent.com/render/math?math=%5Cleft(%5Cmathbf%7Bw%7D%5E%7B0%7D%2C%20b%5E%7B0%7D%5Cright)"> ，然后迭代更新。假设目前已经更新了步<img src="https://render.githubusercontent.com/render/math?math=%5Cleft(%5Cmathbf%7Bw%7D%5E%7Bt%7D%2C%20b%5E%7Bt%7D%5Cright)">，你已经获得了更新后的参数 ，这时候你可以将这个更新后的参数代入到上述求解梯度的公式中，获得在当前点的梯度。

<img src="https://render.githubusercontent.com/render/math?math=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bl%7D%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bw%7D%7D%3D%5Cfrac%7B1%7D%7B3995%7D%20%5Csum_%7Bi%3D1%7D%5E%7B3995%7D%5Cleft(X_%7Bi%7D%20%5Cmathbf%7Bw%7D%5E%7Bt%7D%2Bb%5E%7Bt%7D-y_%7Bi%7D%5Cright)%20X_%7Bi%7D%5E%7BT%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20b%7D%3D%5Cfrac%7B1%7D%7B3995%7D%20%5Csum_%7Bi%3D1%7D%5E%7B3995%7D%5Cleft(X_%7Bi%7D%20%5Cmathbf%7Bw%7D%5E%7Bt%7D%2Bb%5E%7Bt%7D-y_%7Bi%7D%5Cright)%0A%5Cend%7Barray%7D%5Cright.">

然后根据这个梯度，结合步长$\alpha$可以更新权重和偏置

<img src="https://render.githubusercontent.com/render/math?math=%5Cleft(%5Cmathbf%7Bw%7D%5E%7Bt%2B1%7D%2C%20b%5E%7Bt%2B1%7D%5Cright)%3D%5Cleft(%5Cmathbf%7Bw%7D%5E%7Bt%7D%2C%20b%5E%7Bt%7D%5Cright)-%5Calpha%5Cleft(%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bw%7D%7D%2C%20%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20b%7D%5Cright)%0A">

```python
def gradient_descent(age, feature):
    assert len(age) == len(feature)

    # Init weights and bias
    weights = np.random.randn(2048, 1)
    bias = np.random.randn(1, 1)
    m,n = feature.shape
    # Learning rate
    lr = 10e-3

    # control the times of processing
    for e in range(epoch):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        l_w = np.zeros((2048, 1), dtype=float)
        l_b = 0
        
        for i in range(m):
            tmp = feature[i, :].reshape(-1, 1)
            l_w += (np.dot(feature[i, :], weights) + bias - float(age[i])) * tmp
        l_w /= m
        for i in range(m):
            l_b += np.dot(feature[i, :], weights) + bias - float(age[i])
        l_b /= m

        weights -= lr * l_w
        bias -= lr * l_b
        if momentum:
            pass  # You  can also consider the gradient descent with momentum
    return weights, bias
```

### 随机梯度下降

随机梯度下降方法，每次从全部样本中随机抽取 B 个样本，然后利用这 B 个样本构建一个损失函数，计算梯度，然后更新权重和偏置。除了梯度计 算不是相对于全部的 3995 个样本，而是相对于随机抽取的 B 个样本外， 其他更新迭代步骤都是一样的。

```python


def stochastic_gradient_descent(age, feature):
    # check the inputs
    assert len(age) == len(feature)
    
    # Set the random seed
    np.random.seed(0)

    # Init weights and bias
    weights = np.random.rand(2048, 1)
    bias = np.random.rand(1, 1)

    # Learning rate
    lr = 10e-5

    # Batch size
    batch_size = 16
 
    # Number of mini-batches
    t = len(age) // batch_size

    for e in range(epoch_sgd):
        # Shuffle training data
        n = np.random.permutation(len(feature))  
        
        for m in range(t):
            # Providing mini batch with fixed batch size of 16
            batch_feature = feature[n[m * batch_size : (m+1) * batch_size]]
            batch_age = age[n[m * batch_size : (m+1) * batch_size]]
            
            ##########################################################################
            # TODO: YOUR CODE HERE
            ########################################################################## 
            
            l_w = np.zeros((2048, 1), dtype=float)
            l_b = 0
            
            for i in range(batch_size):
                tmp = batch_feature[i, :].reshape(2048, 1)
                l_w += (np.dot(batch_feature[i, :], weights) + bias - float(batch_age[i])) * tmp
            l_w /= batch_size
            
            for i in range(batch_size):
                l_b += np.dot(batch_feature[i, :], weights) + bias - float(batch_age[i])
            l_b /= batch_size
            
            weights -= lr * l_w
            bias -= lr * l_b
   
                
            if momentum:
                pass # You can also consider the gradient descent with momentum
        
        print('=> epoch:', e + 1, '  Loss:', round(loss,4))
    return weights, bias
```

## Linear Perceptron

### 线性感知机

1)在`__init__`这个函数中初始化权重以及偏置，可以通过` numpy` 随机变 量生成器` random` 中高斯噪声生成器来生成一个初始的权重和偏置。

使用`w = np.random.randn(num_dims, 1)`生成一个`(num_dims, 1)`维的随机权重。

使用`b = np.random.randn(1, 1)`生成一个1维的随机偏置。

2. 填充 `predict` 函数，这个函数返回两个值，一个是 `preds`，这个记录 着线性感知机在进行` sign `函数运算之前的实数值，另外一个是 `y_hat` 记 录着进行 `sign `函数运算之后的结果

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7Bpreds%7D_i%20%3D%20%5Cmathbf%7Bw%7D%5Ccdot%5Cmathbf%7BX_i%7D%20%2B%20b%5C%20%2C%5Cquad%20i%5Cin%5B0%2C%5Ctext%7Bnum_sample%7D)">

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7By%5C_hat%7D%20%3D%20sign(%5Cmathbf%7Bpreds%7D)">

 3)填充 `update `函数，这个函数实现的是梯度下降法，轮询一次所有的样本，更新参数。

每次`update`需要根据所有分类不正确的结果修正。

当<img src="https://render.githubusercontent.com/render/math?math=y_i%5Ccdot%20y%5C_hat_i%5Cle%200">时，说明分类不正确，此时需要

<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bw%7D%20%3D%20%5Cmathbf%7Bw%7D%20-%20lr%20%5Ccdot%20y_i%20%5Ccdot%20%5Cmathbf%7BX%7D_i%0A">

<img src="https://render.githubusercontent.com/render/math?math=b%3Db-%20lr%5Ccdot%20y_i">

`python`代码如下：

```python
class PrimalPerceptron(object):
    def __init__(self, x, y, w=None, b=None):
        num_sample, num_dims = x.shape

        np.random.seed(0)
        ####################################
        # TODO: YOUR CODE HERE： init weights
        ####################################
        if not w:
            w = np.random.randn(num_dims, 1)
        if not b:
            b = np.random.randn(1, 1)

        self.x, self.y, self.w, self.b = x, y, w, b
        self.lr = 0.1

        self.y_hat = None
        self.predict()

    def predict(self):
        ####################################
        # TODO: YOUR CODE HERE, forward
        ####################################
        num_sample, num_dims = self.x.shape
        preds = np.zeros([num_sample, 1])

        for i in range(num_sample):
            tmp = self.x[i].reshape(1,-1)
            preds[i,0] = tmp.dot(self.w)+self.b

        self.y_hat = self.sign(preds)
        return preds, self.y_hat

    def update(self):
        ####################################
        # TODO: YOUR CODE HERE, backward
        ####################################
        # update the weights and bias

        w_index = (self.y*self.y_hat < 0)  # wrong sample bool index
        for i in range(w_index.size):
            if not w_index[i,0]:
                a=self.y[i,0]
                b=self.x[i,:].reshape(1,-1).T

                self.w -= self.lr * a * b
                self.b -= self.lr * a
        
        return

    def sign(self,v):
        v[v > 0] = 1
        v[v <= 0] = -1
        return v
```

