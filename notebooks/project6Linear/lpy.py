# -*- coding:utf-8 -*-
#
#  Author       : magicwenli
#  Date         : 2021-05-29 19:47:45
#  LastEditTime : 2021-05-29 19:47:45
#  Description  :
#  FilePath     : /com_vis/notebooks/project6Linear/lpy.py
#

import numpy as np
import os
from helperP import *

base_dir = 'DATASET/'

age_train, features_train = prepare_data('train', base_dir)
age_val, features_val = prepare_data('val', base_dir)
_, features_test = prepare_data('test', base_dir)
show_data(base_dir)

# [markdown]
# # Implement Closed Form Solution
# ```
# Arguments:
#     age          -- numpy array, shape (n, )
#     features     -- numpy array, shape (n, 2048)
# Returns:
#     weights      -- numpy array, (2048, )
#     bias         -- numpy array, (1, )
# ```

#


def closed_form_solution(age, features):
    # Preprocess
    H = features
    ones = np.ones(len(H))
    H = np.column_stack((ones, H))  # 按列合并
    Y = age

    weights = None
    weights = np.linalg.inv(H.T.dot(H)).dot(H.T).dot(Y)
    bias = weights[0]
    weights = weights[1:]

    return weights, bias


w, b = closed_form_solution(age_train, features_train)
loss, pred = evaluate(w, b, age_val, features_val)
print("Your validate loss is:", round(loss, 3))


prediction = test(w, b, features_test, 'cfs.txt')
print("Test results has saved to cfs.txt")
print(prediction[:10])

# [markdown]
# # Implement Gradient descent
# Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.
#
# ```
# Arguments:
#     age          -- numpy array, label, (n, )
#     feature      -- numpy array, features, (n, 2048)
# Return:
#     weights      -- numpy array, (2048, )
#     bias         -- numpy array, (1, )
# ```

#


def gradient_descent(age, feature):
    assert len(age) == len(feature)

    # Init weights and bias
    weights = np.random.randn(2048, 1)
    bias = np.random.randn(1, 1)

    ws_new = weights
    b_new = bias

    n = feature.shape[0]

    # Learning rate
    lr = 10e-3

    for e in range(epoch):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################

        # forward pass
        weights = ws_new
        bias = b_new

        # calculate loss

        # calculate gradient
        l_w = 1/n * (np.sum(feature*feature, axis=1).T*weights + bias * np.sum(feature, axis=1).T + np.sum(np.multiply(feature, age[:, np.newaxis]), axis=1).T)
        l_b = 1/n * (np.sum(np.multiply(feature, weights[:, np.newaxis]), axis=1) + bias + np.sum(age, axis=1))

        # update weights
        ws_new = weights-lr*l_w
        b_new = bias-lr*l_b

        if momentum:
            pass  # You  can also consider the gradient descent with momentum

    return weights, bias

# [markdown]
# # Train and validate


#
w, b = gradient_descent(age_train, features_train)
loss, pred = evaluate(w, b, age_val, features_val)
print("Your validate score is:", round(loss, 3))

# # [markdown]
# # #  Test and Generate results file

# #
# prediction = test(w, b, features_test, 'gd.txt')
# print("Test results has saved to gd.txt")
# print(prediction[:10])

# # [markdown]
# # # Implement Stochastic Gradient descent
# # Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.
# # ```
# # Arguments:
# #     age          -- numpy array, label, (n, )
# #     feature      -- numpy array, features, (n, 2048)
# # Return:
# #     weights      -- numpy array, (2048, )
# #     bias         -- numpy array, (1, )
# # ```

# #
# def stochastic_gradient_descent(age, feature):
#     # check the inputs
#     assert len(age) == len(feature)

#     # Set the random seed
#     np.random.seed(0)

#     # Init weights and bias
#     weights = np.random.rand(2048, 1)
#     bias = np.random.rand(1, 1)

#     # Learning rate
#     lr = 10e-5

#     # Batch size
#     batch_size = 16

#     # Number of mini-batches
#     t = len(age) // batch_size

#     for e in range(epoch_sgd):
#         # Shuffle training data
#         n = np.random.permutation(len(feature))

#         for m in range(t):
#             # Providing mini batch with fixed batch size of 16
#             batch_feature = feature[n[m * batch_size : (m+1) * batch_size]]
#             batch_age = age[n[m * batch_size : (m+1) * batch_size]]

#             ##########################################################################
#             # TODO: YOUR CODE HERE
#             ##########################################################################
#             # forward pass

#             # calculate loss

#             # calculate gradient

#             # update weights


#             if momentum:
#                 pass # You can also consider the gradient descent with momentum

#         print('=> epoch:', e + 1, '  Loss:', round(loss,4))
#     return weights, bias

# # [markdown]
# # # Train and validate

# #
# w, b = stochastic_gradient_descent(age_train, features_train)
# loss, pred = evaluate(w, b, age_val, features_val)
# print("Your validate score is:", round(loss, 3))

# # [markdown]
# # # Test and Generate results file

# #
# prediction = test(w, b, features_test, 'sgd.txt')
# print("Test results has saved to sgd.txt")
# print(prediction[:10])
