# attempted implementation of local representation alignment -- https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4389
# as in this paper:
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from functions import *
from models import *
import sys

batch_size = 10
num_batches = 10
num_test_batches = 20
lr = 0.01
amortised_learning_rate = 0.001
layer_sizes = [784, 300, 100, 10]
n_layers = len(layer_sizes)
n_epochs = 101
inference_thresh = 0.5
beta =0.1
gamma = 0.9
f = tanh
df = tanhderiv

w3 = np.random.normal(0,0.05, [10,100])
w2 = np.random.normal(0,0.05,[100,300])
w1 = np.random.normal(0,0.05, [300,784])
E3 = same_sign(w3.T, np.random.normal(0,0.05, [10,100]).T)
E2 = same_sign(w2.T, np.random.normal(0,0.05, [100,300]).T)

train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
#num_batches = len(train_set)// batch_size
print("Num Batches",num_batches)
img_list = [np.array([np.array(train_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_batches)]
label_list = [np.array([onehot(train_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
test_img_list = [np.array([np.array(test_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_test_batches)]
test_label_list = [np.array([onehot(test_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_test_batches)]

img_batch = img_list[0]
label_batch = label_list[0]
e3s = []
for i in range(1000):
    x = img_batch
    y = label_batch
    h1 = np.dot(w1, x)
    z1 = f(h1)
    h2 = np.dot(w2,z1)
    z2 = f(h2)
    h3 = np.dot(w3,z2)
    z3 = f(h3)
    y3 = deepcopy(y)
    e3 = -(y3 - z3)
    y2 = f(h2 - beta * (np.dot(E3,e3)))
    e2 = -1 * (y2 - z2)
    y1 = f(h1 - beta *(np.dot(E2,e2)))
    e1 = -1 * (y1 - z1)
    #update _weights
    dw3 = np.dot(e3 * df(h3),z2.T)
    dw2 = np.dot(e2 * df(h2),z1.T)
    dw1 = np.dot(e1 * df(h1),x.T)
    dE3 = gamma * dw3.T
    dE2 = gamma * dw2.T
    w3 -= lr * dw3
    w2 -= lr * dw2
    w1 -= lr * dw1
    E3 -= lr * dE3
    E2 -= lr * dE2
    print("Error: ", np.sum(e3))
    e3s.append(deepcopy(np.sum(e3)))

plt.plot(e3s)
plt.show()
