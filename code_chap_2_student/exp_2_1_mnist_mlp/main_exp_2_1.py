import os
import sys
from stu_upload.layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from stu_upload.mnist_mlp_cpu import MNIST_MLP, build_mnist_mlp
import numpy as np
import struct
import time


def evaluate(mlp):
    pred_results = np.zeros([mlp.test_data.shape[0]])
    for idx in range(mlp.test_data.shape[0]//mlp.batch_size):
        batch_images = mlp.test_data[idx*mlp.batch_size:(idx+1)*mlp.batch_size, :-1]
        prob = mlp.forward(batch_images)
        pred_labels = np.argmax(prob, axis=1)
        pred_results[idx*mlp.batch_size:(idx+1)*mlp.batch_size] = pred_labels
    if mlp.test_data.shape[0] % mlp.batch_size >0: 
        last_batch = mlp.test_data.shape[0]/mlp.batch_size*mlp.batch_size
        batch_images = mlp.test_data[-last_batch:, :-1]
        prob = mlp.forward(batch_images)
        pred_labels = np.argmax(prob, axis=1)
        pred_results[-last_batch:] = pred_labels
    accuracy = np.mean(pred_results == mlp.test_data[:,-1])
    print('Accuracy in test set: %f' % accuracy)

if __name__ == '__main__':
    h1,h2,e=256,128,50
    mlp=MNIST_MLP(hidden1=h1,hidden2=h2,max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.load_model('mlp-%d-%d-%depoch.npy'%(h1,h2,e))
    evaluate(mlp)
