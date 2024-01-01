import numpy as np
import struct
import os
import scipy.io
import time

class ContentLossLayer(object):
    def __init__(self):
        print('\tContent loss layer.')
    def forward(self, input_layer, content_layer):
         # TODO： 计算风格迁移图像和目标内容图像的内容损失
    
        loss = np.square(input_layer - content_layer).sum() / (2 * input_layer.size)
        return loss
    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        bottom_diff = (input_layer - content_layer) /  input_layer.size
        return bottom_diff

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的Gram 矩阵(3.12)
        style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        #self.gram_style = np.zeros([style_layer.shape[0], style_layer.shape[1], style_layer.shape[1]])
        #Yiwen Xu's method
        self.gram_style = np.dot(style_layer_reshape[0,:,:], style_layer_reshape[0,:,:].T)
        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])
        for idxn in range(input_layer.shape[0]):
            #TODO: check right
            self.gram_input[idxn, :, :] = np.dot(self.input_layer_reshape[idxn,:,:], self.input_layer_reshape[idxn,:,:].T)
        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失(3.13, 3.14)
        style_diff = np.sum(np.square(self.gram_input-self.gram_style)) 
        loss = 1.0 / (4*input_layer.shape[0]*self.div) * style_diff
        return loss
    def backward(self, input_layer, style_layer):
        bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播(3.15)
            diff = self.gram_input[idxn,:,:]-self.gram_style
            bottom_diff[idxn, :, :] = 1.0 / (input_layer.shape[0]*self.div) * np.dot(diff, self.input_layer_reshape[idxn,:,:])
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
