import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=1):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
        show_matrix(self.weight, 'conv weight ')
        show_matrix(self.bias, 'conv bias ')
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        h_start = idxh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = idxw * self.stride
                        w_end = w_start + self.kernel_size
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] \
                                                                     * self.input_pad[idxn, :, h_start: h_end, w_start : w_end]) \
                                                                     + self.bias[idxc]                                
        self.forward_time = time.time() - start_time
        return self.output

    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        # TODO: 边界扩充
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding: self.padding + input.shape[2], self.padding: self.padding + input.shape[3]] = self.input
        height_out = int((height - self.kernel_size) / self.stride + 1)
        width_out = int((width - self.kernel_size) / self.stride + 1)
        mat_w = int(self.kernel_size * self.kernel_size * self.channel_in)
        mat_h = int(height_out * width_out)

        self.col = np.empty((input.shape[0], int(mat_h), int(mat_w)))
        col_index = 0
        for x in range(int(height_out)):
            for y in range(int(width_out)):
                bias_x = x * self.stride
                bias_y = y * self.stride
                self.col[:, col_index, :] = self.input_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(input.shape[0], -1)
                col_index = col_index + 1
        output = np.matmul(self.col, self.weight.reshape(-1, self.weight.shape[-1])) + self.bias
        self.output = np.moveaxis(output.reshape(input.shape[0], height_out, width_out, self.channel_out), 3, 1)
        self.forward_time = time.time() - start_time
        return self.output



    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        start_time = time.time()

        height_out = int((self.input.shape[2] + 2 * self.padding - self.kernel_size) / self.stride + 1)
        width_out = int((self.input.shape[3] + 2 * self.padding - self.kernel_size) / self.stride + 1)

        top_diff_col = np.transpose(top_diff, [1, 0, 2, 3]).reshape(top_diff.shape[1], -1)
    
        tmp = np.transpose(self.col.reshape(-1, self.col.shape[-1]), [1, 0])
        self.d_weight = np.matmul(tmp, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
        self.d_bias = top_diff_col.sum(axis=1)
        
        backward_col = np.empty((top_diff.shape[0], self.input.shape[2] * self.input.shape[3], self.kernel_size * self.kernel_size * self.channel_out))
        pad_height = int(((self.input.shape[2] - 1) * self.stride + self.kernel_size - height_out) / 2)
        pad_width = int(((self.input.shape[3] - 1) * self.stride + self.kernel_size - width_out) / 2)
        top_diff_pad = np.zeros((top_diff.shape[0], top_diff.shape[1], height_out + 2 * pad_height, width_out + 2 * pad_width))
        top_diff_pad[:, :, pad_height: height_out + pad_height, pad_width: width_out + pad_width] = top_diff
        col_index = 0
        for x in range(self.input.shape[2]):
            for y in range(self.input.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                backward_col[:, col_index, :] = top_diff_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(top_diff.shape[0], -1)
                col_index = col_index + 1

        weight_tmp = np.transpose(self.weight, [3, 1, 2, 0]).reshape(self.channel_out, -1, self.channel_in)[:, ::-1, :].reshape(-1, self.channel_in)
        bottom_diff = np.matmul(backward_col, weight_tmp)
 
        bottom_diff = np.transpose(bottom_diff.reshape(top_diff.shape[0], self.input.shape[2], self.input.shape[3], self.input.shape[1]), [0, 3, 1, 2])
        
        self.backward_time = time.time() - start_time
        return bottom_diff
    
    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失 # [N, C, H, W]
            
                        self.d_weight[:, :, :, idxc] += self.input_pad[idxn,idxc,idxh*self.stride:idxh*self.stride+self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size] * top_diff[idxn, idxc, idxh, idxw]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        print(bottom_diff.shape)    
        bottom_diff = bottom_diff[:,:,self.padding:-self.padding,self.padding:-self.padding]
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        show_matrix(self.weight, 'conv weight ')
        show_matrix(self.bias, 'conv bias ')
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=1):
        self.kernel_size = kernel_size
        self.stride = stride
        ### adding
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1: # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup

        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        h_start = idxh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = idxw * self.stride
                        w_end = w_start + self.kernel_size
                        # 提取局部区域
                        local_region = self.input[idxn, idxc, h_start:h_end, w_start:w_end]
                        self.output[idxn, idxc, idxh, idxw] = np.max(local_region)
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        self.input = input # [N, C, H, W]
        height_out = int((self.input.shape[2] - self.kernel_size) / self.stride + 1)
        width_out = int((self.input.shape[3] - self.kernel_size) / self.stride + 1)
        mat_w = self.kernel_size * self.kernel_size
        mat_h = height_out * width_out

        col = np.empty((input.shape[0], self.input.shape[1], mat_h, mat_w))
        cur = 0
        for x in range(height_out):
            for y in range(width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                col[:, :, cur, :] = self.input[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(input.shape[0], input.shape[1], -1)
                cur = cur + 1

        self.output = np.max(col, axis=3, keepdims=True)
        max_index = np.argmax(col.reshape(input.shape[0], input.shape[1], height_out, width_out, self.kernel_size * self.kernel_size), axis=4)
        self.max_elements = np.zeros((input.shape[0], self.input.shape[1], height_out, width_out, self.kernel_size * self.kernel_size))

        n, c, h, w = self.max_elements.shape[: 4]
        N, C, H, W = np.ogrid[:n, :c, :h, :w]
        self.max_elements[N, C, H, W, max_index] = 1
        self.output = self.output.reshape(input.shape[0], input.shape[1], height_out, width_out)
        return self.output

    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        bottom_diff = np.zeros(self.input.shape)
        contrib = self.max_elements * (top_diff.reshape(list(top_diff.shape) + [1]))
        for x in range(top_diff.shape[2]):
            for y in range(top_diff.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] += contrib[:, :, x, y, :].reshape(top_diff.shape[0], top_diff.shape[1], self.kernel_size, self.kernel_size)
        return bottom_diff
 
    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        max_index = np.unravel_index(max_index, [self.kernel_size, self.kernel_size])
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw] 
        show_matrix(top_diff, 'top_diff--------')
        show_matrix(bottom_diff, 'max pooling d_h ')
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        show_matrix(bottom_diff, 'flatten d_h ')
        return bottom_diff
