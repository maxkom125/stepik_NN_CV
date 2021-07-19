#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 02:34:10 2021

@author: maksym
"""

import torch
from abc import ABC, abstractmethod

def Conv_as_mul_transform(conv_matrix, input_matrix, output_height, output_width): # Матрица нужного вида
    input_width, input_height = input_matrix.shape
    height = output_height * output_width
    
    conv_matrix_width = conv_matrix.shape[0]

    ans = -1

    for j in range(height):
        curr_row = torch.tensor([])   
        i = 0
        
        if i != 0:
            curr_row = torch.zeros(j * input_width)
        
        # отступ на длину строк, уже умноженных
        i += j * input_width
        
        for wi in range(conv_matrix_width): # == conv_matrix_height !!!!
        
            curr_row = torch.cat((curr_row, conv_matrix[wi]), 0)
            
            curr_row = torch.cat((curr_row, torch.zeros(input_width - conv_matrix_width)), 0)
            
        # отступ на суммарную длину строк, которые ниже приложенной матрицы
        curr_row = torch.cat((curr_row, torch.zeros((input_height - conv_matrix_width - j) * input_width)), 0)
        
        if ans != -1:
            ans = torch.cat((ans, torch.unsqueeze(curr_row, 0)), 1)
        else:
            ans = torch.unsqueeze(curr_row, 0)
           
    return ans


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dMatrix(ABCConv2d):
    # Функция преобразование кернела в матрицу нужного вида.
    def _unsqueeze_kernel(self, torch_input, output_height, output_width):
        kernel_unsqueezed = torch.cat(self.kernel_matrix_format(torch_input, output_height, output_width), 0) # преобразованный кернел
        #print(kernel_unsqueezed, sep='\n')
        return kernel_unsqueezed
    
    def kernel_matrix_format(self, torch_input, output_height, output_width):
        x = []
        for kernel_filter in self.kernel:
                                                                                                                       #  m_1
            x.append(torch.cat(self.kernel_filter_format(kernel_filter, torch_input, output_height, output_width), 1)) #  ...
                                                                                                                       #  m_n
        return x
    
    def kernel_filter_format(self, kernel_filter, torch_input, output_height, output_width):
        x = []
        for in_cannel in range(len(kernel_filter)): # conv_matrix in kernel_filter:
            
            x.append(Conv_as_mul_transform(kernel_filter[in_cannel], torch_input[0, in_cannel], output_height, output_width))
            
        return x
            

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        kernel_unsqueezed = self._unsqueeze_kernel(torch_input, output_height, output_width)
        result = kernel_unsqueezed @ torch_input.view((batch_size, -1)).permute(1, 0)
        return result.permute(1, 0).view((batch_size, self.out_channels,
                                          output_height, output_width))

# print(test_conv2d_layer(Conv2dMatrix))