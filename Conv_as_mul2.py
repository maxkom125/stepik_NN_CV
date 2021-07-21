#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 2021

@author: maksym
"""

import torch
from abc import ABC, abstractmethod

def convert_matrix(m, conv_m_height, conv_m_width, stride, padding = 0): #НАЛИЧИЕ PADDINGа НЕ РЕАЛИЗОВАНО!!
    m_height, m_width = m.shape
    
    converted_columns = []
    i = 0
    j = 0
    
    
    while j <= m_height - conv_m_height:
        while i <= m_width - conv_m_width:
            converted_columns.append(m[j : j + conv_m_height, i : i + conv_m_width].reshape(-1, 1))
            
            i += stride
            
        i = 0
        j += stride
        
    return torch.cat(converted_columns, 0)
        
        
    
    

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


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


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


class Conv2dMatrixV2(ABCConv2d):
    # Функция преобразования кернела в нужный формат.
    def _convert_kernel(self):
        converted_filters = []
        
        for kernel_filter in self.kernel:
            converted_filters.append(self._convert_kernel_filter(kernel_filter))
            
        return torch.cat(converted_filters, 0)
    
    def _convert_kernel_filter(self, kernel_filter):
        x = []
        for conv_matrix in kernel_filter:
            x.append(conv_matrix.reshape(1, -1))
            
        return torch.cat(x, 1)
        

    # Функция преобразования входа в нужный формат.
    def _convert_input(self, torch_input, output_height, output_width):
        converted_images = []
        
        for img in torch_input:
            converted_images.append(self._convert_image(img))
            
        return torch.cat(converted_images, 1)
    
    def _convert_image(self, img):
        x = []
        conv_m_size = self.kernel_size
        
        for color in img:
            x.append(convert_matrix(color, conv_m_size, conv_m_size, self.stride))
            
        return torch.cat(x, 0)

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        converted_kernel = self._convert_kernel()
        converted_input = self._convert_input(torch_input, output_height, output_width)
        
        #print("converted:", converted_kernel, converted_input, sep='\n')

        conv2d_out_alternative_matrix_v2 = converted_kernel @ converted_input
        return conv2d_out_alternative_matrix_v2.transpose(0, 1).view(torch_input.shape[0],
                                                     self.out_channels, output_height,
                                                     output_width).transpose(1, 3).transpose(2, 3)

