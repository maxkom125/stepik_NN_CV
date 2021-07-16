# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:49:02 2021

@author: Maksym
"""

import torch
from abc import ABC, abstractmethod


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
    
    #print("FINAL TEST")
    #print(custom_conv2d_out)
    #print(conv2d_out)
    
    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


# Сверточный слой через циклы.
class Conv2dLoop(ABCConv2d):
    def __call__(self, input_tensor):
        output_tensors = torch.tensor([])
        
        for tensor in input_tensor: # for images in batch
            output_img = torch.tensor([])
            
            for out_channel in range(self.out_channels): # for filtres in kernel                
                #print("output_tensor 1 ", output_img)
                kernel_filter = self.kernel[out_channel]
                
                batch_size, out_channels, output_height, output_width = \
                    calc_out_shape(list(input_tensor.size()), self.out_channels, self.kernel_size, self.stride, 0)
                
                output_filter = torch.zeros(output_height, output_width)
                
                for in_channel in range(self.in_channels): # for channels in image
                    #print("output_tensor 2 ", output_filter)
                    
                    wx_tensor = self.Get_wx_tensor(tensor[in_channel], kernel_filter[in_channel],\
                                                   output_height, output_width)
                    
                    output_filter += wx_tensor
                    
                    #print("output_tensor 3 ", output_filter)
               
                output_img = torch_append(output_img, output_filter)
                
            output_tensors = torch_append(output_tensors, output_img)
            
        return output_tensors
    
    def Get_wx_tensor(self, tensor, conv_matrix, output_height, output_width):
        stride = self.stride
        kernel_size = self.kernel_size
        
        wx_tensor = torch.zeros(output_height, output_width)
        #tensor = get_padding2d(tensor)
        
        #print("Get_wx_tensor tensor: ", tensor)
        
        for j in range(output_height):
            for i in range(output_width):
                wx_tensor[i][j] = \
                self._matrix_mul_sum(tensor[i * stride : i * stride + kernel_size,\
                                       j * stride : j * stride + kernel_size],\
                               conv_matrix)
                    
        #print("Get_wx_tensor wx_tensor: ", wx_tensor)
        return wx_tensor
                 
    def _matrix_mul_sum(self, A, B):
        #print("A B: ", A, B)
        return (A * B).sum()  
            
def get_padding2d(input_images):
    padded_images = torch.nn.functional.pad(input_images, pad=(1,1,1,1)) 
    return padded_images.float() 

def torch_append(dest_tensor, add_tensor):
    #print("torch_append ????", dest_tensor)
    
    if torch.equal(dest_tensor, torch.tensor([])):
        #print(*add_tensor.shape, add_tensor, add_tensor.expand(1, *add_tensor.shape))
        return add_tensor.expand(1, *add_tensor.shape)
    
    #print("torch_append", dest_tensor, add_tensor)
    #print("torch_append", dest_tensor.tolist(), [add_tensor.tolist()])
    return torch.tensor(dest_tensor.tolist() + [add_tensor.tolist()])

# Корректность реализации определится в сравнении со стандартным слоем из pytorch.
# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
print(test_conv2d_layer(Conv2dLoop))