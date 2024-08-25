import math
from typing import Union

import numpy as np
import torch


def debug_nn_size(network, state):
    x0 = torch.as_tensor(np.array(state)).float()
    print("input shape:", x0.shape)

    with torch.no_grad():
        x = x0
        for layer in network:
            x0 = layer(x)
            print(f"{type(layer)} {x.shape} -> {x0.shape}")
            x = x0


def debug_count_params(network):
    params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("params: ", params)


def debug_get_conv_out(conv_nn, shape):
    o = conv_nn(torch.zeros(1, *shape))
    return int(np.prod(o.size()))


# from https://github.com/pytorch/pytorch/issues/79512
def conv_2d_return_shape(
    dim: tuple[int, int],
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple] = (1, 1),
    padding: Union[int, tuple] = (0, 0),
    dilation: Union[int, tuple] = (1, 1),
) -> tuple[int, int]:
    """
    Calculates the return shape of the Conv2D layer.
    Works on the MaxPool2D layer as well

    See Also: https://github.com/pytorch/pytorch/issues/79512

    See Also: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    See Also: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    Args:
        dim: The dimensions of the input. For instance, an image with (h * w)
        kernel_size: kernel size
        padding: padding size
        dilation: dilation size
        stride: stride size

    Returns:
        Dimensions of the output of the layer
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(stride, int):
        stride = (stride, stride)
    h_out = conv_output(dim[0], padding[0], dilation[0], kernel_size[0], stride[0])
    w_out = conv_output(dim[1], padding[1], dilation[1], kernel_size[1], stride[1])
    return h_out, w_out


# from: https://github.com/Adillwma/Pytorch-Conv-Net-Output-Size-Calculator/blob/main/Conv_layers_output_size_calculator_V2.py
def conv_3d_return_shape(
    dim: tuple[int, int, int],
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple] = (1, 1, 1),
    padding: Union[int, tuple] = (0, 0, 0),
    dilation: Union[int, tuple] = (1, 1, 1),
) -> tuple[int, int, int]:
    """
    Calculates the return shape of the Conv3D layer.

    Args:
        dim: The dimensions of the input. For instance, an image with (h * w * d)
        kernel_size: kernel size
        padding: padding size
        dilation: dilation size
        stride: stride size

    Returns:
        Dimensions of the output of the layer
    """
    # H_in = height of the inputs\n
    # W_in = width of the inputs\n
    # D_in = depth of the input (Only used if one of the 3D conv types is selected above)\n

    # Following values can be input as either an integer or an integer tuple of same dimension as convoloution 2 or 3)\n
    # K = kernel size \n
    # P = padding \n
    # S = stride  \n
    # D = dilation \n
    # O = output padding (used only in the Transposed convolutions)\n

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    h_out = conv_output(dim[0], padding[0], dilation[0], kernel_size[0], stride[0])
    w_out = conv_output(dim[1], padding[1], dilation[1], kernel_size[1], stride[1])
    d_out = conv_output(dim[2], padding[2], dilation[2], kernel_size[2], stride[2])
    return h_out, w_out, d_out


def conv_output(input, padding, dilation, kernel_size, stride):
    out = math.floor(
        (input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    return out
