import torch
import torch.nn as nn
import torch.nn.functional as F

def train_net(network,dataset_obj,epoch_count,device):
    pass

# Functions to Do CNN Size Math
def get_conv_output_size(ip_height,ip_width,ip_channels,
        kernel,padding,stride,filters):

    op_height = int((ip_height - kernel + 2*padding)/(stride)) + 1
    op_width = int((ip_width - kernel + 2*padding)/(stride)) + 1
    op_channels = filters
    return (op_height,op_width,op_channels)

def get_layer_param(ip_height,ip_width,ip_channels,layer):
    kernel = get_value_from_obj(layer.kernel_size)
    padding = get_value_from_obj(layer.padding)
    stride = get_value_from_obj(layer.stride)
    try:
        out_channels = layer.out_channels
    except torch.nn.modules.module.ModuleAttributeError:
        out_channels = ip_channels

    return (ip_height,ip_width,ip_channels,kernel,padding,stride,
        out_channels)

def get_value_from_obj(obj,index=0):
    '''
    Function to help the get_layer_param() function for situations
    when a layer object has attributes that could be int or tuples.
    '''
    if isinstance(obj,tuple):
        return obj[index]
    else:
        return obj

class NetworkError(Exception):
    def __init__(self,message="Network Related Error!"):
        self.message = message
        super().__init__(self.message)
