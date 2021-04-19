import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def run_training(network,train_dataset_loader,epochs,device):
    network.train()
    for epoch in range(epochs):
        loss_over_epoch = []
        for batch_idx, sample in tqdm(enumerate(train_dataset_loader)):
            data = (sample['image']).double()
            target = (sample['angles']).double()
            data, target = data.to(device), target.to(device)
            network.optimizer.zero_grad()
            output = network(data)
            loss = (F.mse_loss(output,target))
            loss_over_epoch.append(loss.item())
            loss.backward()
            network.optimizer.step()

            '''
            if batch_idx%10 == 0:
                print("Train Epoch: {}\tBatch: {}\tLoss:"
                      " {:.6f}".format(epoch,batch_idx,loss.item()))
            '''
        mean_train_loss = torch.mean(torch.tensor(loss_over_epoch)).item()
        print("Train Epoch: {}\tMean Loss:"
              " {:.6f}".format(epoch,mean_train_loss))

def run_inference(network,test_dataset_loader,device):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        num = 0
        for sample in test_dataset_loader:
            num+=1
            data = (sample['image']).double()
            target = (sample['angles']).double()
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.mse_loss(output,target).item()
            pred = output
    
    print("Num = {}\nLen = {}".format(num,len(test_dataset_loader.dataset)))
    test_loss_average = test_loss/len(test_dataset_loader.dataset)
    print("Test Set Average Loss {:.4f}".format(test_loss))

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

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

    except AttributeError:
        out_channels = ip_channels
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
