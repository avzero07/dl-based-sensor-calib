import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def run_training(network,train_dataset_loader,epochs,device):
    network.train()
    loss_over_training = []
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
        loss_over_training.append(mean_train_loss)
        if (epoch%10 == 0):
            save_model(epoch,network.state_dict(),network.optimizer.state_dict(),
                    mean_train_loss)
    return loss_over_training

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

def run_inference_single(network,sample,device):
    '''
    The sample passed in as input is supposed to be a single
    sample from the DataSet Class.
    '''
    network.eval()
    test_loss = 0
    correct = 0

    # Unsqueeze to account for Batch Dimension
    sample['image'] = torch.unsqueeze(sample['image'],0)
    sample['angles'] = torch.unsqueeze(sample['angles'],0)

    with torch.no_grad():
        num = 0
        num+=1
        data = (sample['image']).double()
        target = (sample['angles']).double()
        data, target = data.to(device), target.to(device)
        output = network(data)
        test_loss += F.mse_loss(output,target).item()
        pred = output

    print("Num = {}\nLen = {}".format(num,1))
    test_loss_average = test_loss/1
    print("Test Set Average Loss {:.4f}".format(test_loss))

    print("Real = {}".format(target))
    print("Pred = {}".format(output))
    return output

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

# Functions to Save and Load Model Checkpoints
def save_model(epoch,model_state_dict,optimizer_state_dict,loss,path=os.getcwd(),
        name='checkpoint'):
    '''
    Wrapper function for the torch save function. This will call the
    version of save that saves the entire model for general checkpoint
    backup, inference and the resume training.
    '''
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss},
        os.path.join(path,'{}-{}.pt'.format(name,epoch)))

def load_model(new_model,new_optimizer,path_to_checkpoint):
    '''
    Wrapper function to simplify the use of torch load. Pass in the newly
    initialized model and optimizer and this function will take care of the
    rest.

    NOTE: After this call finishes, make sure to call model.eval() or
    model.train() depedning on inference / training.
    '''
    checkpoint_dict = torch.load(path_to_checkpoint,map_location=get_device())
    new_model.load_state_dict(checkpoint_dict['model_state_dict'])
    new_optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    epoch = checkpoint_dict['epoch']
    loss = checkpoint_dict['loss']
    return epoch, loss

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
