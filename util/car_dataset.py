'''
Custom Dataset Class for the Project
'''

import sys
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, utils
from torchvision.transforms import ToTensor

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise DatasetError("Path {} Does not exist!".format(dir_path))

def check_file(file_path):
    if not os.path.isfile(file_path):
        raise DatasetError("File {} Does not exist!".format(file_path))

# TODO: Better name
class CarDataset(torch.utils.data.Dataset):
    '''
    Dataset of Images from the vehicle and associated [pitch,yaw]
    angles.
    '''

    def __init__(self, root_dir, image_folder_list, label_file_list,
            transform=None):
        # Initialize Attributes
        self.image_locations = list() # List of lists
        self.label_frames = list() # List of lists
        self.transform = transform

        # Check if the root_dir exists
        check_dir(root_dir)
        os.chdir(root_dir)

        # Iterate through Image Folders and Check if they Exist
        folder_index = -1
        for folder in image_folder_list:
            folder_index += 1
            check_dir(folder)
            # If they do, create sub_list
            self.image_locations.append(list())

            # populate self.image_locations
            os.chdir(folder)
            work_list = sorted(os.listdir())
            for item in work_list:
                if '.jpg' not in item: # TODO: Better filters?
                    work_list.remove(item)
                # Get Full Path to Image and append
                self.image_locations[folder_index].append(os.path.join(folder,item))
            # Head back upto root_dir
            os.chdir(root_dir)

        # Iterate through Label Files and Check if they Exist
        label_file_index = -1
        for label_file in label_file_list:
            label_file_index += 1
            check_file(label_file)
            # If File Exists, Read into self.label_frames
            working_frame = pd.read_csv(label_file,sep=' ',names=['pitch','yaw'])
            #TODO: Dealing with NaN should happen around here
            '''
            Can't just drop NaN from the dataframe without reconciling
            the image list.
            '''
            self.label_frames.append(working_frame)

        # Run Validate To Catch any Error
        self.validate()

    def __len__(self):
        return self.validate()

    def __getitem__(self,idx):
        folder_idx, sub_idx = self.get_offset(idx)

        # Read Image
        image = Image.open(self.image_locations[folder_idx][sub_idx])
        image = ToTensor()(image)
        if self.transform:
            image = perform_transform(self.transform,image)
        # Retrieve Label
        angles = self.label_frames[folder_idx].iloc[sub_idx].values
        angles = torch.tensor(angles)

        sample = {'image':image, 'angles':angles}
        return sample

    def validate(self):
        '''
        The number of images and labels should be the same. This
        function will validate.

        Returns the count.
        '''
        l_images=0
        for folder in self.image_locations:
            l_images += len(folder)

        l_labels=0
        for label_frame in self.label_frames:
            l_labels += len(label_frame)

        if l_images == l_labels:
            return l_images

        else:
            raise DatasetError(("Error! Images [{}] and Labels [{}]"
                               " Mismatch! Check Dataset").format(l_images,l_labels))

    def get_offset(self,idx):
        '''
        Takes in index and returns a tuple [folder_idx,sub_idx] which
        helps quickly locate and item from the dataset.
        '''
        offset = 0
        for folder_idx in range(len(self.image_locations)):
            len_in_folder = len(self.image_locations[folder_idx])
            if idx < len_in_folder:
                # Required Index is Here!
                return (folder_idx,idx)
            offset+=len_in_folder
            idx -= offset

        raise DatasetError(("Index {} Out of Bounds in dataset of size"
                " {}").format(idx,len(self)))

    def has_nan(self,item):
        '''
        Returns True if item has a NaN value in angles.
        '''
        return any(torch.isnan(item['angles']))

    def drop_nan(self):
        '''
        Labelframes may contain NaN values. Rows with NaN should be
        dropped. Corresponding images should also be discarded.

        Loops through dataset and drops items in the image list and
        corresponding dataframe.
        '''
        new_image_locations = list()
        new_label_frames = list()
        for df in self.label_frames:
            new_label_frames.append(df.iloc[0:0,:].copy())
            new_image_locations.append(list())

        # Start Loop
        for ind in range(len(self)):
            if self.has_nan(self[ind]):
                continue
            else:
                folder_idx, sub_idx = self.get_offset(ind)
                # Copy to New image list
                loc = (self.image_locations[folder_idx][sub_idx])
                new_image_locations[folder_idx].append(loc)
                # Copy to New Dataframe
                val = self.label_frames[folder_idx].iloc[sub_idx]
                new_df = new_label_frames[folder_idx].append(val)
                new_label_frames[folder_idx] = new_df

        self.image_locations = new_image_locations
        self.label_frames = new_label_frames

def get_norm_param(image):
    '''
    Returns a list of per-channel mean and std
    for the input image. Used to normalize the input.
    '''
    num_channels = image.size()[-3]
    mean_list = list()
    std_list = list()
    for ch in range(num_channels):
        mean_list.append(image[ch,:,:].mean().item())
        std_list.append(image[ch,:,:].std().item())
    return mean_list, std_list

def perform_transform(transform_list,image):
    '''
    Transforms is a list of transforms that will be applied
    sequentially on the input image.

    Note that normalization will always be the last step.

    Returns the modified image tensor back to caller.

    eg: transform = ['gs','ccrop','norm']
    '''
    if 'ccrop' in transform_list:
        c_crop_dim = [int(0.5*image.size()[-2]),image.size()[-1]]
        tf = transforms.CenterCrop(c_crop_dim)
        image = tf(image)

    if 'gs' in transform_list:
        tf = transforms.Grayscale(num_output_channels=1)
        image = tf(image)

    if 'norm' in transform_list:
        mean_list, std_list = get_norm_param(image)
        tf = transforms.Normalize(mean_list,std_list)
        image = tf(image)

    return image

def perform_augment(augment_list,image):
    '''
    Similar to transformation but this is to perform
    random augmentation useful for training. Should
    not be called in eval mode.
    '''
    pass

class DatasetError(Exception):
    def __init__(self,message="Problem loading data!"):
        self.message = message
        super().__init__(self.message)

