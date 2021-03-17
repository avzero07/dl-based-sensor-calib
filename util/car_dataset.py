'''
Custom Dataset Class for the Project
'''

import sys
import os
import torch
import pd

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise DatasetError("Path {} Does not exist!".format(dir_path))

def check_file(file_path):
    if not os.path.isfile(file_path):
        raise DatasetError("File {} Does not exist!".format(file_path))

# TODO: Better name
class CarHudAnglesDataset(torch.utils.data.Dataset):
    '''
    Dataset of Images from the vehicle and associated [pitch,yaw]
    angles.
    '''

    def __init__(self, root_dir, image_folder_list, label_file_list):
        # Check if the root_dir exists
        check_dir(root_dir)
        os.chdir(root_dir)

        # Iterate through Image Folders and Check if they Exist
        for folder in image_folder_list:
            check_dir(folder)

    def __len__(self):
        pass


class DatasetError(Exception):
    def __init__(self,message="Problem loading data!"):
        self.message = message
        super().__init__(self.message)
