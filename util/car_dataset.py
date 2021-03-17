'''
Custom Dataset Class for the Project
'''

import sys
import os
import torch
import pandas as pd

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

    def __init__(self, root_dir, image_folder_list, label_file_list):
        # Initialize Attributes
        self.image_locations = list() # List of lists
        self.label_frames = list() # List of lists

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
            self.label_frames.append(working_frame)

    def __len__(self):
        return self.validate()

    def __getitem__(self,idx):
        folder_idx, sub_idx = self.get_offset(idx)

        # Read Image
        image = Image.open(self.image_locations[folder_idx][sub_idx])
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
        for folder_idx in len(self.image_locations):
            len_in_folder = len(self.image_locations[folder_idx])
            if idx <= len_in_folder:
                # Required Index is Here!
                sub_idx = idx - offset
                return (folder_idx,)
            offset+=len_in_folder - 1

        raise DatasetError(("Index {} Out of Bounds in dataset of size"
                " {}").format(idx,len(self)))

class DatasetError(Exception):
    def __init__(self,message="Problem loading data!"):
        self.message = message
        super().__init__(self.message)

