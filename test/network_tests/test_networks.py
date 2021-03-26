'''
Tests Networks
'''
import pytest
import random
import string
import tempfile
import shutil
import subprocess as sp
import os

from PIL import Image
from torch import nn as nn
# TODO: Currently assumes that pytest runs from project root
from util.car_dataset import check_dir, check_file, CarDataset, DatasetError
from networks.common import NetworkError, get_conv_output_size, get_layer_param
from networks.CNNbase import CNNBasic

# Helpers

def rand_string(length=5):
    return ''.join(random.choices(string.ascii_letters +
        string.digits, k=length))

# Fixtures
@pytest.fixture(scope="module")
def temp_dir():
    '''
    Create a temp directory and move over two video files and
    corresponding label files for the tests in this module
    to use.

    The tempdir and its contents are cleaned up at the end of the
    module.
    '''
    with tempfile.TemporaryDirectory() as tempdir:
        curr_dir = os.getcwd()

        # Copy Over Two Videos and TextFiles from Dataset
        files = ['0','3']
        extensions = ['.txt','.hevc']
        data_dir = os.path.join("labeled")

        for f in files:
            for ext in extensions:
                full_src = os.path.join(data_dir,''.join([f,ext]))
                shutil.copy(full_src,tempdir)

        # Run Split Code
        abs_path_util = os.path.abspath("util")
        op = sp.run(["python",os.path.join(abs_path_util,"hevc.py"),tempdir])
        assert not op.returncode, "{}".format(op)

        yield tempdir

        # Go back to curr_dir before cleanup
        os.chdir(curr_dir)

@pytest.fixture(scope="module")
def loaded_dataset(temp_dir):
    '''
    Loads part of the car_dataset and returns the CarDataset object.
    '''
    image_folder_list = ['0_Frames','3_Frames']
    label_file_list = ['0.txt','3.txt']
    car_data = CarDataset(temp_dir,image_folder_list,label_file_list)

    yield car_data

@pytest.fixture(scope="module")
def network():
    yield CNNBasic()

# Start Tests

@pytest.mark.parametrize(
        "ip,op",
        [((28,28,1,3,0,1,32),(26,26,32)),
         ((26,26,32,3,0,1,64),(24,24,64)),
         ((24,24,64,2,0,2,64),(12,12,64)),
         ((24,24,64,2,0,0,64),None)
        ])
def test_get_conv_output_size(ip,op):
    try:
        op_test = get_conv_output_size(*ip)

        assert op_test == op, ("Mismatch!\n Expected: {}, Got:"
            " {}".format(op,op_test))

    except ZeroDivisionError:
        pass

@pytest.mark.parametrize(
        "ip,op",
        [((28,28,1,nn.Conv2d(1,32,3,1)),(28,28,1,3,0,1,32)),
         ((26,26,32,nn.Conv2d(32,64,3,1)),(26,26,32,3,0,1,64)),
         ((24,24,64,nn.MaxPool2d(2)),(24,24,64,2,0,2,64)),
        ])
def test_get_layer_param(ip,op):
    op_test = get_layer_param(*ip)
    assert op_test == op, ("Mismatch!\n Expected: {}, Got:"
            " {}".format(op,op_test))

def test_network_init(temp_dir,loaded_dataset):
    '''
    Initializes Neural Network Object
    '''
    net = CNNBasic()

def test_network_forward_pass(loaded_dataset,network):
    item = loaded_dataset[1202]
    image = item['image']
    angles = item['angles']
    angles_predicted = network(image)

    print("Predicted = {}\nActual = {}".format(angles_predicted,angles))
