'''
Tests Dataset Functions
'''
import pytest
import random
import string
import tempfile
import shutil
import subprocess as sp
import os
import sys
import torch

from PIL import Image
from torchvision.transforms import ToTensor

# TODO: Currently assumes that pytest runs from project root
from util.car_dataset import check_dir, check_file, CarDataset, DatasetError

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

# Start Tests

@pytest.mark.parametrize(
        "test_dir,outcome",[
            (os.path.join("util"),True),
            (os.path.join("test"),True),
            (os.path.join(rand_string()),False)])
def test_check_dir(test_dir,outcome):
    try:
        check_dir(test_dir)
    except DatasetError as e:
        assert not outcome, "Expected {} but not found!".format(test_dir)
        assert "Does not exist" in e.message, "Error Message Seems Off!"

@pytest.mark.parametrize(
        "test_file,outcome",[
            (os.path.join("README.md"),True),
            (os.path.join("util","car_dataset.py"),True),
            (os.path.join(rand_string()),False)])
def test_check_file(test_file,outcome):
    try:
        check_file(test_file)
    except DatasetError as e:
        assert not outcome, "Expected {} but not found!".format(test_file)
        assert "Does not exist" in e.message, "Error Message Seems Off!"

@pytest.mark.parametrize(
        "folder_list,label_file_list,outcome",[
        (['0_Frames','3_Frames'],['0.txt','3.txt'],True),
        (['foo','bar'],['0.txt','3.txt'],False),
        (['0_Frames','3_Frames'],['foo','bar'],False),
        ])
def test_cardataset_init(temp_dir,folder_list,label_file_list,outcome):
    try:
        car_data = CarDataset(temp_dir,folder_list,label_file_list)
    except DatasetError as e:
        assert not outcome, "Got Error {}".format(e.message)

def test_cardataset_len(loaded_dataset):
    car_dataset_length = len(loaded_dataset)
    assert car_dataset_length == 2400, "Expected 2400; Got {}".format(
            car_dataset_length)

@pytest.mark.parametrize(
        "index,outcome",[
        (0,True),
        (1199,True),
        (1200,True),
        (1201,True),
        (2399,True),
        (2400,False),
        (-1,False)])
def test_cardataset_get_item_index(loaded_dataset,index,outcome):
    try:
        item = loaded_dataset[index]
    except DatasetError as e:
        assert not outcome, "Got Error {}".format(e.message)

@pytest.mark.parametrize(
        "index,outcome",[
        (0,[3.346066188150387949e-02, 3.149205029088487234e-02]),
        (1199,[3.376331391075349658e-02, 3.256499232905601254e-02]),
        (1200,[2.162449375247458075e-02, 2.197126520637055977e-02]),
        (1201,[2.167494525873153027e-02, 2.196024970631171858e-02]),
        (2399,[2.761138162404873017e-02, 2.093151905506832750e-02])])
def test_cardataset_get_item_value_angles(loaded_dataset,index,outcome):
    item = loaded_dataset[index]
    angles = item['angles']

    pitch_retr = round(angles[0].item(),4)
    pitch_expected = round(outcome[0],4)
    assert pitch_retr == pitch_expected, ("Pitch Mismatch, Expected "
        "{}, Got {}".format(pitch_expected,pitch_retr))

@pytest.mark.parametrize(
        "index,outcome",[
        (0,os.path.join("0_Frames","0_0001.jpg")),
        (1199,os.path.join("0_Frames","0_1200.jpg")),
        (1200,os.path.join("3_Frames","3_0001.jpg")),
        (1201,os.path.join("3_Frames","3_0002.jpg")),
        (2399,os.path.join("3_Frames","3_1200.jpg"))])
def test_cardataset_get_item_value_images(temp_dir,loaded_dataset,index,outcome):
    item = loaded_dataset[index]
    image_retr = item['image']
    image_expected = Image.open(os.path.join(temp_dir,outcome))
    image_expected = ToTensor()(image_expected).unsqueeze(0)

    assert torch.all(torch.eq(image_retr,image_expected)), ("Images Are Not the same!!")

def test_nan_removal(loaded_dataset):
    cur_length = len(loaded_dataset)
    loaded_dataset.drop_nan()
    new_length = len(loaded_dataset)
    assert new_length == 2054, ("Original Length = {}\nNew Length = "
        "{}\nExpected New Length = 2054".format(cur_length,new_length))

@pytest.mark.parametrize(
        "index,outcome",[
        (0,True),
        (1199,True),
        (1200,True),
        (1201,True),
        (2053,True),
        (2054,False),
        (2399,False),
        (2400,False),
        (-1,False)])
def test_after_nan_cardataset_get_item_index(loaded_dataset,index,outcome):
    try:
        item = loaded_dataset[index]
    except DatasetError as e:
        assert not outcome, "Got Error {}".format(e.message)

@pytest.mark.parametrize(
        "index,outcome",[
        (0,[3.346066188150387949e-02, 3.149205029088487234e-02]),
        (1199,[3.376331391075349658e-02, 3.256499232905601254e-02]),
        (1200,[2.162449375247458075e-02, 2.197126520637055977e-02]),
        (1201,[2.167494525873153027e-02, 2.196024970631171858e-02]),
        (2053,[2.761138162404873017e-02, 2.093151905506832750e-02])])
def test_after_nan_cardataset_get_item_value_angles(loaded_dataset,index,outcome):
    item = loaded_dataset[index]
    angles = item['angles']

    pitch_retr = round(angles[0].item(),4)
    pitch_expected = round(outcome[0],4)
    assert pitch_retr == pitch_expected, ("Pitch Mismatch, Expected "
        "{}, Got {}".format(pitch_expected,pitch_retr))

@pytest.mark.parametrize(
        "index,outcome",[
        (0,os.path.join("0_Frames","0_0001.jpg")),
        (1199,os.path.join("0_Frames","0_1200.jpg")),
        (1200,os.path.join("3_Frames","3_0001.jpg")),
        (1201,os.path.join("3_Frames","3_0002.jpg")),
        (2053,os.path.join("3_Frames","3_1200.jpg"))])
def test_after_nan_cardataset_get_item_value_images(temp_dir,loaded_dataset,index,outcome):
    item = loaded_dataset[index]
    image_retr = item['image']
    print(os.listdir())
    image_expected = Image.open(os.path.join(temp_dir,outcome))
    image_expected = ToTensor()(image_expected).unsqueeze(0)

    assert torch.all(torch.eq(image_retr,image_expected)), ("Images Are Not the same!!")
