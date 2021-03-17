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

# TODO: Currently assumes that pytest runs from project root
sys.path.append(os.path.join("util"))
from car_dataset import check_dir, check_file, CarDataset, DatasetError

# Helpers

def rand_string(length=5):
    return ''.join(random.choices(string.ascii_letters +
        string.digits, k=length))

# Fixtures
@pytest.fixture(scope="module")
def temp_dir():
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
        print("{}".format(os.listdir(tempdir)))

        yield tempdir

        # Go back to curr_dir before cleanup
        os.chdir(curr_dir)

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

def test_cardataset_init(temp_dir):
    folder_list = ['0_Frames','3_Frames']
    label_file_list = ['0.txt','3.txt']

    car_data = CarDataset(temp_dir,folder_list,label_file_list)
    print("Length = {}".format(len(car_data)))
