'''
Tests Dataset Functions
'''
import pytest
import random
import string
import tempfile
import subprocess as sp
import os
import sys

# TODO: Currently assumes that pytest runs from project root
sys.path.append(os.path.join("util"))
from car_dataset import check_dir, check_file, DatasetError

def rand_string(length=5):
    return ''.join(random.choices(string.ascii_letters +
        string.digits, k=length))

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

def test_cardataset_init():
    pass
