'''
Tests HEVC Functions
'''
import pytest
import datetime
import tempfile
import subprocess as sp
import os
import sys

# TODO: Currently assumes that pytest runs from project root
from util.hevc import gen_ffmpeg_command,run_ffmpeg,get_time_string,create_op_dir

@pytest.mark.parametrize(
        "ip_file_path,fps,op_dir,outcome",
        [(os.path.join("abc.hevc"),15,"test","ffmpeg -r 15 -i"
                " abc.hevc -qscale:v 2 {}".format(os.path.join("test",
                    "abc_%04d.jpg"))),
         (os.path.join("abc.hevc"),27,None,"ffmpeg -r 27 -i"
                " abc.hevc -qscale:v 2 {}".format(os.path.join("abc_Frames"
                    ,"abc_%04d.jpg"))),
         (os.path.join(".","0.hevc"),20,"pla","ffmpeg -r 20 -i"
                " {} -qscale:v 2 {}".format(os.path.join(".",
                    "0.hevc")
                    ,os.path.join(".","pla","0_%04d.jpg"))),
         (os.path.join("home","0.hevc"),10,"0_Frame","ffmpeg -r 10 -i"
                " {} -qscale:v 2 {}".format(os.path.join("home",
                    "0.hevc"),os.path.join("home","0_Frame","0_%04d.jpg")))
         ])
def test_gen_ffmpeg_command(ip_file_path,fps,op_dir,outcome):
    f_com = gen_ffmpeg_command(ip_file_path,fps,op_dir)
    assert f_com == outcome, "Mismatch\nGenerated '{}'\nExpected '{}'".format(
            f_com,outcome)

@pytest.mark.parametrize(
        "command,outcome",
        [("ffmpeg -version","ffmpeg Execution Successful!"),
         ("foo -version","non-ffmpeg command specified!"),
         ("ffmpeg -blah","ffmpeg Execution Failed!"),
        ])
def test_run_ffmpeg(command,outcome):
    result = run_ffmpeg(command)
    assert result == outcome,("Mismatch\nGenerated '{}'\nExpected"
                              " '{}'".format(result,outcome))

@pytest.mark.parametrize(
        "dt_object,outcome",
        [(datetime.datetime(1984,12,3),"1984_12_03_00_00_00"),
         (datetime.datetime(1999,1,31,23,5,0),"1999_01_31_23_05_00"),
         (None,"Compare With Curr Time")
        ])
def test_get_time_string(dt_object,outcome):
    rt = get_time_string(dt_object)

    if dt_object == None:
        oc = datetime.datetime.now()
        logic = ("{}".format(oc.year) in rt) and \
                ("{:02d}".format(oc.month) in rt) and \
                ("{:02d}".format(oc.day) in rt) and \
                ("{:02d}".format(oc.hour) in rt) and \
                ("{:02d}".format(oc.minute) in rt)
        assert logic, ("Error\nSent {}\nGot {}".format(oc,rt))

    else:
        assert rt == outcome, ("Error\nSent {}\nGot {}".format(dt_object,rt))

@pytest.mark.parametrize(
        "op_dir",[
        (None),
        ("dummy"),
        ("foo")])
def test_create_op_dir(op_dir):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_file_path = os.path.join(tmp_dir,'blah.hevc')

        if op_dir == None:
            targ_dir_name = "{}_Frames".format('blah')
        else:
            targ_dir_name = op_dir
        targ_dir_path = os.path.join(tmp_dir,targ_dir_name)

        assert not os.path.isdir(targ_dir_path), "Directory Already Exists!"
        create_op_dir(dummy_file_path,op_dir)
        assert os.path.isdir(targ_dir_path), "Directory Creation Failed!"
