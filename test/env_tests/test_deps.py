'''
Tests Environment
'''
import pytest
import subprocess as sp
import os
import sys

'''
TODO: Modify to parameterized test when all dependencies
are fleshed out.
'''

def run_cmd(command_string):
    command_list = command_string.split(' ')
    op = sp.run(command_list,stderr=sp.PIPE,stdout=sp.PIPE)
    return op

def test_deps_pytest():
    op = run_cmd("python3 -m pytest --version")
    assert "pytest 6." in op.stderr.decode(), ("Is pytest installed?\n"
                                              "{}\n".format(op.stderr.decode()))

def test_deps_ffmpeg():
    op = run_cmd("ffmpeg -version")
    assert "ffmpeg version" in op.stdout.decode(), ("Is ffmpeg installed?\n"
                                                   "{}\n".format(op.stderr.decode()))


