'''
Script to Convert an HEVC Video into image frames.
'''

import sys
import os
import logging
import datetime
import subprocess as sp
import concurrent.futures

def run_cmd(command_string):
    command_list = command_string.split()
    op = sp.run(command_list,stderr=sp.PIPE,stdout=sp.PIPE)
    return op

def gen_ffmpeg_command(ip_file_path,fps,op_dir=None):
    file_name = os.path.split(ip_file_path)[1].split('.')[0]
    if op_dir == None:
        op_dir_name = "{}_Frames".format(file_name)
    else:
        op_dir_name = "{}".format(op_dir)

    op_file_pattern = "{}_%03d.jpg".format(file_name)
    op_dir = os.path.join(os.path.split(ip_file_path)[0],op_dir_name,op_file_pattern)
    ffmpeg_command = "ffmpeg -r {} -i {} -qscale:v 2 {}".format(fps,ip_file_path
                            ,op_dir)
    return ffmpeg_command

def run_ffmpeg(command_string):
    logging.info("Attempting to Run '{}'".format(command_string))

    if 'ffmpeg' not in command_string:
        logging.error("Command '{}'' Doesn't seem to be related to "
                "ffmpeg!".format(command_string))
        rt_string = "non-ffmpeg command specified!"
    else:
        op = run_cmd(command_string)
        if not op.returncode:
            rt_string = "ffmpeg Execution Successful!"

        else:
            rt_string = "ffmpeg Execution Failed!"

        logging.debug("Outputs\nstdout : {}\nstderr : {}".format(op.stdout.decode(),
                    op.stderr.decode()))
    return rt_string

def get_time_string(dt_object=None):
    if dt_object == None:
        dt = datetime.datetime.now()
    else:
        dt = dt_object
    time_string = "{}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(dt.year,dt.month,dt.day
            ,dt.hour,dt.minute,dt.second)
    return time_string

def create_op_dir(ip_file_path,op_dir):
    ip_file_path_list = os.path.split(ip_file_path)
    logging.debug("Attempting to Create OP Directory for"
            " {}".format(ip_file_path_list[1]))
    if op_dir == None: # Similar logic as gen_ffmpeg_command()
        op_dir_name = "{}_Frames".format(ip_file_path_list[1].split('.')[0])
    else:
        op_dir_name = op_dir

    op_dir_path = os.path.join(ip_file_path_list[0],op_dir_name)
    if os.path.isdir(op_dir_path):
        logging.debug("'{}' exists!".format(op_dir_path))

    else:
        os.mkdir(op_dir_path)
        logging.debug("Created '{}'".format(op_dir_path))

def main(args):
    '''
    Only Argument will be the directory containing
    labelled videos.
    '''

    logging.basicConfig(filename="hevc_split_{}.log".format(get_time_string())
            ,filemode="w+",level=logging.DEBUG)

    if len(args)<1:
        logging.error("Insufficient Arguments! Requires path to root"
                      "containing HEVC files.")
        sys.exit(1)

    # Move into Root Dir
    os.chdir(args[0])
    root_dir = os.getcwd()

    # Map Directory and Find HEVC Files
    hevc_manifest = set()
    logging.debug("Looping through items in {}".format(root_dir))
    for item in os.listdir():
        if ('hevc' in item) or ('HEVC' in item):
            ip_file_path = os.path.join(root_dir,item)
            fps = 20 # TODO: Add as argument
            op_dir = None
            ffmpeg_cmd = gen_ffmpeg_command(ip_file_path,fps,op_dir)
            hevc_manifest.add(ffmpeg_cmd)
            logging.info("'{}' added to manifest!".format(ffmpeg_cmd))

            # Create op_dir
            create_op_dir(ip_file_path,op_dir)

    if len(hevc_manifest) < 1:
        logging.error("Empty Manifest! No HEVC files in"
                " {}".format(rrot_dir))
        sys.exit(1)

    logging.debug("Manifest Generated! Concurrent Parallel Execution!")

    # Concurrent Execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for rt_string in executor.map(run_ffmpeg,hevc_manifest):
            logging.debug(rt_string)

    logging.debug("Program Execution Complete!")
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
