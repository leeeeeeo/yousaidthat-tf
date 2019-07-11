import time
from datetime import datetime, timedelta
from time import sleep
import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str)
args = parser.parse_args()


SECONDS_PER_DAY = 24 * 60 * 60


def needTfrecords(dataset_folder):
    file_list = os.listdir(dataset_folder)
    for file in file_list:
        if file.endswith('tfrecords'):
            return False
    if str(time.strftime('%m%d', time.localtime())) in dataset_folder:
        return False
    return True


def doFunc():
    if args.device == 'server009':
        dataset_root = '/media/server009/data/dataset'
    elif args.device == 'ssh7':
        dataset_root = '/media/liuhan/liuhan_4T/dataset'
    elif args.device == 'ssh':
        dataset_root = '/workspace/liuhan/work/avasyn/data'
    dataset_folders = os.listdir(dataset_root)
    for dataset_folder in dataset_folders:
        foldername = dataset_folder
        if dataset_folder[:10] == 'lrs2_matla' or dataset_folder[:10] == 'lrs3_matla' or dataset_folder[:10] == 'lrw_matlab' or dataset_folder[:10] == 'vox2_matla':
            dataset_folder = os.path.join(dataset_root, dataset_folder)
            if needTfrecords(dataset_folder):
                print "{} START {}".format(
                    str(time.strftime('%m%d %H:%M:%S', time.localtime())), foldername)
                command = ("python build_data.py --device %s --func tfrecords --npz %s/%s --tfrecords %s/%s/%s_passnan.tfrecords --passnan 1" %
                           (args.device, dataset_root, foldername, dataset_root, foldername, foldername))
                print command
                output = subprocess.call(command, shell=True, stdout=None)
                print '{} END {}'.format(
                    str(time.strftime('%m%d %H:%M:%S', time.localtime())), foldername)


def doFirst():
    curTime = datetime.now()
    print 'cur time: ', curTime
    desTime = curTime.replace(hour=1, minute=0, second=0, microsecond=0)
    print 'des time: ', desTime
    delta = curTime - desTime
    skipSeconds = SECONDS_PER_DAY - delta.total_seconds()
    print 'wait for: ', skipSeconds, ' seconds'
    sleep(skipSeconds)
    doFunc()


if __name__ == "__main__":
    while 1:
        doFirst()
    # doFunc()
