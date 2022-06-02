import os
import argparse
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pickle
import multiprocessing

parser = argparse.ArgumentParser(description='Preprocess MPI-INF-3DHP')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

def extract(seq_path, imgs_path, vid_i):

    # if doesn't exist
    if not os.path.isdir(imgs_path):
        os.makedirs(imgs_path)

    # video file
    vid_file = os.path.join(seq_path,
                            'imageSequence',
                            'video_' + str(vid_i) + '.avi')
    vidcap = cv2.VideoCapture(vid_file)

    # process video
    frame = 0
    while 1:
        # extract all frames
        success, image = vidcap.read()
        if not success:
            break
        frame += 1
        # image name
        imgname = os.path.join(imgs_path,
            'frame_%06d.jpg' % frame)
        # save image
        cv2.imwrite(imgname, image)

def train_data(dataset_path, out_path, joints_idx, scaleFactor, extract_img=False, fits_3d=None):

    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    h, w = 2048, 2048

    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))

    counter = 0

    data = []

    for user_i in tqdm(user_list):
        for seq_i in tqdm(seq_list):
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))

            imgs_paths = []
            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,
                                         'imageFrames',
                                         'video_' + str(vid_i))
                imgs_paths.append(imgs_path)
                
            pool_obj = multiprocessing.Pool()
            pool_obj.starmap(extract, zip([seq_path] * len(imgs_paths), imgs_paths, vid_list))

def mpi_inf_3dhp_extract(dataset_path, out_path, extract_img=True):

    scaleFactor = 1.0
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    train_data(dataset_path, out_path,
               joints_idx, scaleFactor,
               extract_img=extract_img)

if __name__ == '__main__':
    args = parser.parse_args()
    mpi_inf_3dhp_extract(args.dataset_path, args.out_path)
