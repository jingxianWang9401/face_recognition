# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:18:08 2020

@author: wangjingxian
"""

#python cap_one_image.py --video G:\python\face-recognization\videos\eye_wuyuge.mp4


import cv2
import dlib
import os
import random
import argparse
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
from pathlib import Path

faces_my_path = '../cache/videos'
size = 300
if not os.path.exists(faces_my_path):
    os.makedirs(faces_my_path)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="1.mp4",
                help="path to input video file")
args = vars(ap.parse_args())


def cache_face(video):
    faces_my_path = '../cache/videos'
    size = 300
    if not os.path.exists(faces_my_path):
        os.makedirs(faces_my_path)

    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(args["video"])
    num = 1
    while True:
        if (num <= 1):
            success, img = cap.read()
            if success is True:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                break
            dets = detector(gray_img, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))
                a=int(time.time() * 1000000)
                cv2.imwrite(faces_my_path + '/' + str(a) + '.jpg', face)
                print("======= "+faces_my_path + '/' + str(a) + '.jpg =======')
                num += 1

        else:

            break

def cap_one_image(url):
    size = 300
    faces_my_path = './cache/videos'
    if not os.path.exists(faces_my_path):
        os.makedirs(faces_my_path)

    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(url)
    num = 1
    status=1
    image = ""
    while True:
        if (num <= 1):
            success, img = cap.read()
            if success is True:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            else:
                status=status-1
                break
            dets = detector(gray_img, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))
                a = int(time.time() * 1000000)
                cv2.imwrite(faces_my_path + '/' + str(a) + '.jpg', face)
                # print("======= " + faces_my_path + '/' + str(a) + '.jpg =======')
                #image = image + "======= " + faces_my_path + '/' + str(a) + '.jpg ======='
                image=faces_my_path + '/' + str(a) + '.jpg'
                num += 1

        else:
            break

    return status,image

if __name__ == "__main__":
    video = args["video"]
    cache_face(video)

