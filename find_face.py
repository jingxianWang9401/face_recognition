# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import dlib
import os
import random
import argparse
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import time
import argparse

import numpy as np
import tensorflow as tf




faces_my_path = 'cache_face'
size = 300
if not os.path.exists(faces_my_path):
    os.makedirs(faces_my_path)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="1.mp4",
                help="path to input video file")
args = vars(ap.parse_args())


def eye_aspect_ratio(eye):
   
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[9])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[7])  # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    mar = (A + B) / (2.0 * C)

    return mar



def live_detection_eye(url):
    status=1
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0
    TOTAL = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = cv2.VideoCapture(url)
    time.sleep(1.0)
    while True:
        success, frame = vs.read()
        if not success: break
        try:
            frame = imutils.resize(frame, width=450)
        except:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

        if TOTAL > 1:
            return status
            break
    if TOTAL < 1:
        status=status-1
        return status



def live_detection_mouth(url):
    status=1
    MAR_THRESH = 0.5
    MOUTH_AR_CONSEC_FRAMES = 3
    COUNTER = 0
    TOTAL = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    vs = cv2.VideoCapture(url)
    time.sleep(2.0)
    while True:
        success, frame = vs.read()
        if not success: break
        try:
            frame = imutils.resize(frame, width=450)
        except:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            #mouthHull = cv2.convexHull(mouth)
            #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            if mar < MAR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

        if TOTAL > 1:
            return status
            break
    if TOTAL < 1:
        status=status-1
        return status
    
    
    
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        # graph_def.ParseFromString(f.read())
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    tf.compat.v1.disable_eager_execution()
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.io.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def get_faces(url):
    status=1
    file_name = url
    model_file = "./data_label/faces.pb"
    label_file = "./data_label/faces.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
      input_operation.outputs[0]: t
    })
        
    
    results = np.squeeze(results)

    top_k = results.argsort()[-1:][::-1]
    user_name = load_labels(label_file)
    for i in top_k:
        if results[i] > 0.8:
            return status, user_name[i]
        else:
            status=status-1
            return status
            
    # print(labels[i],results[i])
    