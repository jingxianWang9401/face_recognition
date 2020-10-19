# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:18:37 2020

@author: wangjingxian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import shutil
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
import os
from shutil import copyfile

from pathlib import Path

import shutil
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
import os
from shutil import copyfile

from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
ap.add_argument("-u", "--username", type=str, default="",
                help="the name of user")
'''
ap.add_argument("-n", "--number", type=int, default=100,
                help="the number of one class in the dataset")
'''
args = vars(ap.parse_args())
# AVX Model path Restoring parameters from
# Restoring



def frame(url):
    frame_count = 0
    all_frames = []
    while(True):
        video_cap = cv2.VideoCapture(url)
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    return frame_count




def dataset_construction(url,username):
    size = 300
    number = frame(url)
    #number=100
    path1 = './data_faces'
    path2 = username
    faces_my_path = os.path.join(path1, path2)
    if not os.path.exists(faces_my_path):
        os.makedirs(faces_my_path)

    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(url)
    num = 1
    
    while True:
        if (num <= number):
            success, img = cap.read()
            if success is True:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                #status=status-1
                status=0
                break
            dets = detector(gray_img, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))
                cv2.imwrite(faces_my_path + '/' + str(num) + '.jpg', face)
                #print(faces_my_path + '/' + str(num) + '.jpg')
                num += 1
                status=0
        else:
            status=1
      
            break

            
    return status

#AVX Model path Restoring parameters from
#Restoring


image_dir='./data_faces'

output_graph='./cache_label/faces.pb'

intermediate_output_graphs_dir='./cache_label/intermediate_graph/'

intermediate_store_frequency=0

output_labels='./cache_label/faces.txt'

summaries_dir='./cache_label/retrain_logs'

how_many_training_steps=3000

learning_rate=0.01

testing_percentage=10

validation_percentage=10

eval_step_interval=100

train_batch_size=100

test_batch_size=-1

validation_batch_size=100

print_misclassified_test_images=False

model_dir='./models/imagenet'

bottleneck_dir='./cache_label/bottleneck'

final_tensor_name='final_result'

flip_left_right=False

random_crop=0

random_scale=0

random_brightness=0

architecture='inception_v3'

saved_model_dir='./cache_label/saved_models/1/'











MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = './cache_label/variable/_retrain_checkpoint'


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """
    从文件系统构建训练图像列表。

   分析图像目录中的子文件夹，将它们分成稳定的训练，测试和验证集，并返回描述每个标签及其路径的图像列表的数据结构。

  ARGS：
     image_dir：包含图像子文件夹的文件夹的字符串路径。
     testing_percentage：要为测试保留的图像的整数百分比。
     validation_percentage：为验证保留的图像的整数百分比。

  返回：
     包含每个标签子文件夹条目的字典，图像在每个标签内分为训练，测试和验证集。
  """
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    #tf.compat.v1.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # 这看起来有点神奇，但是我们需要确定这个文件
      # 进入训练，测试或验证集，并且我们希望
      # 将现有文件保留在同一个集合中，即使随后添加了更多文件。
      # 为此，我们需要一种基于文件名本身的稳定决策方式，
      # 因此我们对其进行哈希处理，
      # 然后使用它来生成我们用来分配它的概率值。
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  """"返回给定索引处标签的图像路径。

  ARGS：
     image_lists：每个标签的训练图像字典。
     label_name：我们想要获取图像的标签字符串。
     index：我们想要的图像的Int偏移量。 这将通过标签的可用图像数量来模数，因此它可以是任意大的。
     image_dir：包含训练图像的子文件夹的根文件夹字符串。
     category：从中提取图像的名称字符串 - 训练，测试或验证。

  返回：
     文件系统路径字符串到满足请求参数的图像。

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
  """"返回给定索引处标签的瓶颈文件的路径。

  ARGS：
     image_lists：每个标签的训练图像字典。
     label_name：我们想要获取图像的标签字符串。
     index：我们想要的图像的整数偏移量。 这将通过标签的可用图像数量来模数，因此它可以是任意大的。
     bottleneck_dir：保存瓶颈值缓存文件的文件夹字符串。
     category：从中提取图像的名称字符串 - 训练，测试或验证。
     architecture：模型体系结构的名称。

  返回：
     文件系统路径字符串到满足请求参数的图像。
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
  """"从保存的GraphDef文件创建图形并返回Graph对象。

  ARGS：
     model_info：包含有关模型体系结构的信息的字典。

  返回：
     图表持有训练有素的初始网络，以及我们将要操纵的各种张量。
  """
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(model_dir, model_info['model_file_name'])
    #print('Model path: ', model_path)
    with tf.io.gfile.GFile(model_path, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
  return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """对图像进行推理以提取“瓶颈”层特征向量。

  ARGS：
     sess：当前活动的TensorFlow会话。
     image_data：原始JPEG数据的字符串。
     image_data_tensor：图中的输入数据层。
     decoding_image_tensor：初始图像大小调整和预处理的输出。
     resized_input_tensor：识别图的输入节点。
     bottleneck_tensor：最终softmax之前的图层。

  返回：
     Numpy数组瓶颈值。
  """
  # 首先解码JPEG图像，调整其大小，然后重新缩放像素值。
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # 然后通过识别网络运行它。
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract(data_url):
  """下载并提取模型文件
  """
  dest_directory = model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded %s %d bytes.', filename,
                    statinfo.st_size)
    #print('Extracting file from ', filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  #else:
    #print('Not extracting or downloading files, model already present in disk')


def ensure_dir_exists(dir_name):
  """确保磁盘上存在该文件夹。

  ARGS：
     dir_name：我们要创建的文件夹的路径字符串。
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  #tf.logging.info('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = tf.io.gfile.GFile(image_path, 'rb').read()
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
  """检索或计算图像的瓶颈值。

如果磁盘上存在缓存版本的瓶颈数据，则返回该数据，否则计算数据并将其保存到磁盘以备将来使用。
ARGS：
    sess：当前活动的TensorFlow会话。
    image_lists：每个标签的训练图像字典。
    label_name：我们想要获取图像的标签字符串。
    index：我们想要的图像的整数偏移量。这将通过标签的可用图像数量进行模块化，因此可以任意大。
    image_dir：包含训练图像的子文件夹的根文件夹字符串。
    category：设置为从中提取图像的名称字符串 - 培训，测试或验证。
    bottleneck_dir：保存瓶颈值缓存文件的文件夹字符串。
    jpeg_data_tensor：将加载的jpeg数据输入的张量。
    decoding_image_tensor：解码和调整图像大小的输出。
    resized_input_tensor：识别图的输入节点。
    bottleneck_tensor：瓶颈值的输出张量。
    architecture：模型体系结构的名称。

  返回：
    由图像的瓶颈层产生的Numpy数组值。
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, architecture)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
  """确保缓存所有训练，测试和验证瓶颈层特征向量。

因为我们可能多次读取相同的图像（如果在训练期间没有应用失真），如果我们在预处理期间
为每个图像计算一次瓶颈层特征向量，那么它可以加快速度，然后只读取那些缓存的值在训练期间
反复利用，我们浏览我们找到的所有图像，计算这些值并将其保存。

  ARGS：
    sess：当前活动的TensorFlow会话。
    image_lists：每个标签的训练图像字典。
    image_dir：包含训练图像的子文件夹的根文件夹字符串。
    bottleneck_dir：保存瓶颈值缓存文件的文件夹字符串。
    jpeg_data_tensor：从文件输入jpeg数据的张量。
    decoding_image_tensor：解码和调整图像大小的输出。
    resized_input_tensor：识别图的输入节点。
    bottleneck_tensor：图的倒数第二个输出层，瓶颈层特征向量。
    architecture：模型体系结构的名称。

  返回：
    没有。
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)

        how_many_bottlenecks += 1
        '''
        if how_many_bottlenecks % 100 == 0:
          tf.compat.v1.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')
         '''


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
  """检索缓存图像的瓶颈值。

如果未应用任何图像增强技术，则此函数可以直接从磁盘检索缓存的瓶颈值以获取图像。它从指定的类别中选择一组随机图像。

  ARGS：
    sess：当前的TensorFlow会话。
    image_lists：每个标签的训练图像字典。
    how_many：如果是正数，将选择此大小的随机样本。如果是否定的，将检索所有瓶颈。
    category：要从中提取的名称字符串 - 训练，测试和验证数据。
    bottleneck_dir：保存瓶颈值缓存文件的文件夹字符串。
    image_dir：包含训练图像的子文件夹的根文件夹字符串。
    jpeg_data_tensor：将jpeg图像数据输入的图层。
    decoding_image_tensor：解码和调整图像大小的输出。
    resized_input_tensor：识别图的输入节点。
    bottleneck_tensor：CNN图的瓶颈输出层。
    architecture：模型体系结构的名称。

  返回：
    瓶颈数组列表，相应的基本事实以及相关的文件名。
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, architecture)
      bottlenecks.append(bottleneck)
      ground_truths.append(label_index)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)
        bottlenecks.append(bottleneck)
        ground_truths.append(label_index)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """在数据增强之后检索训练图像的瓶颈值。

  如果正在训练进行数据增强的图像，必须重新计算每个图像的完整模型，
  因此不能使用缓存的瓶颈值。相反，找到所请求类别的随机图像，通过失真图
  运行它们，然后是完整图形以获得每个图像的瓶颈结果。

  ARGS：
    sess：当前的TensorFlow会话。
    image_lists：每个标签的训练图像字典。
    how_many：要返回的整数瓶颈值。
    category：要获取的图像集的名称字符串 - 训练，测试或验证。
    image_dir：包含训练图像的子文件夹的根文件夹字符串。
    input_jpeg_tensor：我们将图像数据提供给的输入图层。
    distorted_image：失真图的输出节点。
    resized_input_tensor：识别图的输入节点。
    bottleneck_tensor：CNN图的瓶颈输出层。

  返回：
    瓶颈矩阵列表。
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: distorted_image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    bottlenecks.append(bottleneck_values)
    ground_truths.append(label_index)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """是否将输入图像进行相应的图像变换操作。

  ARGS：
     flip_left_right：Boolean是否水平随机翻转图像。
     random_crop：将输入图像进行随机剪裁操作。
     random_scale：改变输入图像比例的整数百分比。
     random_brightness：整数范围，用于随机乘以像素值，调整图像的亮度。

  返回：
     布尔值，指示是否应该应用任何数据变换技术。
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
  """创建应用指定数据增强的操作。

   在训练期间，如果我们通过简单的数据变换（如剪裁，缩放和翻转）来运行图像，它可以帮助
   改善结果。 这些反映了我们在现实世界中所期望的那种变化，因此可以帮助训练模型以
   更有效地处理自然数据。 在这里，我们获取提供的参数并构建操作网络以将它们应用于图像。

  裁剪
  ~~~~~~~~
通过将边界框放置在完整图像中的随机位置来完成裁剪。 cropping参数控制该框相对于输入
图像的大小。 如果它为零，则该框与输入的大小相同，并且不执行裁剪。 如果值为50％，
则裁剪框将为输入的宽度和高度的一半。 在图中它看起来像这样：

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  缩放很像裁剪，除了边界框始终居中并且其大小在给定范围内随机变化。例如，如果比例
  百分比为零，则边界框与输入的大小相同，并且不应用缩放。如果它是50％，
  那么边界框将在宽度和高度的一半与全尺寸之间的随机范围内。

  ARGS：
    flip_left_right：Boolean是否水平随机变换图像。
    random_crop：整数百分比设置裁剪框。
    random_scale：改变图像缩放比例的整数百分比。
    random_brightness：整数范围，用于随机乘以图形像素值，改变图像亮度变化。
    input_width：要建模的预期输入图像的水平大小。
    input_height：要建模的预期输入图像的垂直大小。
    input_depth：预期输入图像应具有的通道数。
    input_mean：图形图像中应为零的像素值。
    input_std：在识别之前将像素值除以多少。

  返回：
    jpeg输入层和图像增强后的结果张量。
  """

  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, input_width)
  precrop_height = tf.multiply(scale_value, input_height)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  offset_image = tf.subtract(brightened_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.compat.v1.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.compat.v1.summary.scalar('stddev', stddev)
    tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
    tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
    tf.compat.v1.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          bottleneck_tensor_size, quantize_layer, is_training):
  """添加一个新的softmax和全连接层用于训练和评估。
我们需要重新训练顶层以识别我们的新分类问题，因此该函数将向图中添加正确的操作，以及
一些用于保持权重的变量，然后设置向后传递的渐变。
  ARGS：
   class_count：试图识别的图像类别的整数。
     final_tensor_name：生成结果的最终新节点的名称。
     bottleneck_tensor：主CNN图的瓶颈层向量输出。
     bottleneck_tensor_size：瓶颈层特征向量的规模。
     quantize_layer：Boolean，指定是否应对新添加的层进行检测以进行量化。
     is_training：Boolean，指定新添加的图层是用于训练还是评估。

  返回：
     用于训练和交叉熵结果的张量，以及用于瓶颈输入和实际输入的张量。
  """
  with tf.name_scope('input'):
    bottleneck_input = tf.compat.v1.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.compat.v1.placeholder(
        tf.int64, [None], name='GroundTruthInput')

  # Organizing the following ops so they are easier to see in TensorBoard.
  layer_name = 'final_retrain_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.random.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      variable_summaries(layer_weights)

    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.compat.v1.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

  # The tf.contrib.quantize functions rewrite the graph in place for
  # quantization. The imported model graph has already been rewritten, so upon
  # calling these rewrites, only the newly added final layer will be
  # transformed.
  if quantize_layer:
    if is_training:
      tf.contrib.quantize.create_training_graph()
    else:
      tf.contrib.quantize.create_eval_graph()

  tf.compat.v1.summary.histogram('activations', final_tensor)

  # If this is an eval graph, we don't need to add loss ops or an optimizer.
  if not is_training:
    return None, None, bottleneck_input, ground_truth_input, final_tensor

  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """插入评估结果准确性所需的操作。

  ARGS：
     result_tensor：生成结果的新最终节点。
     ground_truth_tensor：实际数据输入的节点。

  返回：
     （评估步骤，预测）的元组。
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(prediction, ground_truth_tensor)
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.compat.v1.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def run_final_eval(sess, model_info, class_count, image_lists, jpeg_data_tensor,
                   decoded_image_tensor, resized_image_tensor,
                   bottleneck_tensor):
  """使用测试数据集在eval测试图像上运行最终评估。

  ARGS：
     sess：启动会话。
     model_info：来自create_model_info（）的模型信息字典
     class_count：类的数量
     image_lists：每个标签的训练图像字典。
     jpeg_data_tensor：jpeg图像数据输入的图层。
     decoding_image_tensor：解码和调整图像大小的输出。
     resized_image_tensor：识别图像的输入节点。
     bottleneck_tensor：CNN图的瓶颈输出层。
  """
  (sess, bottleneck_input, ground_truth_input, evaluation_step,
   prediction) = build_eval_session(model_info, class_count)

  test_bottlenecks, test_ground_truth, test_filenames = (
      get_random_cached_bottlenecks(sess, image_lists, test_batch_size,
                                    'testing', bottleneck_dir,
                                    image_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor,
                                    bottleneck_tensor, architecture))
  test_accuracy, predictions = sess.run(
      [evaluation_step, prediction],
      feed_dict={
          bottleneck_input: test_bottlenecks,
          ground_truth_input: test_ground_truth
      })
  #tf.compat.v1.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                  #(test_accuracy * 100, len(test_bottlenecks)))

  if print_misclassified_test_images:
    tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
    for i, test_filename in enumerate(test_filenames):
      if predictions[i] != test_ground_truth[i]:
        tf.logging.info('%70s  %s' % (test_filename,
                                      list(image_lists.keys())[predictions[i]]))


def build_eval_session(model_info, class_count):
  """构建已恢复的eval会话，不进行导出操作。

  ARGS：
     model_info：来自create_model_info（）的模型信息字典
     class_count：类的数量

  返回：
     包含已恢复的eval图的Eval会话。
     瓶颈输入，feed输入，评估步骤和预测张量。
  """
  # If quantized, we need to create the correct eval graph for exporting.
  eval_graph, bottleneck_tensor, _ = create_model_graph(model_info)

  eval_sess = tf.compat.v1.Session(graph=eval_graph)
  with eval_graph.as_default():
    # Add the new layer for exporting.
    (_, _, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, final_tensor_name, bottleneck_tensor,
         model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
         False)

    # Now we need to restore the values from the training graph to the eval
    # graph.
    tf.compat.v1.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

    evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                      ground_truth_input)

  return (eval_sess, bottleneck_input, ground_truth_input, evaluation_step,
          prediction)


def save_graph_to_file(graph, graph_file_name, model_info, class_count):
  """将图保存到文件，必要时创建有效的量化图形"""
  sess, _, _, _, _ = build_eval_session(model_info, class_count)
  graph = sess.graph

  output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [final_tensor_name])

  with tf.gfile.GFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())


def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.io.gfile.exists(summaries_dir):
    tf.io.gfile.rmtree(summaries_dir)
  tf.io.gfile.makedirs(summaries_dir)
  #if FLAGS.intermediate_store_frequency > 0:
  ensure_dir_exists(intermediate_output_graphs_dir)
  return
  
  '''
  if tf.io.gfile.exists(FLAGS.summaries_dir):
    tf.io.gfile.rmtree(FLAGS.summaries_dir)
  tf.io.gfile.makedirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return
'''
def create_model_info(architecture):
  """给定模型体系结构的名称，返回有关它的信息。

  有不同的基本图像识别预训练模型，使用迁移学习算法进行重新训练。

  ARGS：
     architecture：模型体系结构的名称。

  返回：
     有关模型的信息字典，如果名称未被识别，则为None
  """
  architecture = architecture.lower()
  is_quantized = False
  if architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128
  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.5' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25',
  but found '%s' for architecture '%s'""", version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quant':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True

    data_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
    model_name = 'mobilenet_v1_' + version_string + '_' + size_string
    if is_quantized:
      model_name += '_quant'
    data_url += model_name + '.tgz'
    bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
    resized_input_tensor_name = 'input:0'
    model_file_name = model_name + '_frozen.pb'

    bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
      'quantize_layer': is_quantized,
  }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """添加执行JPEG解码和调整大小的操作。

  ARGS：
     input_width：输入识别器图形的图像宽度。
     input_height：输入识别器图形的图像高度。
     input_depth：馈入识别器图形的图像所需通道。
     input_mean：图形图像中应为零的像素值。
     input_std：在识别之前将像素值除以多少。

  返回：
     节点的张量用于将JPEG数据输入到预处理步骤中，以及预处理步骤的输出。
  """
  jpeg_data = tf.compat.v1.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.compat.v1.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image


def export_model(model_info, class_count, saved_model_dir):
  """导出输出模式

  ARGS：
     model_info：当前模型的modelinfo。
     class_count：类的数量。
     saved_model_dir：保存导出的模型和变量的目录。
  """
  # The SavedModel should hold the eval graph.
  sess, _, _, _, _ = build_eval_session(model_info, class_count)
  graph = sess.graph
  with graph.as_default():
    input_tensor = model_info['resized_input_tensor_name']
    in_image = sess.graph.get_tensor_by_name(input_tensor)
    inputs = {'image': tf.compat.v1.saved_model.build_tensor_info(in_image)}

    out_classes = sess.graph.get_tensor_by_name('final_result:0')
    outputs = {
        'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
    }

    signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.PREDICT_METHOD_NAME)

    main_op = tf.group(tf.compat.v1.tables_initializer(), name='legacy_init_op')

    # Save out the SavedModel.
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.SERVING],
        signature_def_map={
            tf.saved_model.
            DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        },
        main_op=main_op)
    builder.save()



def main(_):
  
  
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  # Prepare necessary directories that can be used during training
  prepare_file_system()

  # Gather information about the model architecture we'll be using.
  model_info = create_model_info('inception_v3')
  if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    return -1

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(image_dir, testing_percentage,
                                   validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + image_dir)
    return -1
  if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     image_dir +
                     ' - multiple classes are needed for classification.')
    return -1
  #print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      flip_left_right, random_crop, random_scale,
      random_brightness)

  # Set up the pre-trained graph.
  maybe_download_and_extract(model_info['data_url'])
  graph, bottleneck_tensor, resized_image_tensor = (
      create_model_graph(model_info))

  # Add the new layer that we'll be training.
  with graph.as_default():
    (train_step, cross_entropy, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, final_tensor_name, bottleneck_tensor,
         model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
         True)

  with tf.compat.v1.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           flip_left_right, random_crop, random_scale,
           random_brightness, model_info['input_width'],
           model_info['input_height'], model_info['input_depth'],
           model_info['input_mean'], model_info['input_std'])
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, image_dir,
                        bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, architecture)
    #print('ffffffffffffffffffffffffffffff')
    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.compat.v1.summary.FileWriter(
        summaries_dir + '/validation')

    # Create a train saver that is used to restore values into an eval graph
    # when exporting models.
    train_saver = tf.compat.v1.train.Saver()

    # Set up all our weights to their initial default values.
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    #print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
    # Run the training for as many cycles as requested on the command line.
    for i in range(how_many_training_steps):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, train_batch_size, 'training',
             image_dir, distorted_jpeg_data_tensor,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, train_batch_size, 'training',
             bottleneck_dir, image_dir, jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
             architecture)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)
      
      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == how_many_training_steps)
      if (i % eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        
        #tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        #(datetime.now(), i, train_accuracy * 100))
        #tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        #(datetime.now(), i, cross_entropy_value))
        
        # TODO(suharshs): Make this use an eval graph, to avoid quantization
        # moving averages being updated by the validation set, though in
        # practice this makes a negligable difference.
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, validation_batch_size, 'validation',
                bottleneck_dir, image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                architecture))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        
        #tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        #(datetime.now(), i, validation_accuracy * 100,
                         #len(validation_bottlenecks)))
        
      
      # Store intermediate results
      intermediate_frequency = intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        # If we want to do an intermediate save, save a checkpoint of the train
        # graph, to restore into the eval graph.
        train_saver.save(sess, CHECKPOINT_NAME)
        intermediate_file_name = (intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        
        save_graph_to_file(graph, intermediate_file_name, model_info,
                           class_count)
    #print('ffffffffffffffffffffffffffffffffffff')
    # 训练结束后，强制进行最后一次模型文件的保存。
    train_saver.save(sess, CHECKPOINT_NAME)
    #print('ooooooooooooooooooooooooooooo')
    # 已完成所有培训，因此对我们之前未使用过的一些新图像进行最终测试评估。
    run_final_eval(sess, model_info, class_count, image_lists, jpeg_data_tensor,
                   decoded_image_tensor, resized_image_tensor,
                   bottleneck_tensor)
    #print('ppppppppppppppppppppppppppppppp')
    # 写出训练有素的图形和标签，将权重存储为常量。
    save_graph_to_file(graph, output_graph, model_info, class_count)
    #print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
    with tf.gfile.GFile(output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')
    #print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
    export_model(model_info, class_count, saved_model_dir)
    copyfile('./data_label/faces.pb', './data_label/backup/faces.pb')
    copyfile('./data_label/faces.txt', './data_label/backup/faces.txt')
    copyfile('./cache_label/faces.pb', './data_label/faces.pb')
    copyfile('./cache_label/faces.txt', './data_label/faces.txt')
    PATH1 = r'./cache_label/bottleneck'
    #print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    PATH3 = r'./cache_label/saved_models'

    shutil.rmtree(PATH1)
    
    shutil.rmtree(PATH3)
    #print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
    #shutil.rmtree(PATH2)
    #shutil.rmtree(PATH4)
    #shutil.rmtree(PATH5)
    

    #print('done')



#if __name__ == '__main__':
def dataset_train():
  try:
      #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
      tf.compat.v1.app.run(main=main)
      #print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
      status=1
  except Exception:        
        status=0
  return status
  
  
