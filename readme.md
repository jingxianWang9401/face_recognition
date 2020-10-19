# 面部识别代码整理工作清单

[TOC]

**注意：所有代码应屏蔽不必要的中间输出，执行结果使用print()输出**

## 1. 克隆项目

**创建各自开发分支，不要直接提交到主分支**

工作子目录创建：

- data_faces: 存放截取的面部图片，用于后期标签运算使用
- data_label: 存放运算后的正式结果，pb以及txt文件（faces.pb;faces.txt)
  - backup: 存放上一次运算结果，pb以及txt文件
- cache_label: 存放运算临时结果，pb以及txt文件，运算完成后，将data_label中的pb和txt文件复制到backup文件夹内，再用新的pb和txt文件覆盖data_label中的文件
- models: 存放`shape_predictor_68_face_landmarks.dat,haarcascade_frontalface_default.xml`等文件，执行脚本需要这些内容的直接指定该目录，不再通过参数提交
- videos: 前端传过来的视频文件
- cache_face: 临时存放需要比对的面部图片

## 2. 活体检测代码

> shape-predictor文件从models文件夹中载入，眼睛或嘴巴任一条件判断成功即成功。

- 文件名：`live_detection.py`

- 输入参数：

  - videofile: 视频文件名

- 输出结果：

  - 成功：`======= 面部截取文件名 =======`

    > 确认为活体，截取面部图片，保存到cache_face文件夹。文件名使用`time.time()*1000000`微秒级时间戳命名

  - 失败：`======= failed =======`

## 3. 面部匹配代码

- 文件名：`get_faces.py`
- 输入参数：
  - facefile: 活体检测成功返回的人脸文件名

- 输出结果：
  - 成功：`======= 匹配到的用户名 =======`
  - 失败：`======= failed =======`

## 4. 面部学习代码

>  截取后直接进行学习，学习结果存放在cache_label文件夹中

- 文件名：`learn_faces.py`

- 输入参数：

  - videofile: 视频文件名

  - username: 用户名，用于创建data_face的子目录

  - howtolearn: 学习模式：1-dlib特征分析，2-xml特征分析（默认2）

    > 使用xml特征分析时，默认调用`models/face_default.xml`文件，不再传递文件名参数。需要使用不同的特征文件时，将源文件改名，新文件重新命名为`face_default.xml`即可

- 输出结果

  - 成功：`======= done =======`
  - 失败：`======= failed =======`

## 5. 交互服务（待定）

