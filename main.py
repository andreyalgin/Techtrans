# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import warnings
import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.gfile = tf.io.gfile
import cv2 as cv
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

warnings.filterwarnings('ignore')

def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []

    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                             origin=base_url + filename,
                                             untar=False)
        print(image_path)
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))

    return image_paths

def load_image_into_numpy_array(path):

    return np.array(Image.open(path))

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

def download_labels(file_name):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=file_name,
                                        origin=base_url + file_name,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)


print(tf.version.VERSION)

IMAGE_PATHS = download_images()


# http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz

MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_hg104_1024x1024_kpts_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

# https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()

elapsed_time = end_time - start_time

print('Done ! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

"""
for image_path in IMAGE_PATHS:
    print('Running inference for {}...' .format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')

plt.show()
"""


"""

image_np = load_image_into_numpy_array(IMAGE_PATHS[0])

plt.figure()
plt.imshow(image_np)

# plt.plot([1, 2, 3])

plt.show()

"""

"""
img = cv.imread("/home/andrey/.keras/datasets/image1.jpg")

cv.imshow("Image", img)
cv.waitKey(0)
"""

def detectObjects(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        print(area)
        if area > 25:
            cv.drawContours(imgContour, cnt, -1, (255, 0, 0), 1)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv.boundingRect(approx)

            if objCor == 3: objectType = "Tri"
            else: objectType = "None"

            cv.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.putText(imgContour, objectType, (x + (w//2) - 10, y + (h//2) - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

cap = cv.VideoCapture("The_Matrix_Trim.mp4")
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (720, 304))

while True:
    success, img = cap.read()

    imgContour = img.copy()
    # imgCanny = cv.Canny(img, 50, 50)

    # getContours(imgCanny)

    # cv.imshow("Video", imgContour)

    detectObjects(img)

    out.write(img)
    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print('Done')

cap.release()
out.release()

cv.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
