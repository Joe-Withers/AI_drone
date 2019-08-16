import tensorflow as tf
from distutils.version import StrictVersion
import sys
import os
import glob
import numpy as np
import time
import datetime
from pydnet.utils import *
from pydnet.pydnet import *
from PIL import Image

def _load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

class Depth_Detector():
    """ A simple class for using the tensorflow object detection API """
    #constructor
    def __init__(self, debug_mode = False):
        checkpoint_dir = 'pydnet/checkpoint/IROS18/pydnet'
        self.DEBUG_MODE = debug_mode
        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

            with tf.variable_scope("model") as scope:
                self.model = pydnet(self.placeholders)

            self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

            loader = tf.train.Saver()

        config = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False)
        self.sess = tf.Session(config = config, graph = self.detection_graph)
        self.sess.run(self.init)
        loader.restore(self.sess, checkpoint_dir)

        self.resolution = 1

    #object description
    def __str__(self):
        return '<class/object description here>'

    def detect_depth(self, image):
        with self.detection_graph.as_default():

            image = cv2.resize(image, (512, 256)).astype(np.float32) / 255.
            image = np.expand_dims(image, 0)
            start = time.time()
            disp = self.sess.run(self.model.results[self.resolution-1], feed_dict={self.placeholders['im0']: image})
            end = time.time()

            depth_map = disp[0,:,:,0]
            disp_color = applyColorMap(depth_map*20, 'plasma')
            disp_img = (disp_color*255.).astype(np.uint8)
            disp_img = cv2.resize(disp_img, (512, 256))

        return depth_map, disp_img

if __name__ == '__main__':
    depth_detector = Depth_Detector()
    PATH_TO_TEST_IMAGES_DIR = './images'
    TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/*.jpg')

    max_itr = 100
    i=0
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = _load_image_into_numpy_array(image)
        disp_color = depth_detector.detect_depth(image_np)
        cv2.imshow('image '+str(i), disp_color)
        cv2.waitKey(0)
        i+=1
        if i>=max_itr:
            break
