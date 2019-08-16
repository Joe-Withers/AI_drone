import tensorflow as tf
from tensorflow.python.client import device_lib
from distutils.version import StrictVersion
import os
import glob
import numpy as np
import time
#from object detection module
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
#temp imports
#import matplotlib.pyplot as plt
import cv2
from io import StringIO
from PIL import Image

def _load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


class Object_Detector():
    """ A simple class for using the tensorflow object detection API """
    #constructor
    def __init__(self, frozen_graph_path, labels_path, debug_mode = False):

        self.DEBUG_MODE = debug_mode
        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

        # What model to use
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = frozen_graph_path
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = labels_path

        #load a frozen tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.config = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False)
        self.sess = tf.Session(config = self.config, graph = self.detection_graph)

    #object description
    def __str__(self):
        return '<class/object description here>'

    def detect_objects(self, img):
        """ Input image that is a numpy array with shape [num_of_images, image_width,
        image_height, channels]. This returns a list bounding boxes in the format
        {'detection_boxes':<detection boxes>, 'detection_classes':<detection classes>,
        'detection_scores':<detection scores>}. """
        image = np.copy(np.array(img))
        # Actual detection.
        # start_t_inference = time.time()
        output_dict = self._run_inference_for_single_image(image)
        # t_for_inference = time.time() - start_t_inference
        # print('time for inference',t_for_inference)
        disp_im = image
        if self.DEBUG_MODE:
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                disp_im,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
        return output_dict, disp_im

    def _run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            # with graph.as_default():
            # with tf.Session(config = tf.ConfigProto(device_count = {'GPU': 0})) as sess:
            # with self.sess as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = self.sess.run(tensor_dict,
                feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == "__main__":
    # model_name = './cones_graph'
    # #SSD_mobilenet(20257)    dukecone    faster_RCNN_resnet101(330)     faster_RCNN_resnet50(3374)
    # frozen_graph_name = '/SSD_mobilenet(20257)/frozen_inference_graph.pb'
    # labels = os.path.join('./data', 'object-detection.pbtxt')
    obj_detector = Object_Detector(debug_mode = True)

    PATH_TO_TEST_IMAGES_DIR = './images'
    TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/*.jpg')

    max_itr = 100
    i=0
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = _load_image_into_numpy_array(image)
        output_dicts, category_index, img = obj_detector.detect_objects(image_np)
        cv2.imshow('image '+str(i),cv2.resize(img,(800,600)))
        cv2.waitKey(0)
        i+=1
        if i>=max_itr:
            break
    get_available_gpus()
