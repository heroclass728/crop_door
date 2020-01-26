import numpy as np
import os
import sys
import tensorflow as tf

from object_detection.detect_utils import visualization_utils as vis_util
from object_detection.detect_utils import label_map_util


_cur_dir = os.path.dirname(os.path.abspath(__file__))


class ObjectDetector:

    def __init__(self, threshold=.9):
        label_map = label_map_util.load_labelmap(os.path.join(_cur_dir, 'model', 'label_map.pbtxt'))
        self.categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize,
                                                                         use_display_name=True)
        self.CATEGORY_INDEX = label_map_util.create_category_index(self.categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(os.path.join(_cur_dir, 'model', 'frozen_inference_graph.pb'), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.threshold = threshold

    def detect_object(self, frame):
        im_width = frame.shape[1]
        im_height = frame.shape[0]
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_expanded = np.expand_dims(frame, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.CATEGORY_INDEX,
                    min_score_thresh=self.threshold,
                    use_normalized_coordinates=True,
                    line_thickness=3)

                object_coordinate = []
                object_description = []
                for box, obj_class in zip(boxes[0], classes[0]):
                    ymin, xmin, ymax, xmax = box
                    if ymin != 0 and xmin != 0 and ymax != 0 and ymin != 0:
                        object_coordinate.append((int(xmin * im_width), int(ymin * im_height), int(xmax * im_width),
                                                 int(ymax * im_height)))
                        object_description.append(obj_class)

        return object_coordinate


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    if image.getdata().mode != "RGB":
        image = image.convert('RGB')
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
