import glob
import os
import cv2

from settings import INPUT_IMG_DIR, OUTPUT_IMG_DIR
from object_detection.object_detection_runner import ObjectDetector

obj_detector = ObjectDetector()


def crop_door_from_image():

    frames_path = glob.glob(os.path.join(INPUT_IMG_DIR, "*.*"))
    index = 0

    for frame_path in frames_path:

        frame = cv2.imread(frame_path)
        coordinates, description = obj_detector.detect_object(frame.copy())
        for des, coordinate in zip(description, coordinates):
            if des == 2:
                file_name = "door" + str(index) + ".png"
                file_path = os.path.join(OUTPUT_IMG_DIR, file_name)
                door_img = frame[coordinate[1]-2:coordinate[3]+2, coordinate[0]-2:coordinate[2]+2]
                cv2.imwrite(file_path, door_img)
                index += 1
        print(frame_path)