import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMG_DIR = os.path.join(ROOT_DIR, 'input_image')
OUTPUT_IMG_DIR = os.path.join(ROOT_DIR, 'output_image')
if not os.path.exists(OUTPUT_IMG_DIR):
    os.mkdir(OUTPUT_IMG_DIR)
