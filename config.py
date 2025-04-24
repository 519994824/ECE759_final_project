# CHECKPOINT_DIR = "/Users/chenqinxinghao/Desktop/Code/Courses/Comp759/final_project/python_base/models/checkpoints/"
# SAVED_MODEL_DIR = "/Users/chenqinxinghao/Desktop/Code/Courses/Comp759/final_project/python_base/models/saved_model/"
# CACHE_DIR = "/Users/chenqinxinghao/Desktop/Code/Courses/Comp759/final_project/python_base/bert/"
import os

BASE_DIR = os.getcwd()

CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_model")
CACHE_DIR = os.path.join(BASE_DIR, "bert")
EPOCH = 5
BATCH_SIZE = 32