import os
import tensorflow as tf
from tensorflow.python.client import device_lib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

if __name__ == "__main__":
    print(device_lib.list_local_devices())
    print(tf.test.is_gpu_available())

