import horovod.tensorflow as hvd
import tensorflow as tf
import time
import os

hvd.init()

rank = hvd.rank()

time.sleep(3*rank)

print(f"This thread is running as rank {rank}")

print("This node can see the following GPUs:")

os.system("nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv")

ten_vis_gpu = tf.config.experimental.list_physical_devices('GPU')

print(f"Tensorflow on this thread can see {ten_vis_gpu}")

print("\n\n\n")
