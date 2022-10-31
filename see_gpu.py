import tensorflow as tf
import horovod.tensorflow as hvd
import time
import os

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

rank = hvd.rank()
local_rank = hvd.local_rank()

time.sleep(3*rank)

print(f"This thread is running as rank {rank} and local_rank {local_rank}")


# todo
# Build model...
loss = []
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())





gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# idk about this
print("This node can see the following GPUs:")
os.system("nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv")
ten_vis_gpu = tf.config.experimental.list_physical_devices('GPU')
print(f"Tensorflow on this thread can see {ten_vis_gpu}")
print("\n\n\n")



# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)


# For TensorFlow v2, when using a tf.GradientTape, wrap the tape in hvd.DistributedGradientTape instead of wrapping the optimizer.



# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = hvd.broadcast_variables
# hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None


# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
