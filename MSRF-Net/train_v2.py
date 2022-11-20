import os
import sys
import random
import pickle
import argparse
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
np.random.seed(123)
import itertools
import warnings
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate ,Lambda
import itertools
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from math import sqrt, ceil
from tqdm import tqdm_notebook as tqdm
import cv2
from sklearn.utils import shuffle
from tqdm import tqdm
import tifffile as tif
from model import msrf
from model import *
from tensorflow.keras.callbacks import *
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
import loss
import utils
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()
from glob import glob

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
#tf.compat.v1.enable_eager_execution()


def train(epochs, batch_size, dest_files):
    '''Trains the model'''
    batch_count = int(train_x.shape[0]/batch_size)
    val_bcount = int(val_x.shape[0]/batch_size)
    
    max_val_dice = -1
    G = msrf()
    # G.summary()
    
    # Get the optimise and set the evaluation matrix, Adam optimiser
    optimizer = loss.get_optimizer()
    G.compile(optimizer = optimizer,\
             loss = {'x':loss.seg_loss, 'edge_out':'binary_crossentropy', 'pred4':loss.seg_loss, \
                     'pred2':loss.seg_loss},\
             loss_weights={'x':2., 'edge_out':1., 'pred4':1, 'pred2':1.})
    
    history = {'Metrics':[], 'val_loss':[], 'dc':[], 'iou':[], 'prec':[], 'recall':[]}
    
    log_path = dest_files['output_dir']+dest_files['log_dir']
    hist_path = dest_files['output_dir']+dest_files['hist_dir']
    model_path = dest_files['output_dir']+dest_files['model_dir']
    
    with open(log_path,'w') as logfile:
    
        # Start training
        for e in range(1, epochs+1):
            print('-'*15, 'Epoch %d'% e,'-'*15)

            # Stores values
            train_metrics = np.zeros((batch_count, 5))

            # sp startpoint
            for sp in range(batch_count):
                batch_end = train_x.shape[0] if (sp+1)*batch_size > train_x.shape[0] else (sp+1)*batch_size
                # Getting the train batch image
                x_tot = [utils.get_image_v2(train_x[i]) for i in range((sp*batch_size), batch_end)]
                x_batch, edge_x_batch = [], []
                for i in range(len(x_tot)):
                    x_batch.append(x_tot[i][0])
                    edge_x_batch.append(x_tot[i][1])
                x_batch = np.array(x_batch).astype(np.float32)
                edge_x_batch = np.array(edge_x_batch).astype(np.float32)

                # Getting the train labels for images
                y_tot = [utils.get_image_v2(train_y[i], mask=True) for i in range((sp*batch_size), batch_end)]
                y_batch, edge_y_batch = [], []
                for i in range(len(y_tot)):
                    y_batch.append(y_tot[i][0])
                    edge_y_batch.append(y_tot[i][1])   
                y_batch = np.array(x_batch).astype(np.float32)
                edge_y_batch = np.array(edge_x_batch).astype(np.float32)

                # Train on the batch
                train_metrics[sp] = G.train_on_batch([x_batch, edge_x_batch],\
                                                     [y_batch, edge_y_batch, y_batch, y_batch])
               
            #loss1, loss2, _, _, _ = G.evaluate([val_x, edge_val_x], [y_val, y_val_edge, y_val, y_val])

            # Evaluating on Val dataset
            val_metrics = np.zeros((val_bcount, 5))
            for sp in range(val_bcount):
                batch_end = train_x.shape[0] if (sp+1)*batch_size > train_x.shape[0] else (sp+1)*batch_size
                val_x_batch, edge_x_batch = val_x[sp*batch_size: batch_end], edge_val_x[sp*batch_size: batch_end]
                y_batch, edge_y_batch = y_val[sp*batch_size: batch_end], y_val_edge[sp*batch_size: batch_end]
                
                val_metrics[sp] = G.test_on_batch([val_x_batch, edge_x_batch],[y_batch, edge_y_batch, y_batch, y_batch])
            
            
            # Making predictions after training over all batches
            train_metrics = train_metrics.mean(axis=0)
            val_metrics = val_metrics.mean(axis=0)
            
            # Predictions
            y_pred,_,_,_ = G.predict([val_x, edge_val_x], batch_size=5)
            y_pred = (y_pred >= 0.5).astype(int)
            y_pred = np.array(y_pred).astype(np.float32)
            res = loss.mean_dice_coef(y_val, y_pred)
            #res = loss.seg_loss(y_val, y_pred)
      
            iou, prec, recall = loss.compute_iou(y_val, y_pred) 
            print(f'train_loss: {train_metrics[1]:.4f} Val_loss: {val_metrics[1]:.4f} IOU: {iou:.4f} Precison: {prec:.4f} Recall: {recall:.4f}')
            log = 'Epoch {} :  Val_loss: {} IOU: {} Precision: {} Recall: {}\n'.format(e,train_metrics[1], val_metrics[1], iou, prec, recall)

            # Updating the dic
            history['Metrics'].append(train_metrics[1])
            history['val_loss'].append(val_metrics[1])
            history['dc'].append(res)
            history['iou'].append(iou)
            history['prec'].append(prec)
            history['recall'].append(recall)
            
            # Generating logs and writing the history
            logfile.write(log)
            with open(hist_path, 'wb') as hist:
                pickle.dump(history, hist, protocol=pickle.HIGHEST_PROTOCOL)
        
            # If new dice value is    
            if (res > max_val_dice):
                max_val_dice = res
                G.save(model_path)
                print(f'New Val_Dice High Score: {res:.4f}')
        
        
def set_gpu(GPU_NUM): 
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[GPU_NUM], 'GPU')
    

if __name__ == "__main__":
    # Set specific GPU for training
    #GPU_NUM = 1
    #set_gpu(GPU_NUM)
    path = '/global/cfs/projectdirs/cosmo/work/users/usf_cs690_2022_fall/data/simul'
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lens', dest='lens', action='store_true')
    parser.add_argument('-arcs', dest='arcs', action='store_true')
    
    args = parser.parse_args()
    
    if not args.lens and not args.arcs:
        print("Please choose either -lens or -arcs")
        sys.exit(0)
       
    elif args.arcs:
        print("Training for Arcs")
        train_x, train_y = np.load(f"{path}/train/images/imgs_128.npy"),\
                       np.load(f"{path}/train/arc_mask/arc_masks_128.npy")
    
        val_x, val_y = np.load(f"{path}/val/images/imgs_128.npy"),\
                   np.load(f"{path}/val/arc_mask/arc_masks_128.npy")
        
        dest_files = {'output_dir':'output/arcs/',\
                    'model_dir':'models/simul_arcs_ws2.h5',\
                    'log_dir':'logs/log_arcs_128_2.txt',\
                    'hist_dir':'logs/history_arcs_128.p'}  
                
        
    elif args.lens:
        print("Training for lens")
        train_x, train_y = np.load(f"{path}/train/images/imgs_128.npy"),\
                       np.load(f"{path}/train/lens_mask/lens_masks_128.npy")
        
        val_x, val_y = np.load(f"{path}/val/images/imgs_128.npy"),\
                   np.load(f"{path}/val/lens_mask/lens_masks_128.npy")
        
        dest_files = {'output_dir':'output/lens/',\
                    'model_dir':'models/simul_lens_ws2.h5',\
                    'log_dir':'logs/log_lens_128_2_ss_rmsprop_1e-4_p9_p9.txt',\
                     'hist_dir':'logs/history_lens_128_rmsprop_1e-4_p9_p9.p'}
                 
 
    print("Training shape : ",train_x.shape, train_y.shape)
    # Extracting the images and edges for validation set
    val_x_tot = [utils.get_image_v2(val_x[i]) for i in range(val_x.shape[0])]
    x_val, edge_x_val = [], []
    for i in range(len(val_x_tot)):
        x_val.append(val_x_tot[i][0])
        edge_x_val.append(val_x_tot[i][1])
        
    val_x = np.array(x_val).astype(np.float32)
    edge_val_x = np.array(edge_x_val).astype(np.float32)
    
    # Discarding edges for masks
    y_val_tot = [utils.get_image_v2(val_y[i], mask=True) for i in range(val_y.shape[0])]
    y_val = np.array([y_val_tot[i][0] for i in range(len(y_val_tot))]).astype(np.float32)
    y_val_edge = np.array([y_val_tot[i][1] for i in range(len(y_val_tot))]).astype(np.float32)
    
    print(f'Val Dataset X:{val_x.shape} Edges:{edge_val_x.shape} Y:{y_val.shape}')
    
    # Train the model
    BATCH_SIZE = 4
    train(15, BATCH_SIZE, dest_files)