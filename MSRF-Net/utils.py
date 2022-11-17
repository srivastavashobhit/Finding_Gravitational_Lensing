import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import os
from glob import glob
np.random.seed(123)
import warnings
import numpy as np
from sklearn.utils import shuffle
import cv2


def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    return image, mask

def read_params():
    """ Reading the parameters from the JSON file."""
    with open("params.json", "r") as f:
        data = f.read()
        params = json.loads(data)
        return params

def load_data(path):
    """ Loading the data from the given path. """
    images_path = os.path.join(path, "image/*")
    masks_path  = os.path.join(path, "mask/*")
    
    images = glob(images_path)
    masks  = glob(masks_path)
    
    return images, masks

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def get_image(image_path, image_size_wight, image_size_height,gray=False):
    # load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    if gray==True:
        img = img.convert('L')
    # center crop
    img_center_crop = img
    # resize
    img_resized = img_center_crop.resize((image_size_height, image_size_wight), Image.ANTIALIAS)
    edge = cv2.Canny(np.asarray(np.uint8(img_resized)),10,1000)
    
    flag = False
    # convert to numpy and normalize
    img_array = np.asarray(img_resized).astype(np.float32)/255.0
    edge = np.asarray(edge).astype(np.float32)/255.0
    #print(img_array)
    if gray==True:
        img_array=(img_array >=0.5).astype(int)
    img.close()
    return img_array,edge

def get_image_v2(image_arr, mask=False):
    '''All images are of size 256 x 256'''
    # Scaling the image by a factor of 10
    if not mask:
        image_arr[image_arr > np.percentile(image_arr, 95)] *= 10
        image_arr[image_arr > np.percentile(image_arr, 99)] /= 10
        image_arr[image_arr > 255] = 255
    
    # Get the edge
    edge = cv2.Canny(np.uint8(image_arr), 10, 1000)
    
    # Rescale image
    img_arr = np.asarray(image_arr).astype(np.float32)/255.0
    edge = np.asarray(edge).astype(np.float32)/255.0
    
    if mask:
        img_arr = (img_arr >= 0.5).astype(int)
    
    # adding 1 dimension at the end and converting to float32
    img_arr = np.expand_dims(img_arr, axis=-1)
    edge = np.expand_dims(edge, axis=-1)
    
    return img_arr, edge