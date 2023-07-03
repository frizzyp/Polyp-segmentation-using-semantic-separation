import os
import numpy as np
import tensorflow as tf
from glob import glob
import cv2
from sklearn.model_selection import train_test_split

def load_data(path, split=0.1):
    images=sorted(glob(os.path.join(path, 'images/*')))
    masks=sorted(glob(os.path.join(path, 'masks/*')))

    total_size=len(images)
    tt_size=int(total_size*split)
    validation_size=int(total_size*split)

    train_x, validation_x= train_test_split(images, test_size=validation_size, random_state=42)

    train_y, validation_y= train_test_split(masks, test_size=validation_size, random_state=42)

    train_x, test_x= train_test_split(images, test_size=tt_size, random_state=42)

    train_y, test_y= train_test_split(masks, test_size=tt_size, random_state=42)

    return((train_x, train_y), (validation_x, validation_y), (test_x, test_y))


def read_images(path):
    print("the path is",path)
    x=cv2.imread(path, cv2.IMREAD_COLOR)
    x=cv2.resize(x, (256, 256))
    #scaling
    x/=255.0
    return x

def read_mask(path):
    x=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x=cv2.resize(x, (256, 256))
    #since images read are in 3 dimensions, we need to add a dimension to its last axis 
    x=np.expand_dims(x, axis=-1)
    x/=255
    return x

def tf_dataset(x,y, batch=8):
    dataset=tf.data.Dataset.from_tensor_slices([x,y])
    dataset=dataset.map(tf_parse(x,y))
    dataset=dataset.batch(batch)
    dataset=dataset.repeat()
    return dataset

#this takes in a single image and a single mask path
def tf_parse(x,y):
    def _parse(x,y):
        x=read_images(x)
        y=read_mask(y)
        return x,y

    #the images read are in the form of numpy arrays, to convert it to tensors

    x,y=tf.numpy_function(_parse, [x,y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])

    return x,y



if __name__=="__main__":
    path='/home/frizzyzy/Desktop/prgs/datascience/polypseg/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612'
    ((train_x, train_y), (validation_x, validation_y), (test_x, test_y))=load_data(path)
    print(len(test_y))
    
    ds=tf_dataset(test_x, test_y)
    for x,y in ds:
        print(x.shape, y.shape)
        break

