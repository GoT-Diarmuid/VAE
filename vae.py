import numpy as np
import pandas as pd
from sklearn import preprocessing
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from PIL import Image
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import os
import random
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical

def save_img(imgs,names):
  img_new = Image.new('L',(280,280))
  index = 0
  for i in range(0,280,80):
    for j in range(0,280,80):
      img = imgs[index]
      img = Image.fromarray(img,mode='L')
      img_new.paste(img,(i,j))
      index+=1
  img_new.save(names)

def feature_scale(x):
  x = tf.cast(x,dtype=tf.float32)/255.
#  y = tf.cast(y,dtype=tf.int32)
  return x
  
#Dim reduct nums
dim_reduce = 10
batch_num = 128
lr = 1e-3

filenames = os.listdir("D:/[Study]/Dataset/dog vs cat/dataset/training_set")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    else:
        categories.append('cat')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True)

all_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

all_df['category'].value_counts()

(x,y),(x_test,y_test) = datasets.fashion_mnist.load_data()
data = tf.data.Dataset.from_tensor_slices(x)
data = data.map(feature_scale).shuffle(10000).batch(128)

data_test = tf.data.Dataset.from_tensor_slices(x_test)
data_test = data_test.map(feature_scale).batch(128)

data_iter = iter(data)
samples = next(data_iter)
print(samples[0].shape,samples[1].shape)

class VAE(keras.Model):
  def __init__(self):
    super(VAE,self).__init__()
    #encoder
    self.fc_layer_1 = layers.Dense(128)
    self.fc_layer_2 = layers.Dense(dim_reduce)
    self.fc_layer_3 = layers.Dense(dim_reduce)
    self.fc_layer_4 = layers.Dense(128)
    self.fc_layer_5 = layers.Dense(784)
    

  def model_encoder(self, x):
    h = tf.nn.relu(self.fc_layer_1(x))
    mean_fc = self.fc_layer_2(h)
    var_fc = self.fc_layer_3(h)
    return mean_fc,var_fc

  def model_decoder(self, z):
    out = tf.nn.relu(self.fc_layer_4(z))
    out = self.fc_layer_5(out)
    return out

  def reparameter(self,mean_x,var_x):
    eps = tf.random.normal(var_x.shape)
    std = tf.exp(var_x)**0.5
    z = mean_x + std*eps
    return z
  
  def call(self, inputs, training=None):
    mean_x,var_x = self.model_encoder(inputs)
    z = self.reparameter(mean_x,var_x)
    x = self.model_decoder(z)
    return x,mean_x,var_x

model = VAE()
model.build(input_shape=(4,784))
optimizer = optimizers.Adam(lr=lr)
model.summary()
model.compile(optimizer = optimizer)

optimizer = optimizers.Adam(lr=lr)
for i in range(10):
  for step,x in enumerate(data):
    x = tf.reshape(x,[-1,784])
    with tf.GradientTape() as tape:
      logits,mean_x,var_x = model(x)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=logits)
      loss = tf.reduce_sum(loss)/x.shape[0]
      kl_div = -0.5*(var_x+1-mean_x**2-tf.exp(var_x))
      kl_div = tf.reduce_sum(kl_div)/x.shape[0]
      
      loss = loss + 1.*kl_div
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    
    if step %100==0:
      print(i,step,'loss:',float(loss),'kl_div:',float(kl_div))
      
  x = next(iter(data_test))
  val_x = tf.reshape(x,[-1,784])
  logits,_,_ = model(val_x)
  x_hat = tf.sigmoid(logits)
  x_hat = tf.reshape(x_hat,[-1,28,28])
  x_hat = x_hat.numpy()*255
  x_hat = x_hat.astype(np.uint8)
  save_img(x_hat,'img_result/AE_img_%d.png'%i)