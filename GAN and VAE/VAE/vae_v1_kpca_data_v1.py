# -*- coding: utf-8 -*-

### 兩層normalised function -> the best

### add dropout, adjust LR


# Commented out IPython magic to ensure Python compatibility.
# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import LeakyReLU
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn import preprocessing
import zipfile
import os

from google.colab import files
uploaded = files.upload()

# def un_zip(file_name):
#     """unzip zip file"""
#     zip_file = zipfile.ZipFile(file_name)
#     if os.path.isdir(file_name + "_files"):
#         pass
#     else:
#         os.mkdir(file_name + "_files")
#     for names in zip_file.namelist():
#         zip_file.extract(names,file_name + "_files/")
#     zip_file.close()
    
    
# un_zip("/content/train_all_gan_input (1).zip")

# pwd

data = pd.read_csv("1434.csv",encoding='GB18030')

data['Gross Spread']=data['Gross Spread'].apply(lambda x: 0.015 if x == 'null0.0' else x)
data['Gross Spread']=data['Gross Spread'].apply(lambda x: 0.015 if x == 'X0.0' else x)
data['Time']=data['Time'].apply(lambda x: str(int(x.split('/')[0])+1911)+'/'+x.split('/',1)[1] if len(x.split('/')[0])==3  else x )
data=data.drop(['Time'],axis=1)
for i in data.columns:
  data[i]=data[i].apply(lambda x: str(x).replace(",", "."))
  data[i]=data[i].apply(lambda x: float(x) )
data.shape

for i in data.columns:
  data[i]=data[i].fillna(data[i].mean())

data

data=np.array(data)
## test 
# data=data[:500]

data.shape

# scaler = preprocessing.StandardScaler().fit(data)
# data=scaler.transform(data)

mm=MinMaxScaler()
mm=MinMaxScaler().fit(data)
data=mm.transform(data)

from __future__ import print_function, division
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add,Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import sys
from tensorflow.keras.layers import Layer
import numpy as np
from sklearn import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2
data_dim = 8

encoder_inputs = keras.Input(shape=(data_dim, ))
x = layers.Dense(128, activation="relu")(encoder_inputs)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dense(8, activation="relu")(x)
x = layers.Dense(data_dim, activation="sigmoid")(x)

decoder = keras.Model(latent_inputs, x, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= data_dim
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

epochs=100
batch_size=128
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizer, loss=loss_fn)
history=vae.fit(data, epochs=epochs, batch_size=batch_size)

fig = plt.figure(figsize=(16,4),dpi=100) 
plt.style.use('seaborn-ticks') 
ax1 = fig.add_subplot(121) 
ax1.title.set_text('model loss')
ax1.plot(history.history['kl_loss'],'b-',linestyle="--" )
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend([ 'kl_loss'], loc='upper left')
ax1.grid(linestyle='-.')

ax2 = fig.add_subplot(122)
ax2.title.set_text('model loss') 
ax2.plot(history.history['reconstruction_loss'], 'g-',linestyle="--" )
ax2.plot(history.history['loss'],'r-',linestyle="--" )
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train_loss', 'reconstruction_loss'], loc='upper left')
ax2.grid(linestyle='-.')


plt.show()

import matplotlib.pyplot as plt


def plot_latent(encoder, decoder, data):
  n = 20
  scale = 5.0
  figsize = 15
  figure = np.zeros((n, data.shape[1]))
  # linearly spaced coordinates corresponding to the 2D plot
  # of digit classes in the latent space
  grid_x = np.linspace(-scale, scale, n)
  grid_y = np.linspace(-scale, scale, n)[::-1]

  x_decoded_list=[]
  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          x_decoded=mm.inverse_transform(x_decoded)

          x_decoded_list.extend(x_decoded.tolist())

  return x_decoded_list
x_decoded_list=plot_latent(encoder, decoder, data)

def summarize_performance(x_decoded_list):
  idx = np.random.randint(0, data.shape[0], len(x_decoded_list))
  imgs = data[idx]

  list_all = pd.concat([pd.DataFrame(imgs), pd.DataFrame(x_decoded_list)], ignore_index=True)
  ##############################################################################
  print('tSNE - Start')
  tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
  list_all=np.array(list_all)
  list_all_out = tsne.fit_transform(list_all)
  list_all_out_r=list_all_out[:len(imgs)]
  list_all_out_f=list_all_out[len(imgs):]

  plt.figure(figsize=(8,4),dpi=100)  
  plt.style.use('seaborn-ticks')   
  aa=plt.scatter(list_all_out_r[:,0], list_all_out_r[:,1], color='red')
  bb=plt.scatter(list_all_out_f[:,0], list_all_out_f[:,1], color='blue')  
  plt.legend(handles=[aa,bb],labels=['Real','Fake']) 
  plt.title('tSNE Distribution')
  plt.show()

  tsne = TSNE(n_components=1, verbose=0, perplexity=40, n_iter=300)
  list_all=np.array(list_all)
  list_all_out = tsne.fit_transform(list_all)
  list_all_out_r=list_all_out[:len(imgs)]
  list_all_out_f=list_all_out[len(imgs):]

  sns.set_style("ticks")   
  plt.figure(figsize=(8,4),dpi=100)  
  sns.distplot(list_all_out_r, hist=False, kde=True, rug=True,   
  kde_kws={"color":"lightcoral", "lw":1.5, 'linestyle':'--'},  
  rug_kws={'color':'lightcoral','alpha':1, 'lw':2,}, label='Real Data')
  sns.distplot(list_all_out_f, hist=False, kde=True, rug=True,
  kde_kws={"color":"lightseagreen", "lw":1.5, 'linestyle':'--'}, 
  rug_kws={'color':'lightseagreen', 'alpha':0.5, 'lw':2, 'height':0.1}, label='Fake Data')
  # plt.ylim([0,0.05])
  plt.grid(linestyle='--')
  plt.title(("Real/Fake Data Distribution"))


summarize_performance(x_decoded_list)

x_decoded_list[:100]

from google.colab import files
x_decoded_list=pd.DataFrame(x_decoded_list)
x_decoded_list.to_csv('./x_decoded_list.csv', sep = ',', index = False)
files.download('x_decoded_list.csv')



























