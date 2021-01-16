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
from sklearn.preprocessing import MinMaxScaler

import zipfile
import os

from google.colab import files
uploaded = files.upload()

def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names,file_name + "_files/")
    zip_file.close()
    
    
un_zip("/content/gan_input_kpca.csv.zip")
# un_zip("/content/kaggle/test.zip")
# un_zip("/content/kaggle/train_labels.csv.zip")

ls

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

# data=data[:1000]

# def generate_real_samples(n):
# 	# generate inputs in [-0.5, 0.5]
# 	X1 = rand(n) - 0.5
# 	# generate outputs X^2
# 	X2 = X1 * X1
# 	# stack arrays
# 	X1 = X1.reshape(n, 1)
# 	X2 = X2.reshape(n, 1)
# 	X = hstack((X1, X2))
# 	# generate class labels
# 	y = ones((n, 1))
# 	return X, y

data=np.array(data)
## test 
# data=data[:500]

mm=MinMaxScaler()
mm.fit(data)
data=mm.transform(data)

from __future__ import print_function, division
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add,Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial
import keras.backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import sys
from tensorflow.keras.layers import Layer
import numpy as np

import time

# scaler = preprocessing.StandardScaler().fit(data)
# data=scaler.transform(data)

# mm=MinMaxScaler()
# mm=MinMaxScaler().fit(data)
# data=mm.transform(data)

t1 = time.time()
batch_size=32

class RandomWeightedAverage(Concatenate):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        batch_size=32  ## Attention here
        alpha = K.random_uniform((batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GradientPenalty(Layer):
  def call(self, inputs):
      target, wrt = inputs
      grad = K.gradients(target, wrt)[0]
      return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), 
          axis=1, keepdims=True))-1

  def compute_output_shape(self, input_shapes):
      return (input_shapes[1][0], 1)

class WGANGP():
  def __init__(self, data, batch_size):
    self.data=data
    self.batch_size=batch_size
    self.generate_fake_sample_num=128 # How many sample for the final generation  ## 記得修改為 2000
    self.img_rows = 8
    self.img_shape = (self.img_rows)
    self.latent_dim = 50
    self.n_critic = 5   # How many recursive training per one update from generator for discriminator 
    optimizer_critic = RMSprop(lr= 0.00005)
    optimizer_gen = RMSprop(lr=   0.00005)


    self.generator = self.build_generator()
    self.critic = self.build_discriminator()

    #-------------------------------
    # Construct Computational Graph
    #    for the Critic
    #-------------------------------
    # Freeze generator's layers while training critic
    self.generator.trainable = False
    # Image input (real sample)
    real_img = Input(shape=self.img_shape)

    # Noise input
    z_disc = Input(shape=(self.latent_dim,))
    fake_img = self.generator(z_disc)


    fake = self.critic(fake_img)
    valid = self.critic(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])
    # Determine validity of weighted sample
    validity_interpolated = self.critic(interpolated_img)

    gp = GradientPenalty()([validity_interpolated, interpolated_img])

    self.critic_model = Model(inputs=[real_img, z_disc],
                        outputs=[valid, fake, gp ])
    self.critic_model.compile(loss=[self.wasserstein_loss,self.wasserstein_loss,'mse'],
                                    optimizer=optimizer_critic,
                                    loss_weights=[1, 1, 10])   

    #-------------------------------
    # Construct Computational Graph
    #    for Generator
    #-------------------------------

    # For the generator we freeze the critic's layers
    self.critic.trainable = False
    self.generator.trainable = True

    # Sampled noise for input to generator
    z_gen = Input(shape=(self.latent_dim,))
    # Generate images based of noise
    img = self.generator(z_gen)
    # Discriminator determines validity
    valid = self.critic(img)
    # Defines generator model
    self.generator_model = Model(z_gen, valid)
    self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer_gen) 


  def wasserstein_loss(self, y_true, y_pred):
    return K.mean(y_true * y_pred)

  def build_generator(self):
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(32,  kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(16,  kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(self.img_rows, activation='tanh'))
    model.summary()
    noise = Input(shape=(self.latent_dim,))
    img = model(noise)
    return Model(noise, img)

  def build_discriminator(self):
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=self.img_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(32,  kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(16,  kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(4,  kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1))
    model.summary()
    img = Input(shape=self.img_shape)
    validity = model(img)
    return Model(img, validity)

  def plot_loss(self, d_loss_list, g_loss_list, n_eval, epochs):
    ##############################################################################
    plt.style.use('seaborn-ticks')  
    fig = plt.figure(figsize=(8,4),dpi=100)
    ax1 = fig.add_subplot(111)
    lns1=ax1.plot( [(i+1)*n_eval  for i in range(len(d_loss_list)) ], d_loss_list,'r-',linestyle="--" , label="d_loss_list")
    ax1.set_ylabel('D Loss')
    ax1.set_xlabel('epochs')
    ax1.set_title("Loss Plot")
    ax2 = ax1.twinx() 
    lns2=ax2.plot( [(i+1)*n_eval  for i in range(len(g_loss_list)) ], g_loss_list,'b-',linestyle="--" , label="g_loss_list")
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel('G Loss')
    ax2.set_xlabel('epochs')
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid(linestyle='-.')
    plt.show()
    ##############################################################################


  def evaluation_distribution(self, real, fake):
    '''
    Evaluate the mean and std from fake data and real data
    '''
    real=pd.DataFrame(real)
    fake=pd.DataFrame(fake)
    df = pd.DataFrame(0, index=['real_mean', 'real_std','fake_mean', 'fake_std'], columns=real.columns)
    ## Columns - raaw data feature list
    for i in range(len(real.columns)):
      df.iloc[:2,i] = list([real.iloc[: , i].mean(), real.iloc[: , i].std()])
      df.iloc[2:,i] = list([fake.iloc[: , i].mean(), fake.iloc[: , i].std()])
    print(pd.DataFrame(df))
    ##############################################################################
    list_all = pd.concat([pd.DataFrame(real), pd.DataFrame(fake)], ignore_index=True)
    tsne = TSNE(n_components=1, verbose=0, perplexity=40, n_iter=300)
    list_all=np.array(list_all)
    list_all_out = tsne.fit_transform(list_all)
    list_all_out_r=list_all_out[:len(real)]
    list_all_out_f=list_all_out[len(real):]

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
    ##############################################################################
    return df


  def train(self, epochs, n_eval):
    batch_size=self.batch_size
    X_train=self.data
    # Adversarial ground truths
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
    d_loss_list = []
    g_loss_list = []
    for epoch in range(epochs):
      for _ in range(self.n_critic):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        d_loss = self.critic_model.train_on_batch([imgs, noise],[valid, fake, dummy])

      # ---------------------
      #  Train Generator
      # ---------------------

      g_loss = self.generator_model.train_on_batch(noise, valid)

      # If at save interval => save generated image samples
      if (epoch+1) % n_eval == 0:
        d_loss_list.append(d_loss[-1])
        g_loss_list.append(g_loss)
        print ("%d [D wasserstein loss: %f] [G wasserstein loss: %f]" % (epoch,  d_loss[-1],  g_loss))
        x_fake=self.summarize_performance(epoch, imgs, epochs)

    
    gen_imgs = self.generator.predict(noise)
    self.plot_loss(d_loss_list, g_loss_list, n_eval, epochs)
    df=self.evaluation_distribution(imgs, gen_imgs)
    return x_fake, df


  def summarize_performance(self,epoch, imgs, epochs):
    # prepare real samples
    # x_real, y_real = generate_real_samples(data, n=n)
    # evaluate discriminator on real examples
    # _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    if epoch+1==epochs:
      noise = np.random.normal(0, 1, (self.generate_fake_sample_num, self.latent_dim))
      x_fake = self.generator.predict(noise)

    else:
      noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
      x_fake = self.generator.predict(noise)
    # x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    # _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance


    aa=MinMaxScaler().fit(x_fake)
    x_fake=aa.transform(x_fake)
    x_fake=mm.inverse_transform(x_fake)

    print('epoch - {}'.format(epoch+1))
    print("x_fake - {}".format(x_fake))
    # print('acc_real - {}'.format(acc_real))
    # print('acc_fake - {}'.format(acc_fake))

    list_all = pd.concat([pd.DataFrame(imgs), pd.DataFrame(x_fake)], ignore_index=True)
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
    plt.title('tSNE - Epoch {}'.format(epoch+1))
    plt.show()
    ##############################################################################

    return x_fake

if __name__ == '__main__':
  wgan = WGANGP(data, batch_size)
  x_fake, df=wgan.train(epochs=450, n_eval=2)
  t2 = time.time()
  print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')

from google.colab import files
x_fake=pd.DataFrame(x_fake)
df=pd.DataFrame(df)
x_fake.to_csv('./x_fake.csv', sep = ',', index = False)
df.to_csv('./df.csv', sep = ',', index = False)

# files.download('submission111111.csv')

files.download('x_fake.csv')
files.download('df.csv')

!ls



















