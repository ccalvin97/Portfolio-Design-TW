# -*- coding: utf-8 -*-


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

from google.colab import files
uploaded = files.upload()

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

data.shape

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import seaborn as sns
import keras.backend as K

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time

mm=MinMaxScaler()
mm.fit(data)
data=mm.transform(data)

t1 = time.time()
class WGAN():
  def __init__(self, data):
    self.data=data
    self.img_rows = 8
    self.img_shape = (self.img_rows)
    self.latent_dim = 50
    # How many recursive training per one update from generator for discriminator 
    self.n_critic = 10
    self.clip_value = 0.01
    optimizer_critic = RMSprop(lr=0.00005)
    optimizer_gen = RMSprop(lr=0.000005)


    # Build and compile the critic
    self.critic = self.build_discriminator()
    self.critic.compile(loss=self.wasserstein_loss,
        optimizer=optimizer_critic,
        metrics=['accuracy'])

    # Build the generator
    self.generator = self.build_generator()

    # The generator takes noise as input and generated imgs
    z = Input(shape=(self.latent_dim,))
    img = self.generator(z)

    # For the combined model we will only train the generator
    self.critic.trainable = False

    # The critic takes generated images as input and determines validity
    valid = self.critic(img)

    # The combined model  (stacked generator and critic)
    self.combined = Model(z, valid)
    self.combined.compile(loss=self.wasserstein_loss,
        optimizer=optimizer_gen,
        metrics=['accuracy'])

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

  def plot_acc_loss(self, d_loss_list, g_loss_list, n_eval, epochs):
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

  def train(self, epochs, n_eval, batch_size=128):
    X_train=self.data
    # Adversarial ground truths
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    eval_loss_real = []
    eval_acc_fake = []
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

        # Generate a batch of new images
        gen_imgs = self.generator.predict(noise)

        # Train the critic
        d_loss_real = self.critic.train_on_batch(imgs, valid)
        d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        # eval_loss_real.append(d_loss_real[0])
        # eval_acc_fake.append(d_loss_fake[0])
        
        # Clip critic weights
        for l in self.critic.layers:
          weights = l.get_weights()
          weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
          l.set_weights(weights)

      # ---------------------
      #  Train Generator
      # ---------------------

      g_loss = self.combined.train_on_batch(noise, valid)
      # If at save interval => save generated image samples
      if (epoch+1) % n_eval == 0:
        eval_loss_real.append(d_loss[0])
        eval_acc_fake.append(g_loss[0])
        print ("%d [D wasserstein loss: %f] [G wasserstein loss: %f]" % (epoch,  d_loss[0]*10000,  g_loss[0]*10000))
        x_fake=self.summarize_performance(epoch,imgs)

    self.plot_acc_loss(eval_loss_real, eval_acc_fake, n_eval, epochs)
    df=self.evaluation_distribution(imgs, gen_imgs)
    return x_fake, df

  def summarize_performance(self,epoch, imgs):
    # prepare real samples
    # x_real, y_real = generate_real_samples(data, n=n)
    # evaluate discriminator on real examples
    # _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    noise = np.random.normal(0, 1, (32, self.latent_dim))
    x_fake = self.generator.predict(noise)

    # x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    # _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    aa=MinMaxScaler().fit(x_fake)
    x_fake=aa.transform(x_fake)
    x_fake=mm.inverse_transform(x_fake)

    print('epoch - {}'.format(epoch))
    print("x_fake - {}".format(x_fake))
    # print('acc_real - {}'.format(acc_real))
    # print('acc_fake - {}'.format(acc_fake))

    list_all = pd.concat([pd.DataFrame(imgs), pd.DataFrame(x_fake)], ignore_index=True)

    print('tSNE - Start')
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    list_all=np.array(list_all)
    list_all_out = tsne.fit_transform(list_all)
    list_all_out_r=list_all_out[:len(imgs)]
    list_all_out_f=list_all_out[len(imgs):]

    plt.figure(figsize=(8,4),dpi=100)  
    plt.style.use('seaborn-ticks') 
    aa=plt.scatter(list_all_out_r[:, 0], list_all_out_r[:, 1], color='red')
    bb=plt.scatter(list_all_out_f[:, 0], list_all_out_f[:, 1], color='blue')  
    plt.legend(handles=[aa,bb],labels=['Real','Fake']) 
    plt.title('tSNE - Epoch {}'.format(epoch+1))
    plt.show()

if __name__ == '__main__':
  wgan = WGAN(data)
  x_fake, df= wgan.train(epochs=2000, batch_size=32, n_eval=50)
  t2 = time.time()
  print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')

from google.colab import files
x_fake_wgan=pd.DataFrame(x_fake)
df_wgan=pd.DataFrame(df)
x_fake_wgan.to_csv('./x_fake_wgan.csv', sep = ',', index = False)
df_wgan.to_csv('./df_wgan.csv', sep = ',', index = False)

files.download('x_fake_wgan.csv')
files.download('df_wgan.csv')

































