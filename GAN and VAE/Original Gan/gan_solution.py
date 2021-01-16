# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from keras.optimizers import Adam
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
data

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



# x,y=generate_real_samples(data, 1)

# x.shape

# y.shape

# example : x.shape (100, 2)

# example : y.shape (100, 1)

# size of the latent space
latent_dim = 64
dis_input_dim = gen_output_dim = 8
n_epochs=100000
count_visualisation=50
generate_fake_sample_num=64

data.shape

# define the standalone discriminator model
def define_discriminator(n_inputs=dis_input_dim):
  model = Sequential()
  model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(32,  kernel_initializer='he_uniform'))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(16,  kernel_initializer='he_uniform'))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(4,  kernel_initializer='he_uniform'))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(1, activation='sigmoid'))
  # compile model
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
  return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=gen_output_dim):
  model = Sequential()
  model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(32,  kernel_initializer='he_uniform'))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(16,  kernel_initializer='he_uniform'))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(n_outputs, activation='linear'))
  return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005))
	return model

def generate_real_samples(data, n):
  data=np.array(data)
  # generate class labels
  y = np.ones((data.shape[0], 1))
  X_train, X_test, y_train, y_test = train_test_split( data, y, test_size=n)
  return X_test, y_test


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim=latent_dim, n= generate_fake_sample_num, data=data):
  # prepare real samples
  x_real, y_real = generate_real_samples(data, n=n)
  # evaluate discriminator on real examples
  _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
  # prepare fake examples
  x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
  # evaluate discriminator on fake examples
  _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
  # summarize discriminator performance
  print('epoch - {}'.format(epoch))
  print("x_fake - {}".format(x_fake))
  print('acc_real - {}'.format(acc_real))
  print('acc_fake - {}'.format(acc_fake))

  list_all = pd.concat([pd.DataFrame(x_real), pd.DataFrame(x_fake)], ignore_index=True)

  print('tSNE - Start')
  tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
  list_all=np.array(list_all)
  list_all_out = tsne.fit_transform(list_all)
  list_all_out_r=list_all_out[:len(x_real)]
  list_all_out_f=list_all_out[len(x_real):]
  plt.figure(figsize=(8,4),dpi=100)  
  plt.style.use('seaborn-ticks') 
  aa=plt.scatter(list_all_out_r[:, 0], list_all_out_r[:, 1], color='red')
  bb=plt.scatter(list_all_out_f[:, 0], list_all_out_f[:, 1], color='blue')  
  plt.legend(handles=[aa,bb],labels=['Real','Fake']) 
  plt.title('tSNE - Epoch {}'.format(epoch+1))
  plt.show()

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim,data=data, n_epochs=n_epochs, n_batch=128, n_eval=2000):
  # determine half the size of one batch, for updating the discriminator
  half_batch = int(n_batch / 2)
  # manually enumerate epochs
  for i in range(n_epochs):
    # prepare real samples
    x_real, y_real = generate_real_samples(data, n=half_batch)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    # update discriminator
    d_model.train_on_batch(x_real, y_real)
    d_model.train_on_batch(x_fake, y_fake)
    # prepare points in latent space as input for the generator
    x_gan = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    y_gan = ones((n_batch, 1))
    # update the generator via the discriminator's error
    gan_model.train_on_batch(x_gan, y_gan)
    # evaluate the model every n_eval epochs
    if (i+1) % n_eval == 0:
      summarize_performance(i, g_model, d_model, latent_dim)


# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)













# def generate_real_samples(data,n):
#   data=np.array(data)
#   # generate class labels
#   y = np.ones((data.shape[0], 1))

#   X_train, X_test, y_train, y_test = train_test_split( data, y, test_size=n)
#   return X_train, y_train

# # define the standalone discriminator model
# def define_discriminator(n_inputs=dis_input_dim):
#   model = Sequential()
#   model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(32,  kernel_initializer='he_uniform'))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(16,  kernel_initializer='he_uniform'))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(4,  kernel_initializer='he_uniform'))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(1, activation='sigmoid'))
#   # compile model
#   model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
#   return model

# # define the standalone generator model
# def define_generator(latent_dim, n_outputs=gen_output_dim):
#   model = Sequential()
#   model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(32,  kernel_initializer='he_uniform'))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(16,  kernel_initializer='he_uniform'))
#   model.add(LeakyReLU(alpha=0.01))
#   model.add(Dense(n_outputs, activation='linear'))
#   return model

# # define the combined generator and discriminator model, for updating the generator
# def define_gan(generator, discriminator):
# 	# make weights in the discriminator not trainable
# 	discriminator.trainable = False
# 	# connect them
# 	model = Sequential()
# 	# add generator
# 	model.add(generator)
# 	# add the discriminator
# 	model.add(discriminator)
# 	# compile model
# 	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
# 	return model

# # generate points in latent space as input for the generator
# def generate_latent_points(latent_dim, n):
# 	# generate points in the latent space
# 	x_input = randn(latent_dim * n)
# 	# reshape into a batch of inputs for the network
# 	x_input = x_input.reshape(n, latent_dim)
# 	return x_input

# # use the generator to generate n fake examples, with class labels
# def generate_fake_samples(generator, latent_dim, n):
# 	# generate points in latent space
# 	x_input = generate_latent_points(latent_dim, n)
# 	# predict outputs
# 	X = generator.predict(x_input)
# 	# create class labels
# 	y = zeros((n, 1))
# 	return X, y

# # evaluate the discriminator and plot real and fake points
# def summarize_performance(epoch, generator, discriminator, latent_dim, data, gen_output_dim= gen_output_dim, count_visualisation=count_visualisation, n=1):
#   fake_list=[]
#   x_real, y_real = generate_real_samples(data,n)
#   real_list=np.concatenate((x_real, y_real), axis=1)
 
#   for i in range(count_visualisation):
#     x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
#     for i in x_fake:
#       i=list(i)
#       # i.append(float(y_fake))
#       i.append(0)
#       fake_list.append(i)
#   list_all = pd.concat([pd.DataFrame(real_list), pd.DataFrame(fake_list)], ignore_index=True)
#   # evaluate discriminator on examples
#   _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
#   _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
#   # summarize discriminator performance
#   print('epoch - {}'.format(epoch))
#   print("x_fake - {}".format(x_fake))
#   print('acc_real - {}'.format(acc_real))
#   print('acc_fake - {}'.format(acc_fake))

#   # scatter plot real and fake data points

#   print('tSNE - Start')
#   tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
#   list_all=np.array(list_all)
#   list_all_out = tsne.fit_transform(list_all)
#   # classes = ['Fake Data', 'True Data']
#   # colours = ListedColormap(['r','b'])
#   list_all_out_r=list_all_out[:len(real_list)]
#   list_all_out_f=list_all_out[len(real_list):]
#   aa=plt.scatter(list_all_out_r[:, 0], list_all_out_r[:, 1],  color='red' )
#   bb=plt.scatter(list_all_out_f[:, 0], list_all_out_f[:, 1], color='blue' )  
#   plt.legend(handles=[aa,bb],labels=['Real','Fake']) 
#   # plt.scatter(list_all_out[:, 0], list_all_out[:, 1], c=list_all[:,-1], cmap=colours )
#   # plt.gca().legend(('Fake Data', 'True Data'))
#   # plt.legend(*scatter.legend_elements(), labels=classes)
#   # plt.scatter(list_all_out[:, 0], list_all_out[:, 1], c=list_all[:,2] )
#   # plt.scatter(fake_list[:, 0], fake_list[:, 1], color='blue')
#   plt.show()

  
# # train the generator and discriminator
# def train(g_model, d_model, gan_model, latent_dim,data ,n_epochs=n_epochs, n_batch=128, n_eval=2000):
#   # determine half the size of one batch, for updating the discriminator
#   half_batch = int(n_batch / 2)
#   n=half_batch/n_batch
#   # manually enumerate epochs
#   for i in range(n_epochs):
#     # prepare real samples
#     x_real, y_real = generate_real_samples(data,n)
#     # prepare fake examples
#     x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#     # update discriminator
#     d_model.train_on_batch(x_real, y_real)
#     d_model.train_on_batch(x_fake, y_fake)
#     # prepare points in latent space as input for the generator
#     x_gan = generate_latent_points(latent_dim, n_batch)
#     # create inverted labels for the fake samples
#     y_gan = ones((n_batch, 1))
#     # update the generator via the discriminator's error
#     gan_model.train_on_batch(x_gan, y_gan)
#     # evaluate the model every n_eval epochs
#     if (i+1) % n_eval == 0:
#       summarize_performance(i, g_model, d_model, latent_dim, data)



# # create the discriminator
# discriminator = define_discriminator()
# # create the generator
# generator = define_generator(latent_dim)
# # create the gan
# gan_model = define_gan(generator, discriminator)
# # train model
# train(generator, discriminator, gan_model, latent_dim, data)

data=data[:1000]

# ## TEST ### 

# # define the standalone discriminator model
# def define_discriminator(n_inputs=dis_input_dim):
# 	model = Sequential()
# 	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# compile model
# 	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# 	return model

# # define the standalone generator model
# def define_generator(latent_dim, n_outputs=gen_output_dim):
# 	model = Sequential()
# 	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
# 	model.add(Dense(n_outputs, activation='linear'))
# 	return model

# # define the combined generator and discriminator model, for updating the generator
# def define_gan(generator, discriminator):
# 	# make weights in the discriminator not trainable
# 	discriminator.trainable = False
# 	# connect them
# 	model = Sequential()
# 	# add generator
# 	model.add(generator)
# 	# add the discriminator
# 	model.add(discriminator)
# 	# compile model
# 	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005))
# 	return model

# # generate points in latent space as input for the generator
# def generate_latent_points(latent_dim, n):
# 	# generate points in the latent space
# 	x_input = randn(latent_dim * n)
# 	# reshape into a batch of inputs for the network
# 	x_input = x_input.reshape(n, latent_dim)
# 	return x_input

# # use the generator to generate n fake examples, with class labels
# def generate_fake_samples(generator, latent_dim, n):
# 	# generate points in latent space
# 	x_input = generate_latent_points(latent_dim, n)
# 	# predict outputs
# 	X = generator.predict(x_input)
# 	# create class labels
# 	y = zeros((n, 1))
# 	return X, y

# # evaluate the discriminator and plot real and fake points
#   ###################################  prepare fake examples  #################################
# def summarize_performance(epoch, generator, discriminator, latent_dim, data, gen_output_dim= gen_output_dim, count_visualisation=3, n=1):
#   fake_list=[]
#   x_real, y_real = generate_real_samples(data,n)
#   real_list=np.concatenate((x_real, y_real), axis=1)
 
#   for i in range(count_visualisation):
#     x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
#     for i in x_fake:
#       i=list(i)
#       # i =[1,1,1,1,1]
#       # i.append(float(y_fake)) -> [1,1,1,1,1,0]
#       # i.extend(float(y_fake))
#       i.append(float(y_fake))
#       fake_list.append(i)
#       # fake_list.extend(i)


#   list_all = pd.concat([pd.DataFrame(real_list), pd.DataFrame(fake_list)], ignore_index=True)
#   # evaluate discriminator on examples
#   _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
#   _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
#   # summarize discriminator performance
#   print("x_fake - {}".format(x_fake))
  
#   print('epoch - {}'.format(epoch))
#   print('acc_real - {}'.format(acc_real))
#   print('acc_fake - {}'.format(acc_fake))

#   # scatter plot real and fake data points

#   print('tSNE - Start')
#   tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
#   list_all=np.array(list_all)
#   import pdb
#   pdb.set_trace()
#   list_all_out = tsne.fit_transform(list_all)
#   plt.scatter(list_all_out[:, 0], list_all_out[:, 1], c=list_all[:,2] )
#   # plt.scatter(fake_list[:, 0], fake_list[:, 1], color='blue')
#   plt.show()


#   ###################################  prepare fake examples  #################################  

  
# # train the generator and discriminator
# def train(g_model, d_model, gan_model, latent_dim,data ,n_epochs=100000, n_batch=128, n_eval=2000):
#   # determine half the size of one batch, for updating the discriminator
#   half_batch = int(n_batch / 2)
#   n=half_batch/n_batch
#   # manually enumerate epochs
#   for i in range(n_epochs):
#     # prepare real samples
#     x_real, y_real = generate_real_samples(data,n)
#     # prepare fake examples
#     x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#     # update discriminator
#     d_model.train_on_batch(x_real, y_real)
#     d_model.train_on_batch(x_fake, y_fake)
#     # prepare points in latent space as input for the generator
#     x_gan = generate_latent_points(latent_dim, n_batch)
#     # create inverted labels for the fake samples
#     y_gan = ones((n_batch, 1))
#     # update the generator via the discriminator's error
#     gan_model.train_on_batch(x_gan, y_gan)
#     # evaluate the model every n_eval epochs
#     if (i+1) % n_eval == 0:
#       summarize_performance(i, g_model, d_model, latent_dim, data)



# # create the discriminator
# discriminator = define_discriminator()
# # create the generator
# generator = define_generator(latent_dim)
# # create the gan
# gan_model = define_gan(generator, discriminator)
# # train model
# train(generator, discriminator, gan_model, latent_dim, data)

