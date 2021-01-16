
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
import os
import gym
import time
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import pdb
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from stock_env_v5 import stock
import copy
import data_cleaning as clean
import plot_v5 as plot

'''
==============================================================
Version 5:

A3C + Stock_env 環境
ok to run
Using 2330 TW stck data 
back test update
Apply env to TW stock market

--------------------------
Plot update

==============================================================
'''



# A3C 為一個off-line training Algo

# global variables for threading
step = 0
history = {'episode': [], 'Episode_reward': []}
lock = threading.Lock()


class A3C:
    """A3C Algorithms with sparse action.
    """
    def __init__(self, observation_dim, env_list, env_list2):
        self.gamma = 0.97 # discount rate，以便计算未来reward的折扣回报, 因為讓電腦盡快做決策, 越剛開始reward越高
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.observation_dim = observation_dim
        self.action_dim = 3
        self.env_list=env_list
        self.env_list2 = env_list2
        self._build_model()
        self.optimizer = self._build_optimizer()
        # handle error
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_actor(self):
        """actor model. 
        output: action
        """
        inputs = Input(shape=(self.observation_dim,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(self.action_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def _build_critic(self):
        """critic model.
        output:reward
        """
        inputs = Input(shape=(self.observation_dim,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def _build_model(self):
        """build model for multi threading training.
        """
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Pre-compile for threading, a function that compiles the predict function
        self.actor._make_predict_function() 
        self.critic._make_predict_function()

    def _build_optimizer(self):
        """build optimizer and loss method.
        Returns:
            [actor optimizer, critic optimizer].
        """
        # actor optimizer - action - cross entropy loss
        actions = K.placeholder(shape=(None, self.action_dim))
        advantages = K.placeholder(shape=(None, 1))
        action_pred = self.actor.output

        '''
        Regularizaiton with Policy Entropy
        为何要加这一项呢？我们想要在 agent 与 environment 进行互动的过程中，平衡探索和利用，我们想去以一定的几率来尝试
        其他的 action，从而不至于采样得到的样本太过于集中。所以，引入这个 entropy，来使得输出的分布，能够更加的平衡
        '''
        entropy = K.sum(action_pred * K.log(action_pred + 1e-10), axis=1)
        closs = K.categorical_crossentropy(actions, action_pred)
        actor_loss = K.mean(closs * K.flatten(advantages)) - 0.01 * entropy

        actor_optimizer = Adam(lr=self.actor_lr)
        actor_updates = actor_optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        actor_train = K.function([self.actor.input, actions, advantages], [], updates=actor_updates)

        # critic optimizer - MSE loss 目標 - NN out = discounted_reward
        discounted_reward = K.placeholder(shape=(None, 1))
        value = self.critic.output

        critic_loss = K.mean(K.square(discounted_reward - value))

        critic_optimizer = Adam(lr=self.critic_lr)
        critic_updates = critic_optimizer.get_updates(self.critic.trainable_weights, [], critic_loss)
        critic_train = K.function([self.critic.input, discounted_reward], [], updates=critic_updates)

        return [actor_train, critic_train]

    def train(self, episode, n_thread, update_iter):
        """training A3C. parameter - (200, 4, 10)
        Arguments:
            episode: total training episode.
            n_thread: number of thread.
            update_iter: update iter.
        """
        # Multi threading training.
        import copy
        threads=[]
        for i in range(n_thread):
            locals()['env_list_'+str(i)] = copy.deepcopy(self.env_list)
#             locals()['env_list_val'+str(i)] = copy.deepcopy(self.env_list2)
            threads.append(Agent(i, self.actor, self.critic, self.optimizer, self.gamma, 
                                 episode, update_iter, self.observation_dim, self.action_dim, locals()['env_list_'+str(i)]))
#         threads = [Agent(i, self.actor, self.critic, self.optimizer, self.gamma, episode, update_iter, self.observation_dim, self.action_dim, env_list) for i in range(n_thread)]

        for t in threads:
            t.start()
            time.sleep(1)

        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")

        self.save()
        
    def test(self):
        self.load()
        print('======== Back Test Strart ========')
#         total_profit = 0
#         i = 0
        agent_back=Agent(10, self.actor, self.critic, self.optimizer, self.gamma, 1, 2, self.observation_dim, 
              self.action_dim, self.env_list2)
        
        agent_back.BackTest()


    def load(self):
        """Load model weights.
        """
        if os.path.exists('model/actor_a3cs.h5') and os.path.exists('model/critic_a3cs.h5'):
            self.actor.load_weights('model/actor_a3cs.h5')
            self.critic.load_weights('model/critic_a3cs.h5')

    def save(self):
        """Save model weights.
        """
        self.actor.save_weights('model/actor_a3cs.h5')
        self.critic.save_weights('model/critic_a3cs.h5')


class Agent(threading.Thread):
    """Multi threading training agent.
    """
    def __init__(self, index, actor, critic, optimizer, gamma, episode, update_iter, 
                 observation_dim, action_dim,  env_list):
        threading.Thread.__init__(self)

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.episode = episode
        self.update_iter = update_iter
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.env_list = env_list
        
        
    def BackTest(self):
        step=0
        profit = 0

        for id, env in enumerate(self.env_list): # 迴圈所有股票
            states = []
            actions = []
            rewards = []
            observation = env.reset()
            while True:
                
    #             observation, done = game_step(env, observation, train=False, show_log=show_log, my_trick=my_trick)
                # break while loop when end of this episode

                x = observation.reshape(-1, self.observation_dim)
                states.append(x)
                action = np.random.choice(np.array(range(len(self.actor.predict(x)[0]))), p=self.actor.predict(x)[0])
#                 if action == 1:
#                     pdb.set_trace()
                actions.append(action)

                next_observation, reward, done= env.step(action, my_trick=False)
                next_observation = next_observation.reshape(-1, self.observation_dim)
                rewards.append(reward)
                observation = next_observation[0]

                if done:
                    name = 'plt_' 
                    env.draw(name)
                    print('total_profit:%.3f' % (env.total_profit))
                    break
            # print('total_profit:%.3f' % (env.total_profit))
        
    def run(self):
        """training model.
        """
        global history
        global step
        total_profit_max=0
        
        total_profit_history=[]
        episode_reward_history=[]
        # 玩遊戲訓練次數: episode(200)
        while step < self.episode:
            profit = 0
            for id, env in enumerate(self.env_list): # 迴圈所有股票
#                 print('Epoisode {}, Import Stock - {}'.format(self.episode,id+1))

                observation = env.reset()

                states = []
                actions = []
                rewards = []

                # 每一次遊戲在這個迴圈中
                while True:
                    x = observation.reshape(-1, self.observation_dim)
                    states.append(x)

                    # choice action with prob. [ACTOR NN]
    #                 prob = self.actor.predict(x)[0][0]
                    action = np.random.choice(np.array(range(len(self.actor.predict(x)[0]))), p=self.actor.predict(x)[0]) # action = 0 or 1
                    actions.append(action) # actions: list of o or 1
                    
                    next_observation, reward, done= env.step(action)
                    next_observation = next_observation.reshape(-1, self.observation_dim)
                    rewards.append(reward)

                    observation = next_observation[0]

                    # 當玩完這次 episode, 會進入此 if -> 更新 NN 參數
                    # 當玩完次數等於update_iter倍數時也會進if -> 更新 NN 參數
                    if ((step + 1) % self.update_iter == 0) or done:

                        lock.acquire()
                        try:
                            self.train_episode(states, actions, rewards, next_observation, done)

                            if done:
                                episode_reward = sum(rewards)
                                history['episode'].append(step)
                                history['Episode_reward'].append(episode_reward)

                                print('Thread: {} | Episode: {} | Episode reward: {}'.format(self.index, step, episode_reward))
                                episode_reward_history.append(episode_reward)
                                step += 1
                        finally:
                            lock.release()

                    if done:
                        break
                profit += env.total_profit
                
            profit = profit / len(env_list)
            print('epoch:%d, total_profit:%.3f' % (step, env.total_profit))
            total_profit_history.append(env.total_profit)
            
        if self.index == 0:
            plot.plot_acc_loss(episode_reward_history, total_profit_history)
                
                
    def discount_reward(self, rewards, next_state, done):
        """Discount reward
        Arguments:
            rewards: rewards in a episode. [已打散在一個array中]
            next_states: next state of current game step.
            done: if epsiode done.
        Returns:
            discount_reward: n-step discount rewards.
        """
        # compute the discounted reward backwards through time.
        discount_rewards = np.zeros_like(rewards, dtype=np.float32)

        if done:
            cumulative = 0.
        else:            
            cumulative = self.critic.predict(next_state)[0][0]

        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discount_rewards[i] = cumulative

        return discount_rewards

    def train_episode(self, states, actions, rewards, next_observation, done):
        """training algorithm in an epsiode.
        states: list len=這一次episode目前為止次數, states[i]=[一次state的維度] ex:[s1,s2,s3,s4]
        actions:list len=這一次episode目前為止次數, states[i]=[一次action的維度] ex:[左, 右]
        rewards:list len=這一次episode目前為止次數, states[i]=該回合reward ex:3
        next_observation:下一次的環境返回, dim=[state dim] ex:[s1,s2,s3,s4]
        """
        
        # 全部打掉list, 混在一個array中
        states = np.concatenate(states, axis=0)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards)

        # Q_values - critic 預測 discounted_rewards
        values = self.critic.predict(states)
        # discounted rewards
        discounted_rewards = self.discount_reward(rewards, next_observation, done)
        discounted_rewards = discounted_rewards.reshape(-1, 1)
        # advantages
        advantages = discounted_rewards - values

        self.optimizer[1]([states, discounted_rewards])
        self.optimizer[0]([states, actions, advantages])



    
if __name__ == "__main__":
    max_round = 100 # training epoch
    test_num = 60 # test_number, approximately 20 days a month
    label_feature = 'close' # the base of sell or buy signal ---- the feature of dataset
    observation_dim = 6 # dim of features in dataset
    process_num = 3 # number of processing
    update_rate = 10 # update rate in NN
    file_path_list = ['/Users/calvin/GitHub/test/a3c+dqn_env/algo/2330_new.csv'] #### Run in Local Machine ####
    init_money = 500000
    feauture = ['open','high', 'low'] # The rest of the features in the dataset except 'label_feature'
    window_size = 6 # consider how many days for training
    
    env_list = []
    env_list2 = []
    for file_path in file_path_list:
        df = pd.read_csv(file_path)
        df = clean.cleaning(df).TW_day_based_data()
        df = df.sort_values(df.columns[0], ascending=True)
        df = df.iloc[22:].reset_index(drop=True) # 去除前幾天沒有均線訊息的
        train_count = len(df)-test_num 
        env_list.append(stock(df.iloc[0:train_count], label_feature, init_money, feauture, window_size))
        env_list2.append(stock(df.iloc[train_count:].reset_index(drop=True), label_feature, init_money, feauture, window_size))

    RL = A3C(observation_dim = observation_dim, env_list=env_list, env_list2=env_list2)
    RL.train(max_round, process_num, update_rate)
    RL.test()
    
    

