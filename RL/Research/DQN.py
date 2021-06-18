# -*- coding: utf-8 -*-
import os
import gym
import random
import numpy as np

from collections import deque

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')

        # 经验池
        self.memory_buffer = deque(maxlen=2000)
        self.gamma = 0.95
        '''Q_value的discount rate，以便计算未来reward的折扣回报, 因為讓電腦盡快做決策, 越剛開始reward越高
        '''
        self.epsilon = 1.0
        ''' epsilon初始值, 贪婪选择法的随机选择行为的程度, 若0-1的隨機變量大於此值, 則採隨機策略
        '''
        self.epsilon_decay = 0.995
        ''' epsilon的衰減率 
        '''
        # 最小随机探索的概率
        self.epsilon_min = 0.01
        ''' epsilon再衰減的thrshold, 若epsilon小於此值則不再衰減
        '''

        self.env = gym.make('CartPole-v0')

    def build_model(self):
        """ 
        NN 輸入為env
        NN 輸出為在這一個env之下, 預期到整個journey結束的reward
        """
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def update_target_model(self):
        """更新target_model
        """
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        """ε-greedy选择action

        Arguments:
            state: 状态

        Returns:
            action: 动作
        """
        if np.random.rand() <= self.epsilon:
             return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """向经验池添加数据

        Arguments:
            state: 状态
            action: 动作
            reward: 回报
            next_state: 下一个状态
            done: 游戏结束标志
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """更新epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """batch数据处理

        Arguments:
            batch: batch size

        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
         # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, batch)
        # 生成Q_target。
        states = np.array([d[0] for d in data]) # d[0]:state
        next_states = np.array([d[3] for d in data]) # d[3]:next_state

        y = self.model.predict(states) # 
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target
        '''
        - y is a Q table - 
        For example in this batch I have 2 samples and 3 actions, the value is the reward in the table:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        '''
        return states, y


    def train(self, episode, batch):
        """训练
        Arguments:
            episode: 游戏次数
            batch： batch size

        Returns:
            history: 训练记录
        """
        self.model.compile(loss='mse', optimizer=Adam(1e-3))

        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # 通过贪婪选择法ε-greedy选择action。
                x = observation.reshape(-1, 4)
                action = self.egreedy_action(x) # x: state
                observation, reward, done, _ = self.env.step(action)
                # 将数据加入到经验池。
                reward_sum += reward
                self.remember(x[0], action, reward, observation, done) # x[0]: state

                if len(self.memory_buffer) > batch:
                    # 训练
                    X, y = self.process_batch(batch) # X -> states, y -> y值
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # 减小egreedy的epsilon参数。
                    self.update_epsilon()

                    # 固定次数更新target_model
                    '''[注意] 兩個NN, 一個是current state NN, 一個是next state NN(多個回合才更新一次參數)
                    '''
                    if count != 0 and count % 20 == 0:
                        self.update_target_model()

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)
    
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))

        self.model.save_weights('dqn.h5')

        return history

    def play(self):
        """使用训练好的模型测试游戏.
        """
        observation = self.env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()

            x = observation.reshape(-1, 4)
            q_values = self.model.predict(x)[0]
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()


if __name__ == '__main__':
    model = DQN()
    history = model.train(600, 32)
    model.play()
