# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpl_finance as mpf
import matplotlib.ticker as ticker
import pdb
import seaborn as sns 
import plot_v5 as plot
sns.set()


class stock:
    
    def __init__(self, df, feature_label, init_money, feauture, window_size):
        '''
        df: Input data, type DataFrame
        feature_label: column for close price, type str
        '''
        self.feature_label = feature_label
        self.feauture = feauture
        self.n_actions = 3 # 動作數量
        self.n_feature_labels = window_size # 特徵數量, 表示你一次看幾期data
        self.trend = df[feature_label].values # 收盤數據
        self.df = df #數據的DataFrame
        self.init_money = init_money # 初始化資金
        
        self.window_size = window_size #滑動窗口大小
        self.half_window = window_size // 2
        
        self.buy_rate = 0.0003 # 買入費率
        self.sell_rate = 0.0003 # 賣出費率
        self.stamp_duty = 0.001425 # 印花稅
        
    def reset(self):
        self.hold_money = self.init_money # 持有資金
        self.buy_num = 0 # 買入數量
        self.hold_num = 0 # 持有股票數量
        self.stock_value = 0 # 持有股票總市值
        self.maket_value = 0 # 總市值（加上現金）
        self.last_value = self.init_money # 上一天市值
        self.total_profit = 0 # 總盈利
        self.t = self.window_size // 2 # 時間
        self.reward = 0 # 收益
        # self.inventory = []
        
        self.states_sell = [] #賣股票時間
        self.states_buy = []  #買股票時間
        self.maket_value_list =[ self.init_money for i in range(self.t)] # 歷史總市值包含錢和股票
        self.stock_value_list =[ 0 for i in range(self.t) ] # 歷史持有股票總市值
        self.profit_rate_account = [0 for i in range(self.t)] # 賬號盈利
        
        self.profit_rate_stock =[0]
        for i in range(1, int(self.window_size // 2)):
            self.profit_rate_stock.append((self.trend[i] - self.trend[0]) / self.trend[0]) # 股票波動情況
        
        return self.get_state(self.t)
    
    def get_state(self, t): #某t時刻的狀態
        # get_state: 輸入當前時刻，得到當前時刻的狀態。這裡的狀態將過去的收盤價的漲跌幅當做狀態輸入。
        window_size = self.window_size + 1
        d = t - window_size + 1
        #早期天數不夠窗口打小，用0時刻來湊，即填補相應個數
        # block = self.trend[d : t + 1] if d >= 0 else (-d * [self.trend[0]] + self.trend[0 : t + 1])
        block = []
        if d<0:
            for i in range(-d):
                block.append(self.trend[0]) # 收盤數據
            for i in range(t+1):
                block.append(self.trend[i]) # 收盤數據
        else:
            block = self.trend[d : t + 1]   # 收盤數據
                
            
        res = []
        for i in range(window_size - 1):
            res.append((block[i + 1] - block[i])/(block[i]+0.0001)) #每步收益, 0.0001為設定的一個常數避免分母為零
        # res = []
            
        # if self.hold_num > 0:
        #     res.append(1)
        # else:
        #     res.append(0)
            
        # res.append((self.df['close'][t] - self.df['ma21'][t]) / self.df['ma21'][t])
        # res.append((self.df['close'][t] - self.df['ma13'][t]) / self.df['ma13'][t])
        # res.append((self.df['close'][t] - self.df['ma5'][t]) / self.df['ma5'][t])
        # res.append((self.df['vol'][t] - self.df['ma_v_21'][t]) / self.df['ma_v_21'][t])
        return np.array(res) #作為狀態編碼
    
    def buy_stock(self):       
        # 買入股票
        self.buy_num = self.hold_money // (self.trend[self.t] * 1000) # 買入股數
        tmp_money = self.trend[self.t] * self.buy_num * 1000 # 估計的買入總價
        service_charge = tmp_money * self.buy_rate # 計算手續費

        # 如果手續費不夠，就少買1張
        if service_charge + tmp_money > self.hold_money:
            self.buy_num = self.buy_num - 1000
        tmp_money = self.trend[self.t] * self.buy_num * 1000
        self.hold_num += self.buy_num
        self.stock_value += self.trend[self.t] * self.buy_num * 1000
        self.hold_money = self.hold_money - self.trend[self.t] * self.buy_num * 1000 - service_charge
        self.states_buy.append(self.t)
    
    def sell_stock(self, sell_num):
        tmp_money = sell_num * self.trend[self.t] * 1000
        service_charge = tmp_money * self.sell_rate
        stamp_duty = self.stamp_duty * tmp_money
        self.hold_money = self.hold_money + tmp_money - service_charge - stamp_duty
        self.hold_num = 0
        self.stock_value = 0
        self.states_sell.append(self.t)
        
    def trick(self):
        if self.df[self.feature_label][self.t] >= self.df['ma21'][self.t]:
            return True
        else:
            return False
    
    def step(self, action, show_log=False, my_trick=False):
        '''
        action == 1 買入
        action == 2 賣出
        action == 0 不動
        '''
        
        if action == 1 and self.hold_money >= (self.trend[self.t]*1000 + \
            self.trend[self.t]*1000*self.buy_rate) and self.t < (len(self.trend) - self.half_window):
            buy_ = True
            if my_trick and not self.trick(): 
                
            # 如果使用自己的觸發器並不能出發買入條件，就不買
                buy_ = False
            if buy_ : 
                self.buy_stock()
                if show_log:
                    print('day:%d, buy price:%f, buy num:%d, hold num:%d, hold money:%.3f'% \
                          (self.t, self.trend[self.t], self.buy_num, self.hold_num, self.hold_money))
        
        elif action == 2 and self.hold_num > 0:
            # 賣出股票         
            self.sell_stock(self.hold_num)
            if show_log:
                print(
                    'day:%d, sell price:%f, total balance %f,'
                    % (self.t, self.trend[self.t], self.hold_money)
                )
        else:
            if my_trick and self.hold_num>0 and not self.trick():
                self.sell_stock(self.hold_num)
                if show_log:
                    print(
                        'day:%d, sell price:%f, total balance %f,'
                        % (self.t, self.trend[self.t], self.hold_money)
                    )
                    
        self.stock_value = self.trend[self.t] * self.hold_num * 1000  # 持有股票總市值
        self.maket_value = self.stock_value + self.hold_money  # 總市值（加上現金）
        self.total_profit = self.maket_value - self.init_money # 總盈利
        self.stock_value_list.append(self.stock_value) # 持有股票總市值歷史紀錄
        self.maket_value_list.append(self.maket_value) # 總市值歷史紀錄（加上現金）
        
        
        # self.reward = (self.maket_value - self.last_value) / self.last_value
        reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t]
           
        reward = reward - 0.005
        
        if np.abs(reward)<=0.015:
            self.reward = reward * 0.2
        elif np.abs(reward)<=0.03:
            self.reward = reward * 0.7
        elif np.abs(reward)>=0.05:
            if reward < 0 :
                self.reward = (reward+0.05) * 0.1 - 0.05
            else:
                self.reward = (reward-0.05) * 0.1 + 0.05
        
        # reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t]
        if self.hold_num > 0 or action == 2:                                
            self.reward = reward    
            if action == 2:
                self.reward = -self.reward
        else:
            self.reward = -self.reward * 0.1
            # self.reward = 0
        
        self.last_value = self.maket_value
        self.profit_rate_account.append((self.maket_value - self.init_money) / self.init_money)
        self.profit_rate_stock.append((self.trend[self.t] - self.trend[0]) / self.trend[0])
     
        done = False
        self.t = self.t + 1
        ## 不懂為何最後十天不測
#         if self.t == len(self.trend) - 10:
#             done = True
        if self.t == len(self.trend)-1:
            done = True
            
        s_ = self.get_state(self.t)
        reward = self.reward
        
        return s_, reward, done
    
    def get_info(self):
        return self.states_sell, self.states_buy, self.profit_rate_account, self.profit_rate_stock, \
    self.maket_value_list,  self.stock_value_list
    
    def draw(self, save_name):
        # plot        
        states_sell, states_buy, profit_rate_account, profit_rate_stock, maket_value_list, \
        stock_value_list = self.get_info()
        
        
        ##### Plot K-line #####
        basic_plot_instance =plot.basic_plot(save_name, self.window_size, self.df, self.feature_label, self.feauture, self.trend, 
                                       states_sell, states_buy, profit_rate_account, profit_rate_stock, 
                                        maket_value_list, stock_value_list, self.total_profit)
        basic_plot_instance.signal()
        basic_plot_instance.profit_rate()
        basic_plot_instance.absolute_profit()
      
        plot.K_line().line(self.df)


