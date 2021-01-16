# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpl_finance as mpf
import matplotlib.ticker as ticker
import pdb
import seaborn as sns 
sns.set()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot



def plot_acc_loss(episode_reward, total_profit):
#     plt.rcParams.update({'font.size': 15})
    
    fig = plt.figure(figsize = (12,9))
    plt.style.use('dark_background')
    
    gs = GridSpec(2, 1) 
    ax1 = plt.subplot(gs[0, :]) 
    ax2 = plt.subplot(gs[1, :]) 
#     ax1 = fig.add_subplot(111)
    ax1.set_title('Training- Reward / Total Profit - Process 0', fontsize=15) 
    ax1.plot(range(len(episode_reward)), episode_reward, label="reward", color='skyblue')
    ax1.grid(True, linestyle=':', color='darkgrey', alpha=0.5)
    ax1.set_ylabel('Reward') 
    ax1.legend(loc = 'best') 
    
    ax2.plot(range(len(total_profit)), total_profit, label="total_profit", color='r')
    ax2.grid(True, linestyle=':', color='darkgrey', alpha=0.5)
    ax2.set_ylabel('Total Profit') 
    ax2.set_xlabel("steps")
    ax2.legend(loc = 'best') 
    
    
    plt.savefig('training_plot'+ '.png')
    plt.show()
    
    
    


class basic_plot():
    def __init__(self, save_name, window_size, df, feature_label, feauture, trend, states_sell, states_buy,
                 profit_rate_account, profit_rate_stock, maket_value_list, stock_value_list, total_profit):
        self.save_name = save_name
        self.window_size = window_size
        self.df = df
        self.feature_label = feature_label
        self.feauture = feauture
        self.trend = trend
        self.states_sell = states_sell
        self.states_buy = states_buy
        self.profit_rate_account = profit_rate_account
        self.profit_rate_stock = profit_rate_stock
        self.maket_value_list = maket_value_list
        self.stock_value_list = stock_value_list
        self.total_profit = total_profit
        self.invest = self.profit_rate_account[-1]
        self.total_gains = self.total_profit
        self.close = self.trend
        self.maket_value_list.append(self.maket_value_list[-1]) # 歷史總市值包含錢和股票
        self.stock_value_list.append(self.stock_value_list[-1]) # 歷史持有股票總市值
        self.profit_rate_account.append(self.profit_rate_account[-1]) # 賬號盈利
        self.profit_rate_stock.append((self.trend[-1] - self.trend[0]) / self.trend[0])
        
    def signal(self):
        ''' selling signal & buying signal
        '''
        #         dff=self.df.iloc[(self.window_size // 2):]
        times = self.df.iloc[:,0]
        times_seg = round(len(self.df.iloc[:,0])/10)    
        ticks=list(range((self.window_size // 2),len(times),times_seg))
        if ticks[-1]!=len(times)-1:
            ticks.append(len(times)-1)
        labels=[times[i] for i in ticks]
        
        close_up = [self.close[i]+1 for i in range(len(self.close))]
        
        self.feauture.append(self.feature_label)
        fig = plt.figure(figsize = (20,5))
        plt.style.use('dark_background')
        ax1 = fig.add_subplot(111)
        ax1.plot(self.df[self.feauture].iloc[:,0], color='r', lw=2., alpha=0.7)
        ax1.plot(self.df[self.feauture].iloc[:,1], color='skyblue', lw=2., alpha=0.7)
        ax1.plot(self.df[self.feauture].iloc[:,2], color='g', lw=2., alpha=0.7)
        ax1.plot(self.df[self.feauture].iloc[:,3], color='y', lw=2., alpha=0.7)
        ax1.plot(close_up, 'v', markersize=8, color='gold', label = 'selling signal', markevery = self.states_sell)
        ax1.plot(close_up, '^', markersize=8, color='tomato', label = 'buying signal', markevery = self.states_buy)        
        
        ax1.set(xlim=[0,len(times)-1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels, rotation=20, horizontalalignment='center')
        ax1.legend(loc = 'best')
        ax1.grid(True, linestyle=':', color='darkgrey', alpha=0.5)
        plt.title('Total Gains %f, Total Investment %f%%'%(self.total_gains, self.invest))
        plt.savefig(self.save_name + '_signal' + '.png')
        plt.close()
        
    
    def profit_rate(self):
        ''' profit_rate_account & profit_rate_stock
        '''
        dff=self.df.iloc[(self.window_size // 2):]
        times = dff.iloc[:,0]
        times_seg = round(len(dff.iloc[:,0])/10)    
        ticks=list(range((self.window_size // 2),len(times),times_seg))
        if ticks[-1]!=len(times)-1:
            ticks.append(len(times)-1)
        labels=[times[i] for i in ticks]
        
        fig = plt.figure(figsize = (20,5))
        plt.style.use('dark_background')
        ax2 = fig.add_subplot(111)
        ax2.plot(self.profit_rate_account, label='my account')
        ax2.plot(self.profit_rate_stock, label='stock')
        ax2.set(xlim=[0,len(times)-1])
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels, rotation=20, horizontalalignment='center')
        ax2.legend(loc = 'best')
        ax2.grid(True, linestyle=':', color='darkgrey', alpha=0.5)
        plt.title('Profit Rate')
        plt.savefig(self.save_name + '_profit_rate' + '.png')
        plt.close()
        
        
    def absolute_profit(self):
        ''' absolute_profit_account & absolute_profit_stock
        '''
        holding_value=[j-i for i,j in zip(self.stock_value_list, self.maket_value_list)]
        
        
        dff=self.df.iloc[(self.window_size // 2):]
        times = dff.iloc[:,0]
        times_seg = round(len(dff.iloc[:,0])/10)    
        ticks=list(range((self.window_size // 2),len(times),times_seg))
        if ticks[-1]!=len(times)-1:
            ticks.append(len(times)-1)
        labels=[times[i] for i in ticks]
        
        fig = plt.figure(figsize = (20,5))
        plt.style.use('dark_background')
        ax3 = fig.add_subplot(111)
        ax3.plot(self.stock_value_list, label='stock_value')
        ax3.plot(self.maket_value_list, label='maket_value')
        ax3.plot(holding_value, label='holding_value')
        ax3.set(xlim=[0,len(times)-1])
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(labels, rotation=20, horizontalalignment='center')
        ax3.legend(loc = 'best')
        ax3.grid(True, linestyle=':', color='darkgrey', alpha=0.5)
        plt.title('Absolute Value')
        plt.savefig(self.save_name  + '_absolute_profit' + '.png')
        plt.close()      
    
    
    
class K_line:
    
    def format_date(self, x, pos): 
        if x<0 or x>len(self.date_tickers)-1: 
            return '' 
        return self.date_tickers[int(x)] 

    def line(self, df):
        
        df['dates'] = np.arange(0, len(df))
        df=df.reset_index()
        df['trade_date2'] = df['trade_date'].copy()
#         df['trade_date'] = df['trade_date'].map(date2num)
        aaa = pd.to_datetime(arg=df['trade_date2'], format='%Y/%m/%d')
        self.date_tickers = (aaa).apply(lambda x:x.strftime('%Y%m%d')).values

        # 畫子圖
        figure = plt.figure(figsize=(12, 9))
        gs = GridSpec(3, 1) 
        ax5 = plt.subplot(gs[:2, :]) 
        ax6 = plt.subplot(gs[2, :]) 

        # 畫K線圖
        mpf.candlestick_ochl( ax=ax5, quotes=df[['dates', 'open', 'close', 'high', 'low']].values, width=0.7, \
                             colorup='r', colordown='g', alpha=0.7) 
        ax5.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_date)) 
        # 畫均線，均線可使用talib來計算
        for ma in ['5', '20', '30', '60', '120']: 
            df[ma]=df.close.rolling(int(ma)).mean() 
            ax5.plot(df['dates'], df[ma]) 
        ax5.legend(loc = 'best') 
        ax5.set_title('Candlestick', fontsize=15) 
        ax5.set_ylabel('Index') 
        ax5.grid(True, linestyle=':', color='darkgrey', alpha=0.5)


        # 畫成交量 
        ax6.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_date)) 
        df['up'] = df.apply(lambda row: 1 if row['close'] >= row['open'] else 0, axis=1) 
        ax6.bar(df.query('up == 1')['dates'], df.query('up == 1')['vol'], color='r', alpha=0.7) 
        ax6.bar(df.query('up == 0')['dates'], df.query('up == 0')['vol'], color='g', alpha=0.7) 
        ax6.set_ylabel('Volume')  # 成交量
        ax6.grid(True, linestyle=':', color='darkgrey', alpha=0.5)
        plt.savefig('Candlestick')
        plt.close()

