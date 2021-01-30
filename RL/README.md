# A3C Deep Reinforcement Learning - (Asynchronous Advantage Actor-Critic)  

Trading Robot with reinforcement learning    

<div align="center">
<img src="https://github.com/ccalvin97/Portfolio-Design-TW/blob/main/RL/graph/Full_UML.png" width="700" alt= "Reinforcement Learning" />
</div>


## References  

1. _Playing Atari with Deep Reinforcement Learning_, Mnih et al., 2013  
2. _Human-level control through deep reinforcement learning_, Mnih et al., 2015  
3. _Deep Reinforcement Learning with Double Q-learning_, van Hasselt et al., 2015  
4. _Continuous control with deep reinforcement learning_, Lillicrap et al., 2015  
5. _Asynchronous Methods for Deep Reinforcement Learning_, Mnih et al., 2016  
6. _Continuous Deep Q-Learning with Model-based Acceleration_, Gu et al., 2016  
7. _Learning Tetris Using the Noisy Cross-Entropy Method_, Szita et al., 2006  
8. _Deep Reinforcement Learning (MLSS lecture notes)_, Schulman, 2016  
9. _Dueling Network Architectures for Deep Reinforcement Learning_, Wang et al., 2016  
10. _Reinforcement learning: An introduction_, Sutton and Barto, 2011  
11. _Proximal Policy Optimization Algorithms_, Schulman et al., 2017  
12. [Paper](https://arxiv.org/abs/2002.11523) - Using Reinforcement Learning in the Algorithmic Trading Problem, Evgeny Ponomarev, 2020

# What is included?  
This repo includes implementation:    
Deep Recurrent Q Network (DRQN)    
Asynchronous Advantage Actor Critic (A3C)   
Proximal Policy Optimization (PPO)   
  
  
All implementations are tested on Mac OS & Google Colab Cloud environment.    
Please be aware that all of the codes are still in the development stage.    

# Data  
****The testing data are all real TW stock data from web crawling system.****   

**Stock - 台積電 TSMC(2330)**  
Company Description:  
台灣積體電路製造（英語：Taiwan Semiconductor Manufacturing），通稱台積電、台積或TSMC，與旗下公司合稱時則稱作台積電集團，是一家晶圓代工半導體製造廠，總部位於臺灣新竹科學園區，主要廠房則分布於新竹市、臺中、臺南等臺灣各地的科學園區。  

2019年7月，台積電在美國《財富》雜誌評選「全球最大500家公司」排行榜中，依營收規模名列全球第363名，依獲利規模名列全球第39名。2019年8月，台積電在PwC發表的「全球頂尖100家公司」排行榜中，依公司市場價值名列全球第37名。截至2019年12月，台積電為台灣證券交易所發行量加權股價指數最大成份股，約占台股大盤總市值比重 23%。同時台積電多數股份皆為海外基金投資持有。台積電市值全球十大，半導體晶圓代工全球排名第一。   


**Data Example**   
| Time    	| vol           	| Trading Price	| open       	| high      | low	      | close    	|Gross Spread	|Trading Count|    
| ---       | ---             |  ---          | ---         |  ---      | ---       |  ---      | ---         |   ---       |  
|109/10/22	|25,285,547	      |11435937453	  |450	        |455        |449.5	    |455	      |2         	  |10,354       |    

**Data from 1999 - 2020**     

# Results  

     
<div align="center">
<img src="https://github.com/ccalvin97/Portfolio-Design-TW/blob/main/RL/graph/Candlestick.png" width="800" alt= "Candlestick" />
</div>

    
<div align="center">
<img src="https://github.com/ccalvin97/Portfolio-Design-TW/blob/main/RL/graph/training_plot.png" width="800" alt= "Training Result" />
</div>

     
<div align="center">
<img src="https://github.com/ccalvin97/Portfolio-Design-TW/blob/main/RL/graph/plt__signal.png" width="800" alt= "Sell/Buy Signal" />
</div>

      
<div align="center">
<img src="https://github.com/ccalvin97/Portfolio-Design-TW/blob/main/RL/graph/plt__absolute_profit.png" width="800" alt= "Absolute Profit" />
</div>

## Contributing  

Programme is created by Calvin He `<kuancalvin2016@gmail.com>`.  
