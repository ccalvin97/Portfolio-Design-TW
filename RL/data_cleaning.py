# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pdb
import re

class cleaning:
    def __init__(self, df):
        self.df = df
    
    '''
    df input data, type dataframe
    
    PS. Be aware of column name & order 
    
    df example:
Time	Trading Volume	Trading Price	Start Price	  Max Price	 Min Price	End Price	Gross Spread	Trading Count
109/07/10	60902108	21298923681	    352.5	      353	     345.5	     348.5	    3.5	            32,227
109/07/09	37410376	12912747970	    346	          347	     343	     345	    4	            20,205
109/07/08	33813218	11522915649	    337.5	      342.5      337.5	     341	    2.5	            16,673
    
    '''
    
    def TW_day_based_data(self):
        self.df['Gross Spread']=self.df['Gross Spread'].apply(lambda x: 0.015 if x == 'null0.0' else x)
        self.df['Gross Spread']=self.df['Gross Spread'].apply(lambda x: 0.015 if x == 'X0.0' else x)
        self.df['trade_date']=self.df['trade_date'].apply(lambda x: str(int(x.split('/')[0])+1911)+'/'+x.split('/')[1] \
                                              + '/' + x.split('/')[2] if len(x.split('/')[0])==3  else x )
        
        def function(x):  
            pattern=re.compile("[a-zA-Z,]")
            try:
                aa=float(re.sub(pattern,'',str(x))) 
            except:
                pdb.set_trace()            
            return aa
        
        
        for i in self.df.columns[1:]:
            
#             self.df[i]=self.df[i].apply(lambda x: float(str(x).replace(",", "")))
            self.df[i]=self.df[i].apply(function)

#             self.df[i]=self.df[i].apply(lambda x: float(x) )

        
        for i in self.df.columns[1:]:
            self.df[i]=self.df[i].fillna(method='ffill')
        
        # 因為我們的數據是時間倒反的
        self.df=self.df[::-1].reset_index(drop=True)
        return self.df