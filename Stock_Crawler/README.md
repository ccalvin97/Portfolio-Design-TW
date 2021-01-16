# Stock Crawler(TAIWAN) - Day based

## Programme 1:  
crawl.py - Taiwan Stock Price Crawler, date based stock crawler  
post_process.py - post process for the crawler, cleaning duplicated data  

## Programme 2:  
finance.py - financial season report  
combined.py - Combined financial data and stock price dataset(TBD).   
other.py - Other correlated data such as EU & USA data for input resources(TBD).  

## Programme 3:  
overview_stock.py - Download stock list and filter files in order to have latest stock list files  

目標網頁:  
台灣證券交易所 (http://www.twse.com.tw/)  
證券櫃檯買賣中心 (http://www.tpex.org.tw/)   
本國上市證券國際證券辨識號碼一覽表 (https://isin.twse.com.tw/isin/C_public.jsp?strMode=2)  
公開資訊觀測站 (https://mops.twse.com.tw/mops/web)  
     
### Directory Hierarchy
``` 
.  
│   Stock_Crawler  
│   ├── Programme_1  
│   │   ├── crawl.py  
│   │   ├── post_process.py  
│   │   ├── run.sh  
│   │   
│   ├── Programme_2  
│   │   ├── finance.py  
│   │   
│   ├── Programme_3  
│   │   ├── overview_stock.py  
│   │   
│   requirements.txt  
│   
```  
## Contributing

Programme is created by Calvin He `<kuancalvin2016@gmail.com>`.
