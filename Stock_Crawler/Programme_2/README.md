# Stock Crawler - Day based

## Programme 2:  
finance.py - financial season report  
combined.py - Combined financial data and stock price dataset(TBD)  


## Setup

```
$ pip install -r requirements.txt
```

### Command

爬當日

```
$ python finance.py
```

爬指定日期

```
$ python crawl.py YYYY S

Y:Year
S:season

e.g.

$ python crawl.py 2016 4
```

### Flag

```
    year_season assigned seasonal data (format: YYYY S), default is this season  
    -c crawl back n days for check data  
    -a crawl back from assigned day until 2015 2 season    
    -w choose data scource, 公開資訊觀測站 - 資產負債彙總表,綜合損益彙總表,營益分析彙總表 =1, TBD=2    
``` 


## 資料格式

- 每個檔案的檔名 `XXX_S.csv`，`XXX` 是年份, 'S'是哪一季度  
- 檔案中會有很多NaN, 原因是不同公司財報不同  
- 每個檔案已經根據公司編號ID匯集資所有資訊 - 產負債彙總表,綜合損益彙總表,營益分析彙總表  

