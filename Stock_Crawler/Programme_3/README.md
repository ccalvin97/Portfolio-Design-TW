# Stock Crawler - Day based  

## Programme 3:  
overview_stock.py - Download stock list and filter files in order to have latest stock list files  

## Setup

```
$ pip install -r requirements.txt
```

## Function   

1. 下載最新目前台股還在流動的ID List    
2. 重新整理Stock List File - 將Programme 2 Output 依據目前台股還在流動的ID切分出來    
3. Data Cleaning Function  

### Example for Function 2  
  
update_stock_file - input: file list dir    
PS. 注意須將上面的 "個股list" file 放到 “base_dir” 下面  
