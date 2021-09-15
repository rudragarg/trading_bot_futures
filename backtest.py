import btalib
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import yfinance as yf
import datetime as dt
from datetime import timedelta
from rtstock.stock import Stock
import statistics

import nltk
#nltk.download('vader_lexicon')
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
finwiz_url = 'https://finviz.com/quote.ashx?t='


def get_data(symbol):
    return yf.download(symbol,'2020-12-01','2020-12-31', interval = "1m")

# ===========BUY AND HOLD ============= #
def get_entry_price(data):
    return data.iloc[0,3]

def get_exit_profit(data):
    return data.iloc[-1, 3] - data.iloc[0,3]


#Implementing Buy Sell Strat
def buy_signal(symbol, data):
    
    #add to buy column to data
    
    #use model to get prediction,
    
    '''
    STRAT:
    Buy:
        call model, if model predicts 1, buy
    
    Not Buy:
        model predicts 0
    '''
#FOR BACKTESTING
def sell_signal(symbol, data):
    
    #get buy signal, look into future, find signals
    
    '''
    STRAT:
    
    Sell:
        Price increases by 1% (or X%)
        Price decreases by .5% (or Y%)
    
    '''


def plot(data, symbol):
    fig = plt.figure(figsize=(8, 6))
    plt.title('Buy and Sell {}'.format(symbol))
    
    if(not data["Buy"].isnull().all()):
        plt.scatter(data.index, data["Buy"], color = 'green', marker="^", alpha=1)
    if(not data["Sell"].isnull().all()):
        plt.scatter(data.index, data["Sell"], color = 'red', marker="v", alpha=1)

    fig.savefig('charts/{}_chart.png'.format(symbol))
    plt.close(fig)
    
def get_total_profit(data):
    
    profit = 0
    i = 0 
    buyIndex = -1
    sellIndex = -1

    buyCol = data.columns.get_loc("Buy")
    closeCol = data.columns.get_loc("Close")

    while i < len(data):
        if(not pd.isnull(data.iloc[i,buyCol]) and buyIndex == -1): 
            buyIndex = i
            #print("BUYING at", data.iloc[i,buyCol])
        
        if(not pd.isnull(data.iloc[i,buyCol+1]) and buyIndex != -1):
            sellIndex = i
            #print("SELLING at", data.iloc[i,buyCol+1])
            #print("Profit:", data.iloc[sellIndex,buyCol+1] - data.iloc[buyIndex,buyCol])
            # print("=============")
            profit += (data.iloc[sellIndex,buyCol+1] - data.iloc[buyIndex,buyCol])
            
            buyIndex = -1
            sellIndex = -1
        if((i+1) == len(data) and buyIndex != -1):
            lastRow = -1
            while (pd.isnull(data.iloc[lastRow, closeCol])):
                lastRow -= 1


            profit += data.iloc[lastRow, closeCol] - data.iloc[buyIndex, buyCol]
        i+=1
    return profit


def run():
    holdings = open('data/qqq.csv').readlines()

    symbols = [holding.split(',')[2].strip() for holding in holdings][1:]
    
    totalProfit = 0
    totalEntry = 0
    totalExitProfit = 0


    countWorked = 0
    for symbol in symbols:
     
        no_data = False
        try:
            data = get_data(symbol)
        except:
            no_data = True
        if (no_data):
            continue
        
        data = buy_sell(symbol, data)

        plot(data, symbol)

        entry = get_entry_price(data)
        totalEntry += entry

        exitProfit = get_exit_profit(data)
        totalExitProfit += exitProfit

        profit = get_total_profit(data)
        
        totalProfit += profit

        change = profit/entry
        percentChange = "{:.0%}".format(change)
        
        comparedChange = exitProfit/entry
        comparedChangepercent = "{:.0%}".format(comparedChange)

        print("{} or {} per share of {}...compared to {}".format(profit, percentChange, symbol, comparedChangepercent))
        if(change >= comparedChange):
            print("STRAT WORKED")
            countWorked += 1

        

    totalChange = totalProfit/totalEntry
    percentChange = "{:.0%}".format(totalChange)
    print("====TOTAL PROFIT:", totalProfit)
    print("====TOTAL ENTRY:", totalEntry)
    print("====TOTAL CHANGE:", percentChange)
    print("====NUM TIMES WORK: {}/{}".format(countWorked, len(symbols)))

    woEMAProfit = totalExitProfit/totalEntry
    woEMAProfitPercent = "{:.0%}".format(woEMAProfit)

    print("========Without BBANDS========")
    print("=====TOTAL EXIT PROFIT:", totalExitProfit)
    print("=====TOTAL EXIT PROFIT:", woEMAProfitPercent)

    
    #return totalProfit, totalChange

run()

# temp = yf.download("AAPL", "2020-12-31", "2021-01-04")
# print(temp)



#temp = get_news_sentiment()

# symbol = "MDLZ"
# data = get_data(symbol)

# print(type(data.index[0]))
# print(type(data.index[0].to_pydatetime().date()))


# success = []
# for sellPeriod in range(5, 20):
#     for sellDev in [.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0]:
#         totalProfit, totalChange = run(sellPeriod, sellDev)
#         if (totalChange >= .4):
#             success.append((sellPeriod, sellDev, totalProfit, totalChange))

# print(success)

# symbol = "MDLZ"
# data = get_data(symbol)

# data = get_BBands(data)
# data = get_EMA(data)
# data = get_RSI(data)



# data = buy_sell(data)
# print(data)
# profit = get_total_profit(data)
# print(profit)

# plot(data, symbol)

    
