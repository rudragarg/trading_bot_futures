import requests
from config import *
import json
from tech_analysis import *
from data_collector import *

import os
from datetime import datetime

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDERS_URL = "{}/v2/orders".format(BASE_URL)
ACTIVITIES_URL = "{}/activities".format(ACCOUNT_URL)
HEADERS = {'APCA-API-KEY-ID': APCA_API_KEY, 'APCA-API-SECRET-KEY': APCA_SECRET_KEY}

def create_order(symbol, qty, side, type, time_in_force):
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }
    
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)

    return json.loads(r.content)


def get_orders(symbol):
    data = {
        "symbols": symbol
    }

    r = requests.get(ORDERS_URL, params=data, headers=HEADERS)
    return json.loads(r.content)


def cancel_orders():

    r = requests.delete(ORDERS_URL, headers=HEADERS)
    return json.loads(r.content)

def account_activity():

    r = requests.get(ACTIVITIES_URL, headers=HEADERS)
    return json.loads(r.content)


def is_bought(symbol):
    response = account_activity()

    for r in response:
        if (r.get("symbol") == symbol):
            if(r.get("side") == "buy"):
                return True
            if(r.get("side") == "sell"):
                return False
    
    return False




def log_data(symbol, numberOfShares, side, status):
    new_row = {
        "Symbol" : symbol,
        "Date": date.today(),
        "NumShares": numberOfShares,
        "Side": side,
        "Status": status
    }
    if not os.path.exists('trades/logs/{}_trades.csv'.format(symbol)):
        df = pd.DataFrame(columns = ['Symbol', 'Date', 'NumShares', 'Side', 'Status'])
    else:
        df = pd.read_csv('trades/logs/{}_trades.csv'.format(symbol))
    
    df = df.append(new_row, ignore_index=True)
    df.to_csv('trades/logs/{}_trades.csv'.format(symbol), index=False)
    
        

def trading_bot():

    '''
    Daily Agenda:

    update data every day during the night
    update news every day
    go through every updated stock and see if a buy decision is needed
    call the API if action needed
    log date

    '''
    print("UPDATING PRICE INFO")
    #updatedStocks = update_price_data()
    print("UPDATING NEWS INFO")

    #TESTING PURPOSES
    holdings = open('data/qqq.csv').readlines()

    symbols = [holding.split(',')[2].strip() for holding in holdings][1:]
    
    updatedStocks = symbols

    #update_news()
    print("GOING THROUGH STOCKS")

    # holdings = open('data/qqq.csv').readlines()
    # symbols = [holding.split(',')[2].strip() for holding in holdings][1:]

    numberOfShares = 5
    for symbol in updatedStocks:
        side = buy_sell_today(symbol)
        bought = is_bought(symbol) 
        if(side != "pass"):
            if((bought and side == "sell") or (not bought and side == "buy")):
                response = create_order(symbol, numberOfShares, side, "market", "opg")
                if (response.get("message") == "insufficient buying power"):
                    print("insufficient buying power to buy/sell {}".format(symbol))
                    status = "rejected"
                else:
                    print("{} {} shares of {}".format(side, numberOfShares, symbol))
                    status = "accepted"
                
                log_data(symbol, numberOfShares, side, status)
            
            


#print(is_bought("AMZN"))

print("LOG START")
print(datetime.today().strftime('%Y-%m-%d'))
print("==========")

trading_bot()

print("LOG END")
print("==========")

#cancel_orders()

#print(json.dumps(account_activity(), indent=4, sort_keys=True))
#print(is_bought("FB"))
#print(json.dumps(get_orders("TSLA"), indent=4, sort_keys=True))
# df = pd.read_csv("trades/logs/ADI_trades.csv")
# print(df)

# update_price_data()
#get_historic_data("2020-01-01", "2021-01-09")
