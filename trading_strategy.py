import gc
import matplotlib.pyplot as plt
import numpy as np


def strategy(pred,
             prob,
             close,
             x_len=120,
             plt_len=500,
             limit=10,
             lots=10,
             stop=0.005,
             take=0.01,
             thresh=0.75,
             comsa=0.001,
             verbose=1):
    
    
    def buy(lots, cost, type_):
        nonlocal score, avg, pos, sum_com
        score -= lots * cost
        if type_ in ['stop', 'take']:
            avg = None
        elif pos < 0:
            if (pos + lots) < 0:
                avg = avg
            elif (pos + lots) == 0:
                avg = None
            else:
                avg = cost
        elif pos == 0:
            avg = cost
        else:
            avg = (avg * pos + cost * lots) / (pos + lots)   
        pos += lots
        if verbose:
            print(f'Bought: {lots}, price: {cost}, type: {type_}. Current pos: {pos}, avg: {avg}')
        sum_com += lots * cost * comsa


    def sell(lots, cost, type_):
        nonlocal score, avg, pos, sum_com
        score += lots * cost
        if type_ in ['stop', 'take']:
            avg = None
        elif pos > 0:
            if (pos - lots) > 0:
                avg = avg
            elif (pos - lots) == 0:
                avg = None
            else:
                avg = cost
        elif pos == 0:
            avg = cost
        else:
            avg = (avg * pos - cost * lots) / (pos - lots)   
        pos -= lots
        if verbose:
            print(f'Sold: {lots}, price: {cost}, type: {type_}. Current pos: {pos}, avg: {avg}')
        sum_com += lots * cost * comsa
        
    profit = 0
    score = 0
    avg = None
    pos = 0
    max_pos = 0
    sum_com = 0 
    profit_list = []
    
    for i in range(len(pred)):
        profit_list.append(score + close[i+x_len] * pos - sum_com)
        
        # check stop and take
        if avg is not None:
            if pos < 0:
                if (close[i] / avg - 1) > stop:
                    buy(lots=-pos, cost=close[i+x_len], type_='stop')
                elif (close[i] / avg - 1) < -take:
                    buy(lots=-pos, cost=close[i+x_len], type_='take')
            elif pos > 0:
                if (close[i] / avg - 1) < -stop:
                    sell(lots=pos, cost=close[i+x_len], type_='stop')
                elif (close[i] / avg - 1) > take:
                    sell(lots=pos, cost=close[i+x_len], type_='take')
                    
        # check signal
        if prob[i] >= thresh:
            if pred[i] == -1:
                if ((avg is None) or (close[i] < avg)) and (pos < limit):
                    buy(lots=lots, cost=close[i+x_len], type_='std')
            elif pred[i] == 1:
                if ((avg is None) or (close[i] > avg)) and (pos > -limit):
                    sell(lots=lots, cost=close[i+x_len], type_='std')
                    
        if abs(pos) > max_pos:
            max_pos = abs(pos)       
        
        
    gc.collect()
    plt.figure(figsize=(15,8))
    s = np.random.randint(0, len(close)-plt_len)
    plt.subplot(121)
    plt.plot(close-close[0], 'black', label='close')
    plt.plot(profit_list, 'blue', label='profit')
    plt.legend()
    plt.subplot(122)
    plt.plot(close[s:s+plt_len], 'black')
    plt.plot(np.where((pred[s:s+plt_len-x_len]==1)&(prob[s:s+plt_len-x_len]>=thresh))[0]+x_len, 
             close[s:s+plt_len][np.where((pred[s:s+plt_len-x_len]==1)&(prob[s:s+plt_len-x_len]>=thresh))[0]+x_len], 'r.')
    plt.plot(np.where((pred[s:s+plt_len-x_len]==-1)&(prob[s:s+plt_len-x_len]>=thresh))[0]+x_len, 
             close[s:s+plt_len][np.where((pred[s:s+plt_len-x_len]==-1)&(prob[s:s+plt_len-x_len]>=thresh))[0]+x_len], 'b.')
    plt.show()
    return profit_list