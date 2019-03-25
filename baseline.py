import pandas as pd 
import numpy as np
from tqdm import *
import matplotlib.pyplot as plt
import pickle as pl 
import os
import sys
from datetime import *
import time
import math

def load_data(filename):
    with open('pl/'+filename+'.pkl', 'rb') as f:
        data = pl.load(f)
    return data  

def write_pkl(df,filename):
    with open('pl/'+filename+'.pkl', 'wb') as f:
        pl.dump(df,f,-1)

def get_data(i):
    start = time.clock()
    data = pd.read_csv('data/'+str(i).zfill(3)+'/201807.csv',parse_dates=[0])
    res = pd.read_csv('data/template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
    
    res = res[res['wtid']==i]
    res['flag'] = 1
    data = res.merge(data, on=['wtid','ts'],how = 'outer')
    data = data.sort_values(['wtid','ts']).reset_index(drop = True)
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    return data

def transform(x):
    '''
    从字段分组可以发现，实际上如var004数值并非真正连续的值
    在同一个机器上，其数值可以认为是散点
    这一点将字段排序后就会发现
    '''
    global value_list
    if x in value_list:
        return x 
    else:
        return value_list[[abs(k-x)for k in value_list].index(min([abs(k-x)for k in value_list]))]

'''
本次项目使用纯插值的方法做的，无模型预测，仅提供参考思路
从评分方式分析：
math.exp(-100*abs(label - pred)/max(abs(label),10**(-15)))
当这个分数越高时，最后分数越高
所以对于数值越小抖动越大的值是处理的关键字段
'''

df = pd.DataFrame()
for i in tqdm(range(1,34)):
    '''使用pickle存储数据，加快读取速度'''
    data = get_data(i)
    write_pkl(data,str(i))
    # data = load_data(str(i))

    '''以时间段为分段，以一秒钟为一个分段'''
    data['split_t'] = data.ts.apply(lambda x:str(x).split(' ')[0].split('-')[2]+str(x).split(' ')[1].split(':')[0]+str(x).split(' ')[1].split(':')[1])
    var5 = data.groupby('split_t')

    fe = [ i for i in data.columns if 'var' in i ]

    for j in ['var004','var042','var046']:
            '''通过绘图发现var004,var042,var046字段图形非常相似，同时与var034有很强的线性关系'''
            con = (data[j].isnull()==True)&(data.var034.isnull()==False)
            group_034 = data.groupby('var034')[j].agg('median')
            data[j][con] = group_034.loc[data['var034'][con].values].values

    for j in fe:


        if j in ['var015','var061']:
            '''var015,var061均在其平均值上抖动，因此取其平均值作为填充'''
            split_nan = set(data.split_t[data[j].isnull()==True].values)
            var = {}
            for s in split_nan:
                dd = var5.get_group(s)[j]
                var[s] = dd.agg('mean')
            data[j][data[j].isnull()==True] = data['split_t'][data[j].isnull()==True].apply(lambda x:var[x])
            data[j] = data[j].fillna(data[j].mean())

        elif j in ['var016','var025','var026','var032','var047','var053','var066','var050']:
            '''字段的值抖动不定且会集中在某个值，使用众数插值'''
            split_nan = set(data.split_t[data[j].isnull()==True].values)
            var = {}
            l = []
            for s in split_nan:
                dd = var5.get_group(s)[j]
                if dd.isnull().all() == False:
                    l.append(s)
                
            data[j][(data['split_t'].isin(l))==True] = data[j][(data['split_t'].isin(l))==True].interpolate('nearest')
            data[j] = data[j].fillna(list(data[j].value_counts().head(1).index)[0])

        elif j in ['var060','var044','var031','var030','var029','var027','var018','var017','var012','var009','var003','var002','var033','var028','var021','var061','var024','var022','var006','var007','var011','var014','var001','var005','var035','var036','var037','var038','var040','var045','var051','var052','var055','var056','var057','var062','var067']:
            '''线性插值'''
            split_nan = set(data.split_t[data[j].isnull()==True].values)
            var = {}
            for s in split_nan:
                dd = var5.get_group(s)[j]
                var[s] = dd.agg('median')
            data[j][data[j].isnull()==True] = data['split_t'][data[j].isnull()==True].apply(lambda x:var[x])
            data[j] = data[j].interpolate('linear')
            value_list = list(set(data[j].values))
            data[j] = data[j].apply(transform)

        elif j in ['var004','var068','var064','var063','var034','var010','var008','var046','var043','var015','var065','var059','var042']:
            '''紧邻插值'''
            split_nan = set(data.split_t[data[j].isnull()==True].values)
            var = {}
            for s in split_nan:
                dd = var5.get_group(s)[j]
                var[s] = dd.agg('median')
            data[j][data[j].isnull()==True] = data['split_t'][data[j].isnull()==True].apply(lambda x:var[x])
            
            data[j] = data[j].interpolate('nearest')
        elif j in ['var058','var041','var049']:
            split_nan = set(data.split_t[data[j].isnull()==True].values)
            var = {}
            for s in split_nan:
                dd = var5.get_group(s)[j]
                var[s] = dd.agg('median')
            data[j][data[j].isnull()==True] = data['split_t'][data[j].isnull()==True].apply(lambda x:var[x])
            data[j] = data[j].fillna(list(data[j].value_counts().head(1).index)[0])
            

     
        if j in ['var041','var039']:
            data[j] = data[j].fillna(list(data[j].value_counts().head(1).index)[0])

        elif j in ['var022','var024','var061']:
            data[j] = data[j].interpolate('linear')
            value_list = list(set(data[j].values))
            data[j] = data[j].apply(transform)
        else:
            data[j] = data[j].interpolate('nearest')


    df = pd.concat([df,data[data.flag==1]],axis = 0)
sub = df[df.flag==1].copy().reset_index(drop = True)
del sub['flag']
del sub['split_t']

sub['var016'] = sub['var016'].astype(int)
sub['var020'] = sub['var020'].astype(int)
sub['var047'] = sub['var047'].astype(int)
sub['var053'] = sub['var053'].astype(int)
sub['var066'] = sub['var066'].astype(int)
res = pd.read_csv('data/template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
DF = res.merge(sub, on=['wtid','ts'],how = 'outer')
DF.to_csv('submit/sub_DCIC0309_4.csv',index=False,float_format='%.3f')