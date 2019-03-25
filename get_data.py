import pandas as pd 
import pickle as pl
import numpy as np
from tqdm import *
# import math.exp as exp
import math
from sklearn.model_selection import train_test_split
import xgboost as xgb 
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


def load_data(filename):
    with open('pl/'+filename+'.pkl', 'rb') as f:
        data = pl.load(f)
    return data  

def write_pkl(df,filename):
    with open('pl/'+filename+'.pkl', 'wb') as f:
        pl.dump(df,f,-1)
train = pd.DataFrame()
test = pd.DataFrame()
for i in tqdm(range(1,34)):
    data = load_data(str(i))[['wtid','ts','flag','var039','var031','var041','var049','var058','var068','var054','var023','var027']]
    # print(data)
    train = pd.concat([train,data[data.flag != 1]])

# del train

# for i in tqdm(range(1,34)):
    # data = load_data(str(i))[['wtid','ts','flag','var039','var031','var041','var049','var058','var068','var054']]
    test = pd.concat([test,data[data.flag == 1]])

write_pkl(train,'train0312')
write_pkl(test,'test0312')