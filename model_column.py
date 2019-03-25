import pandas as pd 
import pickle as pl
import numpy as np
# import math.exp as exp
import math
from sklearn.model_selection import train_test_split
import xgboost as xgb 
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error

def write_pkl(df,filename):
    with open('pl/'+filename+'.pkl', 'wb') as f:
        pl.dump(df,f,-1)

def load_data(filename):
    with open('pl/'+filename+'.pkl', 'rb') as f:
        data = pl.load(f)
    return data  

def check_column(data):
    global max_x
    print(data)
    max_x = data.x.apply(lambda x:abs(x)).max()
    
    return exp(-((100*abs(data.x-data.x_pre)/(max(max_x,1e-15)))))


a = ['var004', 'var015', 'var023', 'var028', 'var041', 'var042', 'var043', 'var046', 'var050', 'var053', 'var060', 'var067', 'var068']

def get_df(column,X_test,x_pre,x1_pre):
        df = pd.DataFrame(columns=['x','x_pre'])
        df_pre = pd.DataFrame(columns=['x','x_pre'])
        df.x = X_test[column]
        df.x_pre = x_pre[column]
        df_pre.x = X_test[column]
        df_pre.x_pre = x1_pre[column]
        return df,df_pre

# data = load_data('1')
# data.flag = 0
# print(data)

# split_a = []

# X_train, X_test, y_train, y_test = train_test_split(data,data.var001,test_size=0.33, random_state=666)


# data = pd.read_csv('big_df.csv')
# write_pkl(data,'big_df')

# train = load_data('train0312')
# test = load_data('test0312')
# data = pd.concat([train,test])
# # # print(test)
# # # print(train)

# # # con34 = (data['var034'].isnull()==True)&(data.var004.isnull()==False)
# # # weight1 = data.var004[data['var034'].isnull()==False].mean()/data['var034'].mean()
# # # weight2 = data.var042[data['var034'].isnull()==False].mean()/data['var034'].mean()
# # # weight3 = data.var046[data['var034'].isnull()==False].mean()/data['var034'].mean()
# # # data['var034'][con34] = (data['var004'][con34]/weight1+\
# # # data['var042'][con34]/weight2+\
# # # data['var046'][con34]/weight3)/3
# # # print(data[['var004','var042','var046','var034','flag']][con34])
# # # ++++str(x).split(' ')[1].split(':')[2].split('.')[1])
# # # train['split_day'] = train.ts.apply(lambda x:int(str(x).split(' ')[0].split('-')[2]))
# # # train['split_hour'] = train.ts.apply(lambda x:int(str(x).split(' ')[1].split(':')[0]))
# # # train['split_min'] = train.ts.apply(lambda x:int(str(x).split(' ')[1].split(':')[1]))
# # # train['split_sce'] = train.ts.apply(lambda x:int(str(x).split(' ')[1].split(':')[2].split('.')[0]))
# data['split_day'] = data.ts.apply(lambda x:int(str(x).split(' ')[0].split('-')[2]))
# data['split_hour'] = data.ts.apply(lambda x:int(str(x).split(' ')[1].split(':')[0]))
# data['split_min'] = data.ts.apply(lambda x:int(str(x).split(' ')[1].split(':')[1]))
# data['split_sce'] = data.ts.apply(lambda x:int(str(x).split(' ')[1].split(':')[2].split('.')[0]))
# # # print(data['split_sce'])

# # def s(x):
# #     try:
# #         return int(str(x).split(' ')[1].split(':')[2].split('.')[1])
        
# #     except:
# #         return 0

# # data['split_sce'] = data.ts.apply(s)
# # print(a)
# write_pkl(data,'data0312')
data = load_data('data0312')

train = data[data['var027'].isnull()==False]
test = data[data['var027'].isnull()==True]


feature = ['wtid','split_day','split_hour','split_min','split_sce']
target = 'var027'

param = {
         'learning_rate': 0.9,
         "min_child_samples": 20,
         "boosting": "gbdt",
         # 'objective':'multiclass',
         'objective':'regression',
        #  'num_data_in_leaf': 28,
        #  'num_boost_round': 2000,
         # "feature_fraction": 0.6,
         "bagging_freq": 1,
         # "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         'seed': 2019,
         }

folds = KFold(n_splits=5, shuffle=True, random_state=2018)

oof_lgb = np.zeros(len(train[feature]))
predictions_lgb = np.zeros(len(test))
X_train = train[feature].values
y_train = train[target].values
X_test = test[feature].values

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    if fold_ == 0:
        print("fold n°{}".format(fold_+1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

        num_round = 300000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=300, early_stopping_rounds = 500)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        
        predictions_lgb = clf.predict(X_test, num_iteration=clf.best_iteration) 

print("CV score: {:<8.8f}".format(mean_absolute_error(oof_lgb, y_train)))

def transform(x):
    global j 
    global data
    # global x0 
    # global x1 
    global i
    global value_list
    if x in value_list:
        return x 
    else:
        return value_list[[abs(k-x)for k in value_list].index(min([abs(k-x)for k in value_list]))]


# # value_list = []
# # predictions_lgb = pd.Series(predictions_lgb).apply(transform)

test['var027'] = predictions_lgb
write_pkl(test,'test027_0311_1')

write_pkl(predictions_lgb,'pre027_0311_1')
# test = load_data('test023_0311_1')
# predictions_lgb = load_data('pre023_0311_1')
sub = test[['ts','wtid','flag','var027']]
print(sub.info())
# for i in range(1,34):
#     value_list = list(set(train.var023[train.wtid == i].values))
#     # print(value_list)
#     # print(sub.var023[sub.wtid==i].apply(transform))
#     sub.var023[sub.wtid==i] = sub.var023[sub.wtid==i].apply(transform)
# sub['var023'] = predictions_lgb
print(sub)
sub = pd.concat([sub,train[['ts','wtid','flag','var027']]])
sub = sub[sub.flag == 1]
print(sub.info())
del sub['flag']
res = pd.read_csv('data/template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
DF = res.merge(sub, on=['wtid','ts'],how = 'outer')
DF.to_csv('submit/pre0311_027_1.csv',index=False)