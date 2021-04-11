import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def user_clean(user_id):
    text = 'Запись пользователя № -'
    error = '#error'
    if text in user_id:
        return user_id.replace(text, '').strip()
    elif error in user_id:
        return user_id.replace(error, '').strip()
    else:
        return user_id


log = pd.read_csv('log.csv', header=None)
log.columns = ['user_id', 'time', 'bet', 'win']
log = log[log['user_id'] != '#error']
log.user_id = log.user_id.str.split(' - ').apply(lambda x: x[1])
log['time'] = log['time'].str.replace('[', '')





# print(log.dropna( axis=1,inplace=True))
# print(log[log['time'].isna()]['time'].shape)]
# print(log.dropna(axis=0, subset=['user_id', 'time', 'bet', 'win']))
# print(log.dropna(axis=0, subset=['user_id', 'time', 'bet', 'win']))
# log = log[log['time'].isna()]

# log = log.drop_duplicates(subset=['user_id', 'time'])

# log = log.dropna()
# log['time'] =  log['time'].str.replace('[', '')
# log['time'] = pd.to_datetime(log['time'])



# log['time'] = pd.to_datetime(log['time'])
# log['minute'] = log['time'].dt.minute

# log['time'] = pd.to_datetime(log['time'])
# log['month'] = log['time'].dt.month


# log['time'] = pd.to_datetime(log['time'])
# log['day_of_week'] = log['time'].dt.day_of_week
# print(log[(log['day_of_week'] == 0) | (log['day_of_week'] == 0)] )

# def time_of_day(x):
#     x = int(x)
#     if x <= 5:
#         return 'ночь'
#     elif 6 <= x <= 17:
#         return 'утро'
#     elif 12 <= x <= 17:
#         return 'день'
#     else:
#         return 'вечер'
# log = log.dropna(axis=0, subset= ['user_id', 'time', 'bet', 'win'])
# log['time'] = pd.to_datetime(log['time'])
# log['hour'] =  log['time'].dt.hour
# log['hour_word'] = log['hour'].apply(time_of_day)
# print(log['hour_word'].value_counts())


# log['bet'].fillna(0, inplace=True)

log['time'] = pd.to_datetime(log['time'])
def fillna_win(row):
    win_nan = row.isnull()
    if win_nan['win'] and win_nan['bet']:
        row['bet'] = 0
        row['win'] = 0
    elif win_nan['win'] and not win_nan['bet']:
        row['win'] = -1 * row['bet']
    return row
new_win = log.apply(lambda row: fillna_win(row), axis=1)

def profit(row):
    if row['win'] < 0:
        row['net'] = row['win']
    else:
        row['net'] =  row['win'] - row['bet']
    return row
new_win['net'] = ''
log = new_win.apply(lambda row: profit(row), axis=1)


users = pd.read_csv('users.csv', encoding='KOI8-R', sep='\t')
users.columns = ['user_id', 'email' , 'geo']
users['user_id'] = users['user_id'].apply(lambda x : str(x).lower())


users_log  = pd.merge(left=log, right=users, on=['user_id'])



# group = users_log[users_log.bet==0].groupby('user_id').bet.count()
# group_not_null = users_log[users_log.bet>0].groupby('user_id').bet.count()#
# joined=pd.merge(group, group_not_null, on=['user_id'])
# joined = joined[joined['bet_y']>0]
# print(joined['bet_x'].sum()/len(joined))

# time_min = users_log[users_log.bet==0].groupby('user_id').time.min()
# time_min_not_null = users_log[users_log.bet>0].groupby('user_id').time.min()
# joined = pd.merge(time_min, time_min_not_null, on='user_id')
# joined['delta']=joined['time_y']-joined['time_x']
# print(int((joined['delta'].sum().days)/joined.delta.count()))


# time_min = users_log.groupby('geo')['win'].sum()
# time_min.sort_values()

bet_min = users_log.groupby('geo')['bet'].min()
bet_max = users_log.groupby('geo')['bet'].max()
joined = pd.merge(bet_min, bet_max, on='user_id')
print(bet_min.max())
print(bet_min.min())
print(bet_min.max()/bet_min.min())


exit()

# users)
exit()