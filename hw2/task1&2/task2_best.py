import numpy as np
import pandas as pd
from collections import defaultdict

dataset = pd.read_csv('ntucsie-sdml2018-2-1/rating_train.csv', header=0)

# Data Preprocessing
users = dataset.userid.unique()
user_total_cnt = defaultdict(int)

for userid in users:
    data = dataset[dataset['userid'] == userid]
    user_total_cnt[userid] = data.shape[0]

# Rule-based - count frequency
matrix = dataset.as_matrix()
freq = defaultdict(lambda: defaultdict(int))
user_cnt = defaultdict(int)
user_freq_cnt = defaultdict(int)
user_foods = defaultdict(set)

for row in matrix:
    date, userid, foodid = row[0], row[1], row[2]
    user_cnt[userid] += 1
    if user_cnt[userid] >= user_total_cnt[userid] * 0.85:
        if foodid not in user_foods[userid]:
            user_foods[userid].add(foodid)
        user_freq_cnt[userid] += 1
        freq[userid][foodid] += 1

# Prediction
train = []
for userid in freq:
    for foodid in freq[userid]:
        train.append([userid, foodid, freq[userid][foodid]])
train = np.array(train)

df = pd.DataFrame()
df['userid'] = train[:,0]
df['foodid'] = train[:,1]
df['freq'] = train[:,2]

# Ouput file
with open('pred.csv', 'w') as file:
    file.write('userid,foodid\n')
    
    for userid in users:
        if len(user_foods[userid]) < 20:
            print(userid, len(user_foods[userid]))
        foods = df[df['userid'] == userid].sort_values('freq', ascending=False)['foodid'][:20]
        file.write('{},'.format(userid))
        cnt = 0
        for foodid in foods:
            file.write('{}'.format(foodid))
            cnt += 1
            if cnt >= 20:
                file.write('\n')
                break
            else:
                file.write(' ')