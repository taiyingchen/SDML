import NeuMF
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

dataset = pd.read_csv('ntucsie-sdml2018-2-1/rating_train.csv', header=0)
n_user = dataset['userid'].max()
n_food = dataset['foodid'].max()
layers = [64, 32, 16 , 8]
reg_layers = [0, 0, 0, 0]

model = NeuMF.get_model(n_user, n_food, 8, layers, reg_layers)
model.load_weights('neumf.h5')

foods = dataset['foodid'].unique()
users = dataset['userid'].unique()

with open('history.pickle', 'rb') as file:
    history = pickle.load(file)

with open('pred.csv', 'w') as file:
    file.write('userid,foodid\n')
    
    for userid in users:
        pred = model.predict([np.full(len(foods), userid), foods]).flatten()
        sort_indices = (-pred).argsort()

        cnt = 0
        file.write('{},'.format(userid))
        for foodid in foods[sort_indices]:
            if foodid not in history[userid]:
                file.write('{}'.format(foodid))
                cnt = cnt + 1
                if cnt >= 20:
                    file.write('\n')
                    break
                else:
                    file.write(' ')

        print(foods[sort_indices][:10])
        print(pred[sort_indices][:10])


# history = defaultdict(set)
# for index, row in dataset.iterrows():
#     history[row['userid']].add(row['foodid'])

# for userid in history:
#     for foodid in range(n_food):
#         score = model.predict([userid, foodid])
#         print(score)