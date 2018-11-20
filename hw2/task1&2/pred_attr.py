import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from keras.models import load_model
from NeuMF_attr_tf import get_model

dataset = pd.read_csv('ntucsie-sdml2018-2-1/rating_train.csv', header=0)
foods = dataset['foodid'].unique()
users = dataset['userid'].unique()

with open('history.pickle', 'rb') as file:
    history = pickle.load(file)

with open('usermap.pickle', 'rb') as u_file, open('foodmap.pickle', 'rb') as f_file:
    usermap = pickle.load(u_file)
    foodmap = pickle.load(f_file)

food_attrs = []
for foodid in foods:
    food_attrs.append(foodmap[foodid])
food_attrs = np.array(food_attrs)

model = get_model(9895, 5532, 85, 10, 32, layers=[128, 64, 32, 16], reg_layers=[0, 0, 0, 0])
model = load_model('test.h5')

with open('pred.csv', 'w') as file:
    file.write('userid,foodid\n')
    
    for userid in users:
        user = np.full(len(foods), userid)
        user_attrs = np.tile(usermap[userid], (len(foods), 1))
        pred = model.predict([user, foods, user_attrs, food_attrs], batch_size=1024, verbose=1).flatten()
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

        # print(foods[sort_indices][:10])
        # print(pred[sort_indices][:10])


# history = defaultdict(set)
# for index, row in dataset.iterrows():
#     history[row['userid']].add(row['foodid'])

# for userid in history:
#     for foodid in range(n_food):
#         score = model.predict([userid, foodid])
#         print(score)