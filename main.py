import pandas as pd
import os
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

Rho, d, std_ecc = 1.5, 30, 1.5
random_seed1, random_seed2 = (2, 85) 
if os.path.exists('better_NF.csv') and os.path.exists('better_ML.csv'):
    nf_rating = pd.read_csv('better_NF.csv', header=0, encoding='UTF-8')
    ml_rating = pd.read_csv('better_ML.csv', header=0, encoding='UTF-8')
else:
    import data_processing
    import timestamp_and_rating

weight = 1/np.log10(nf_rating.loc[:,"movieId"].value_counts())
ml_rating.set_index('movieId', inplace=True)
nf_rating.set_index('movieId', inplace=True)
ml_rating.rename(columns={'userId':'uM'}, inplace= True)
nf_rating.rename(columns={'userId':'uN'}, inplace= True)

# I have to decrease the scale of my datasets again because of computing power.
# Find the intersection of first 500 movies in the first quarter of NetFlix Prize dataset and last 400+ movies in MovieLens dataset.
common_movies = nf_rating.index.unique()[:500].intersection(ml_rating.index.unique()[400:]).unique()
ml_rating = ml_rating.loc[common_movies,:]
nf_rating = nf_rating.loc[common_movies,:]
nfs_rating = nf_rating.sample(n=40000, random_state=random_seed1)
mls_rating = ml_rating.loc[list(nfs_rating.index.unique())].sample(n=10000, random_state=random_seed2)
diff = list(set(nfs_rating.index)-set(mls_rating.index))
mls_rating = mls_rating.append(ml_rating.loc[diff,:])
# I need to transfer timestamp in the same form before subtraction.
# Do not shrink dataset in the next line, or you probably get inconsistent datasets(NaN value) which is unsolvable.
res = np.abs(nfs_rating.subtract(mls_rating, fill_value=0).astype(np.int64))
with open('res.pkl', 'wb') as f:
    pickle.dump(res, f)
result = weight.loc[set(nfs_rating.index)]*(np.exp(-res.loc[:,'rating']/Rho) + np.exp(-res.loc[:,'timestamp']/d))
res.loc[:,'score'] = result
res.drop(['rating', 'timestamp'],axis=1,inplace=True)
res.set_index(['uN','uM'],inplace=True)

Mixed_user_DF = res.groupby(level=[0,1]).sum()
NF_user_DF = Mixed_user_DF.index.levels[0]

# Finding best matched users
matched_users = {}
for elem in NF_user_DF:
    ML_user_DF = Mixed_user_DF.loc[elem]
    if len(list(ML_user_DF.index)) > 1:
        tempScore = np.array(ML_user_DF['score'])
        stdev = np.std(tempScore)
        best1 = np.max(tempScore)
        idx1 = np.argmax(tempScore)
        tempScore = np.delete(tempScore, idx1)
        best2 = np.max(tempScore)
        idx2 = np.argmax(tempScore) 
        ecc = (best1-best2)/stdev
        if (ecc >= std_ecc): 
            matched_users[elem, list(ML_user_DF.index)[idx1-1]] = ecc

val2, val3 = [], []
for b in range(math.floor(max(list(matched_users.values())))):
    sum = 0
    for i in list(matched_users.values()):
        if i >= b+1 and i < b+2:
            sum += 1
    val3.append([sum])
    val2.append('({},{})'.format(b+1,b+2))
val1 = ['number']
val2.append('total')
val3.append([len(list(matched_users.values()))])
fig, ax = plt.subplots()
ax.set_axis_off() 
table = ax.table(cellText = val3, rowLabels = val2, colLabels = val1, rowColours =["palegreen"] * 10, colColours =["palegreen"] * 10, cellLoc ='center', loc ='upper left')         
ax.set_title('the number of paired users in each ecc interval', fontweight ="bold")
plt.savefig('ecc table.png')
# plt.show()

new = plt.figure()
plt.hist(list(matched_users.values()), bins=20)
plt.title('histogram of matched users')
plt.xlabel('eccentricity')
plt.ylabel('number')
plt.savefig('eccentricity.png')
# plt.show()

epsilon_ls = [0.1, 0.5, 1]
# Senario 1: applying perturbation to the number of records----------------------------------------------------------------------------------------
num_scale = 1
big_dict_n = {}
for epsilon in epsilon_ls:
    big_dict_n[epsilon] = []
    for t in range(100):
        res = pickle.load(open('res.pkl', 'rb'), encoding='utf-8')
        num_noise = np.random.laplace(0, num_scale/epsilon, size=1)
        while num_noise >= len(res) or num_noise <= 1:
            num_noise = np.random.laplace(0, num_scale/epsilon, size=1)
        result = weight.loc[set(nfs_rating.index)]*(np.exp(-res.loc[:,'rating']/Rho) + np.exp(-res.loc[:,'timestamp']/d))
        res.loc[:,'score'] = result
        deleted_rows = res.sample(n=math.floor(num_noise), replace=False, random_state=None)
        diff = pd.merge(res, deleted_rows, how='outer', indicator=True)
        res = diff.loc[diff._merge == 'left_only']
        # res = res.loc[:10000]
        res.drop(['_merge'],axis=1,inplace=True)
        res.drop(['rating', 'timestamp'],axis=1,inplace=True)
        res.set_index(['uN','uM'],inplace=True)

        Mixed_user_DF = res.groupby(level=[0,1]).sum()
        NF_user_DF = Mixed_user_DF.index.levels[0]

        # Finding best matched users
        matched_La_users = {}
        for elem in NF_user_DF:
            ML_user_DF = Mixed_user_DF.loc[elem]
            if len(list(ML_user_DF.index)) > 1:
                tempScore = np.array(ML_user_DF['score'])
                stdev = np.std(tempScore)
                best1 = np.max(tempScore)
                idx1 = np.argmax(tempScore)
                tempScore = np.delete(tempScore, idx1)
                best2 = np.max(tempScore)
                idx2 = np.argmax(tempScore) 
                ecc = (best1-best2)/stdev
                if (ecc >= std_ecc): 
                    matched_La_users[elem, list(ML_user_DF.index)[idx1-1]] = ecc

        new = plt.figure()
        plt.hist(list(matched_La_users.values()), bins='auto')
        plt.title('histogram of matched users after applying perturbation')
        plt.xlabel('eccentricity')
        plt.ylabel('number')
        plt.savefig('eccentricity_{}_n.png'.format(epsilon))
        # plt.show()

        val2, val3 = [], []
        for b in range(math.floor(max(list(matched_La_users.values())))):
            sum = 0
            for i in list(matched_La_users.values()):
                if i >= b+1 and i < b+2:
                    sum += 1
            val3.append([sum])
            val2.append('({},{})'.format(b+1,b+2))
        val1 = ['number']
        val2.append('total')
        val3.append([len(list(matched_La_users.values()))])
        sub_dict = {}
        for key in val2:
            sub_dict[key] = val3[val2.index(key)]
        new = plt.figure()
        fig, ax = plt.subplots()
        ax.set_axis_off() 
        table = ax.table(cellText = val3, rowLabels = val2, colLabels = val1, rowColours =["palegreen"] * 13, colColours =["palegreen"] * 13, cellLoc ='center', loc ='upper left')         
        ax.set_title('the number of paired users in each ecc interval_{}_n'.format(epsilon), fontweight ="bold")
        plt.savefig('ecc table_{}_n.png'.format(epsilon))
        # plt.show()
        big_dict_n[epsilon].append(sub_dict)

with open('big_dict_n.pkl', 'wb') as f:
    pickle.dump(big_dict_n, f)

# Senario 2: applying perturbation to rating and date----------------------------------------------------------------------------------------------
rating_scale, timestamp_scale = 1, 60*60
big_dict_r = {}
for epsilon in epsilon_ls:
    big_dict_r[epsilon] = []
    for t in range(100):
        res = pickle.load(open('res.pkl', 'rb'), encoding='utf-8')
        rating_noise = np.random.laplace(0, rating_scale/epsilon, size=res.shape[0])
        timestamp_noise = np.random.laplace(0, timestamp_scale/epsilon, size=res.shape[0])
        res['rating_{}'.format(epsilon)] = res['rating'] + rating_noise
        res['timestamp_{}'.format(epsilon)] = res['timestamp'] + timestamp_noise
        res[res['rating_{}'.format(epsilon)] < 1] = 1
        res[res['rating_{}'.format(epsilon)] > 5] = 5
        res[res['timestamp_{}'.format(epsilon)] < 0] = 0
        result = weight.loc[set(nfs_rating.index)]*(np.exp(-res.loc[:,'rating_{}'.format(epsilon)]/Rho) + np.exp(-res.loc[:,'timestamp_{}'.format(epsilon)]/d))
        res.loc[:,'score'] = result
        res.set_index(['uN','uM'],inplace=True)
        Mixed_user_DF = res.groupby(level=[0,1]).sum()
        NF_user_DF = Mixed_user_DF.index.levels[0]

        # Finding best matched users
        matched_La_users = {}
        for elem in NF_user_DF:
            ML_user_DF = Mixed_user_DF.loc[elem]
            if len(list(ML_user_DF.index)) > 1:
                tempScore = np.array(ML_user_DF['score'])
                stdev = np.std(tempScore)
                best1 = np.max(tempScore)
                idx1 = np.argmax(tempScore)
                tempScore = np.delete(tempScore, idx1)
                best2 = np.max(tempScore)
                idx2 = np.argmax(tempScore) 
                ecc = (best1-best2)/stdev
                if (ecc >= std_ecc): 
                    matched_La_users[elem, list(ML_user_DF.index)[idx1-1]] = ecc

        new = plt.figure()
        plt.hist(list(matched_La_users.values()), bins=20)
        plt.title('histogram of matched users after applying perturbation')
        plt.xlabel('eccentricity')
        plt.ylabel('number')
        plt.savefig('eccentricity_{}_r.png'.format(epsilon))
        # plt.show()

        val2, val3 = [], []
        for b in range(math.floor(max(list(matched_La_users.values())))):
            sum = 0
            for i in list(matched_La_users.values()):
                if i >= b+1 and i < b+2:
                    sum += 1
            val3.append([sum])
            val2.append('({},{})'.format(b+1,b+2))
        val1 = ['number']
        val2.append('total')
        val3.append([len(list(matched_La_users.values()))])
        sub_dict = {}
        for key in val2:
            sub_dict[key] = val3[val2.index(key)]
        new = plt.figure()
        fig, ax = plt.subplots()
        ax.set_axis_off() 
        table = ax.table(cellText = val3, rowLabels = val2, colLabels = val1, rowColours =["palegreen"] * 10, colColours =["palegreen"] * 10, cellLoc ='center', loc ='upper left')         
        ax.set_title('the number of paired users in each ecc interval_{}'.format(epsilon), fontweight ="bold")
        plt.savefig('ecc table_{}_R.png'.format(epsilon))
        # plt.show()
        big_dict_r[epsilon].append(sub_dict)

with open('big_dict_r.pkl', 'wb') as f:
    pickle.dump(big_dict_r, f)
pass