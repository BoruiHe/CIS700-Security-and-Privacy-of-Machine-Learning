import pandas as pd
from datetime import datetime
import time


ml_title = pd.read_csv('movies.csv', header=0, encoding='UTF-8')
ml_movie = {}
for index, elem in ml_title.iterrows():
    s = elem['title'].lower().replace(" ","")
    ml_movie[s] = int(index)
ml_movie = pd.Series(ml_movie)

nf_title = pd.read_csv('movie_titles.csv', header=None, encoding='ISO-8859-1', names=['year', 'title'])
nf_movie = {}
for index, elem in nf_title.iterrows():
    if not pd.isnull(elem['year']):
        s = (''.join(elem['title'])+"("+str(int(elem['year']))+")").lower().replace(" ","")
    else:
        s = (''.join(elem['title'])+"(NULL)").lower().replace(" ","")
    nf_movie[s] = int(index)
nf_movie = pd.Series(nf_movie)

common_movies = nf_movie[:4498].index.intersection(ml_movie.index)
matches = pd.Series(data=ml_movie.loc[common_movies].values,index=nf_movie.loc[common_movies].values)
matches.name = 'ml_movieId'
matches.index.name = 'nf_movieId'

ml_rating = pd.read_csv('ratings.csv', header=0, encoding='UTF-8')
start =  time.mktime(datetime(1998,10,1).timetuple())
end =  time.mktime(datetime(2006,1,1).timetuple())
ml_rating = ml_rating[(start < ml_rating['timestamp']) & (ml_rating['timestamp'] < end)]
ml_rating = ml_rating.loc[ml_rating['movieId'].isin(matches),:]
ml_rating.to_csv("ml_rating.csv", index=False)

nf_rating = []
combined_data = pd.read_csv('combined_data_1.txt', header=None, names=['userId','rating','timestamp'], encoding="ISO-8859-1")
cut_point = combined_data[pd.isnull(combined_data['rating'])]
for idx, elem in cut_point.iterrows():
    if int(elem['userId'][:-1]) in list(matches.index):
        start = end = idx + 1
        while not ((end + 1) in list(cut_point.index)):
            end += 1
        seg = combined_data.loc[start:end]
        seg.loc[:, 'movieId'] = matches[int(elem['userId'][:-1])]
        # print(seg)
        nf_rating.append(seg)
nf_rating = pd.concat(nf_rating, copy=False)
nf_rating.to_csv("nf_rating.csv", index=False)

pass