import pandas as pd
import numpy as np
import datetime


ml_rating = pd.read_csv("ml_rating.csv", header=0, encoding="UTF-8")
ml_rating["rating"] = np.ceil(ml_rating["rating"])
nf_rating = pd.read_csv("nf_rating.csv", header=0, encoding="UTF-8")

date = datetime.datetime.strptime('19981001', '%Y%m%d')
tmsp_ml = np.floor((ml_rating["timestamp"].apply(datetime.datetime.fromtimestamp) - date) / datetime.timedelta(days=1))
tmsp_nf = np.floor((nf_rating["timestamp"].astype(str).apply(datetime.datetime.strptime, args=('%Y-%m-%d',)) - date) / datetime.timedelta(days=1))
ml_rating["timestamp"] = tmsp_ml
nf_rating["timestamp"] = tmsp_nf
ml_rating.to_csv("better_ML.csv", index=False)
nf_rating.to_csv("better_NF.csv", index=False)
pass