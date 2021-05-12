import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math


res_n = pickle.load(open('big_dict_n.pkl', 'rb'), encoding='utf-8')
x_axis, total, total_5, total_1  = [], [], [], []
for i in res_n[0.1]:
    total.append(i['total'])
for i in res_n[0.5]:
    total_5.append(i['total'])
for i in res_n[1]:
    total_1.append(i['total'])
for i in range(100):
    x_axis.append(i+1)
plt.plot(x_axis,total,color = 'r',label="epsilon=0.1")
plt.plot(x_axis,total_5,color = 'g',label="epsilon=0.5")
plt.plot(x_axis,total_1,color = 'b',label="epsilon=1")
plt.xlabel("the number of experiment")
plt.ylabel("the number of total matched pairs")
plt.legend(loc = "best")
plt.show()

res_n = pickle.load(open('big_dict_r.pkl', 'rb'), encoding='utf-8')
x_axis, total, total_5, total_1  = [], [], [], []
for i in res_n[0.1]:
    total.append(i['total'])
for i in res_n[0.5]:
    total_5.append(i['total'])
for i in res_n[1]:
    total_1.append(i['total'])
for i in range(100):
    x_axis.append(i+1)
plt.plot(x_axis,total,color = 'r',label="epsilon=0.1")
plt.plot(x_axis,total_5,color = 'g',label="epsilon=0.5")
plt.plot(x_axis,total_1,color = 'b',label="epsilon=1")
plt.xlabel("the number of experiment")
plt.ylabel("the number of total matched pairs")
plt.legend(loc = "best")
plt.show()
pass