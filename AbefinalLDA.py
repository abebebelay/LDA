
# coding: utf-8

# In[111]:


# -*- coding: utf-8 -*-
"""
Created on Sat May 12 06:36:15 2018

@author: AbebeB
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:29:40 2018

@author: AbebeB
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:51:45 2018
@author: AbebeB
"""
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 06:46:17 2018
@author: AbebeB
"""

import pandas as pd
import numpy as np
import math
import numpy.linalg as linalg
from matplotlib import pyplot as plt
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv


# In[112]:



#file accessed from CSV file
#url = 'iris.csv'
url = 'aa.csv'

# Read each features from our file i.e only features that requires dimentional reduction
dataset=pd.read_csv(url,dtype={'AP1': np.float64,'AP2': np.float64,'AP3': np.float64,'AP4': np.float64,'AP5': np.float64,
                               'AP6': np.float64,'AP7': np.float64,'AP8': np.float64,'AP9': np.float64,'AP10': np.float64,
                               'AP11': np.float64,'AP12': np.float64,'AP13': np.float64,'AP14': np.float64,'AP15': np.float64,
                               'AP16': np.float64,'AP17': np.float64,'AP18': np.float64,'AP19': np.float64,'AP20': np.float64,
                               'AP21': np.float64,'AP22': np.float64,'AP23': np.float64,'AP24': np.float64,'AP25': np.float64,
                               'AP26': np.float64,'Class_target': np.object})
#print(dataset) 


#----------differenciate each data frames
df = pd.DataFrame(dataset)


# In[113]:


#print(df) 
#---------Assign Each features to each class by creating data frames i.e 15 for our case
a1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
b1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
c1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
d1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
e1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
f1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
g1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
h1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
i1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
j1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
k1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
l1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
m1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
n1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
o1 = pd.DataFrame(columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16',
                               'AP17','AP18','AP19','AP20','AP21','AP22','AP23','AP24','AP25','AP26','Class_target'])
print('The class 1 contains features of:\n',a1)


# In[114]:


# --------------To filter the orginal data to assign for each classes  i.e for 15 classes for our case 
for t in range (len(df)):
    if df.iloc[t][26]=='a':
        a1=a1.append(df.iloc[t])
    if df.iloc[t][26]=='b':
        b1=b1.append(df.iloc[t])
    if df.iloc[t][26]=='c':
        c1=c1.append(df.iloc[t])
    if df.iloc[t][26]=='d':
        d1=d1.append(df.iloc[t])
    if df.iloc[t][26]=='e':
        e1=e1.append(df.iloc[t])
    if df.iloc[t][26]=='f':
        f1=f1.append(df.iloc[t])
    if df.iloc[t][26]=='g':
        g1=g1.append(df.iloc[t])
    if df.iloc[t][26]=='h':
        h1=h1.append(df.iloc[t])
    if df.iloc[t][26]=='i':
        i1=i1.append(df.iloc[t])
    if df.iloc[t][26]=='j':
        j1=j1.append(df.iloc[t])
    if df.iloc[t][26]=='k':
        k1=k1.append(df.iloc[t])
    if df.iloc[t][26]=='l':
        l1=l1.append(df.iloc[t])
    if df.iloc[t][26]=='m':
        m1=m1.append(df.iloc[t])
    if df.iloc[t][26]=='n':
        n1=n1.append(df.iloc[t])
    if df.iloc[t][26]=='o':
        o1=o1.append(df.iloc[t])

# To print each apeanded classes 
print('The appended values including the target in the 15th class are:\n',a1)


# In[115]:


# calculate the mean of all features corresponding to each class .i.e effect of features in each 15 classes for our case independantly 
# i.e 26 features effect in each 15 classes 
mf1=np.mean(a1)
mf2=np.mean(b1)
mf3=np.mean(c1)
mf4=np.mean(d1)
mf5=np.mean(e1)
mf6=np.mean(f1)
mf7=np.mean(g1)
mf8=np.mean(h1)
mf9=np.mean(i1)
mf10=np.mean(j1)
mf11=np.mean(k1)
mf12=np.mean(l1)
mf13=np.mean(m1)
mf14=np.mean(n1)
mf15=np.mean(o1)
# display mean of all features in the 15th class 
print('The mean of the appended values in the 15th class from each feature(excluding the target values) are:\n',mf15)


# In[116]:


# Appending each 26 features' mean to each classes i.e 26 features mean to each 15 classes  i.e 26X15 matrix is created for our case
smean1=(np.append(np.mean(a1['AP1']),np.append(np.mean(a1['AP2']),(np.append(np.mean(a1['AP3']),(np.append(np.mean(a1['AP4']),(np.append(np.mean(a1['AP5']),(np.append(np.mean(a1['AP6']),(np.append(np.mean(a1['AP7']),(np.append(np.mean(a1['AP8']),(np.append(np.mean(a1['AP9']),(np.append(np.mean(a1['AP10']),(np.append(np.mean(a1['AP11']),(np.append(np.mean(a1['AP12']),(np.append(np.mean(a1['AP13']),(np.append(np.mean(a1['AP14']),(np.append(np.mean(a1['AP15']),(np.append(np.mean(a1['AP16']),(np.append(np.mean(a1['AP17']),(np.append(np.mean(a1['AP18']),(np.append(np.mean(a1['AP19']),(np.append(np.mean(a1['AP20']),(np.append(np.mean(a1['AP21']),(np.append(np.mean(a1['AP22']),(np.append(np.mean(a1['AP23']),(np.append(np.mean(a1['AP24']),(np.append(np.mean(a1['AP25']),np.mean(a1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean2=(np.append(np.mean(b1['AP1']),np.append(np.mean(b1['AP2']),(np.append(np.mean(b1['AP3']),(np.append(np.mean(b1['AP4']),(np.append(np.mean(b1['AP5']),(np.append(np.mean(b1['AP6']),(np.append(np.mean(b1['AP7']),(np.append(np.mean(b1['AP8']),(np.append(np.mean(b1['AP9']),(np.append(np.mean(b1['AP10']),(np.append(np.mean(b1['AP11']),(np.append(np.mean(b1['AP12']),(np.append(np.mean(b1['AP13']),(np.append(np.mean(b1['AP14']),(np.append(np.mean(b1['AP15']),(np.append(np.mean(b1['AP16']),(np.append(np.mean(b1['AP17']),(np.append(np.mean(b1['AP18']),(np.append(np.mean(b1['AP19']),(np.append(np.mean(b1['AP20']),(np.append(np.mean(b1['AP21']),(np.append(np.mean(b1['AP22']),(np.append(np.mean(b1['AP23']),(np.append(np.mean(b1['AP24']),(np.append(np.mean(b1['AP25']),np.mean(b1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean3=(np.append(np.mean(c1['AP1']),np.append(np.mean(c1['AP2']),(np.append(np.mean(c1['AP3']),(np.append(np.mean(c1['AP4']),(np.append(np.mean(c1['AP5']),(np.append(np.mean(c1['AP6']),(np.append(np.mean(c1['AP7']),(np.append(np.mean(c1['AP8']),(np.append(np.mean(c1['AP9']),(np.append(np.mean(c1['AP10']),(np.append(np.mean(c1['AP11']),(np.append(np.mean(c1['AP12']),(np.append(np.mean(c1['AP13']),(np.append(np.mean(c1['AP14']),(np.append(np.mean(c1['AP15']),(np.append(np.mean(c1['AP16']),(np.append(np.mean(c1['AP17']),(np.append(np.mean(c1['AP18']),(np.append(np.mean(c1['AP19']),(np.append(np.mean(c1['AP20']),(np.append(np.mean(c1['AP21']),(np.append(np.mean(c1['AP22']),(np.append(np.mean(c1['AP23']),(np.append(np.mean(c1['AP24']),(np.append(np.mean(c1['AP25']),np.mean(c1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean4=(np.append(np.mean(d1['AP1']),np.append(np.mean(d1['AP2']),(np.append(np.mean(d1['AP3']),(np.append(np.mean(d1['AP4']),(np.append(np.mean(d1['AP5']),(np.append(np.mean(d1['AP6']),(np.append(np.mean(d1['AP7']),(np.append(np.mean(d1['AP8']),(np.append(np.mean(d1['AP9']),(np.append(np.mean(d1['AP10']),(np.append(np.mean(d1['AP11']),(np.append(np.mean(d1['AP12']),(np.append(np.mean(d1['AP13']),(np.append(np.mean(d1['AP14']),(np.append(np.mean(d1['AP15']),(np.append(np.mean(d1['AP16']),(np.append(np.mean(d1['AP17']),(np.append(np.mean(d1['AP18']),(np.append(np.mean(d1['AP19']),(np.append(np.mean(d1['AP20']),(np.append(np.mean(d1['AP21']),(np.append(np.mean(d1['AP22']),(np.append(np.mean(d1['AP23']),(np.append(np.mean(d1['AP24']),(np.append(np.mean(d1['AP25']),np.mean(d1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean5=(np.append(np.mean(e1['AP1']),np.append(np.mean(e1['AP2']),(np.append(np.mean(e1['AP3']),(np.append(np.mean(e1['AP4']),(np.append(np.mean(e1['AP5']),(np.append(np.mean(e1['AP6']),(np.append(np.mean(e1['AP7']),(np.append(np.mean(e1['AP8']),(np.append(np.mean(e1['AP9']),(np.append(np.mean(e1['AP10']),(np.append(np.mean(e1['AP11']),(np.append(np.mean(e1['AP12']),(np.append(np.mean(e1['AP13']),(np.append(np.mean(e1['AP14']),(np.append(np.mean(e1['AP15']),(np.append(np.mean(e1['AP16']),(np.append(np.mean(e1['AP17']),(np.append(np.mean(e1['AP18']),(np.append(np.mean(e1['AP19']),(np.append(np.mean(e1['AP20']),(np.append(np.mean(e1['AP21']),(np.append(np.mean(e1['AP22']),(np.append(np.mean(e1['AP23']),(np.append(np.mean(e1['AP24']),(np.append(np.mean(e1['AP25']),np.mean(e1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean6=(np.append(np.mean(f1['AP1']),np.append(np.mean(f1['AP2']),(np.append(np.mean(f1['AP3']),(np.append(np.mean(f1['AP4']),(np.append(np.mean(f1['AP5']),(np.append(np.mean(f1['AP6']),(np.append(np.mean(f1['AP7']),(np.append(np.mean(f1['AP8']),(np.append(np.mean(f1['AP9']),(np.append(np.mean(f1['AP10']),(np.append(np.mean(f1['AP11']),(np.append(np.mean(f1['AP12']),(np.append(np.mean(f1['AP13']),(np.append(np.mean(f1['AP14']),(np.append(np.mean(f1['AP15']),(np.append(np.mean(f1['AP16']),(np.append(np.mean(f1['AP17']),(np.append(np.mean(f1['AP18']),(np.append(np.mean(f1['AP19']),(np.append(np.mean(f1['AP20']),(np.append(np.mean(f1['AP21']),(np.append(np.mean(f1['AP22']),(np.append(np.mean(f1['AP23']),(np.append(np.mean(f1['AP24']),(np.append(np.mean(f1['AP25']),np.mean(f1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean7=(np.append(np.mean(g1['AP1']),np.append(np.mean(g1['AP2']),(np.append(np.mean(g1['AP3']),(np.append(np.mean(g1['AP4']),(np.append(np.mean(g1['AP5']),(np.append(np.mean(g1['AP6']),(np.append(np.mean(g1['AP7']),(np.append(np.mean(g1['AP8']),(np.append(np.mean(g1['AP9']),(np.append(np.mean(g1['AP10']),(np.append(np.mean(g1['AP11']),(np.append(np.mean(g1['AP12']),(np.append(np.mean(g1['AP13']),(np.append(np.mean(g1['AP14']),(np.append(np.mean(g1['AP15']),(np.append(np.mean(g1['AP16']),(np.append(np.mean(g1['AP17']),(np.append(np.mean(g1['AP18']),(np.append(np.mean(g1['AP19']),(np.append(np.mean(g1['AP20']),(np.append(np.mean(g1['AP21']),(np.append(np.mean(g1['AP22']),(np.append(np.mean(g1['AP23']),(np.append(np.mean(g1['AP24']),(np.append(np.mean(g1['AP25']),np.mean(g1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean8=(np.append(np.mean(h1['AP1']),np.append(np.mean(h1['AP2']),(np.append(np.mean(h1['AP3']),(np.append(np.mean(h1['AP4']),(np.append(np.mean(h1['AP5']),(np.append(np.mean(h1['AP6']),(np.append(np.mean(h1['AP7']),(np.append(np.mean(h1['AP8']),(np.append(np.mean(h1['AP9']),(np.append(np.mean(h1['AP10']),(np.append(np.mean(h1['AP11']),(np.append(np.mean(h1['AP12']),(np.append(np.mean(h1['AP13']),(np.append(np.mean(h1['AP14']),(np.append(np.mean(h1['AP15']),(np.append(np.mean(h1['AP16']),(np.append(np.mean(h1['AP17']),(np.append(np.mean(h1['AP18']),(np.append(np.mean(h1['AP19']),(np.append(np.mean(h1['AP20']),(np.append(np.mean(h1['AP21']),(np.append(np.mean(h1['AP22']),(np.append(np.mean(h1['AP23']),(np.append(np.mean(h1['AP24']),(np.append(np.mean(h1['AP25']),np.mean(h1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean9=(np.append(np.mean(i1['AP1']),np.append(np.mean(i1['AP2']),(np.append(np.mean(i1['AP3']),(np.append(np.mean(i1['AP4']),(np.append(np.mean(i1['AP5']),(np.append(np.mean(i1['AP6']),(np.append(np.mean(i1['AP7']),(np.append(np.mean(i1['AP8']),(np.append(np.mean(i1['AP9']),(np.append(np.mean(i1['AP10']),(np.append(np.mean(i1['AP11']),(np.append(np.mean(i1['AP12']),(np.append(np.mean(i1['AP13']),(np.append(np.mean(i1['AP14']),(np.append(np.mean(i1['AP15']),(np.append(np.mean(i1['AP16']),(np.append(np.mean(i1['AP17']),(np.append(np.mean(i1['AP18']),(np.append(np.mean(i1['AP19']),(np.append(np.mean(i1['AP20']),(np.append(np.mean(i1['AP21']),(np.append(np.mean(i1['AP22']),(np.append(np.mean(i1['AP23']),(np.append(np.mean(i1['AP24']),(np.append(np.mean(i1['AP25']),np.mean(i1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean10=(np.append(np.mean(j1['AP1']),np.append(np.mean(j1['AP2']),(np.append(np.mean(j1['AP3']),(np.append(np.mean(j1['AP4']),(np.append(np.mean(j1['AP5']),(np.append(np.mean(j1['AP6']),(np.append(np.mean(j1['AP7']),(np.append(np.mean(j1['AP8']),(np.append(np.mean(j1['AP9']),(np.append(np.mean(j1['AP10']),(np.append(np.mean(j1['AP11']),(np.append(np.mean(j1['AP12']),(np.append(np.mean(j1['AP13']),(np.append(np.mean(j1['AP14']),(np.append(np.mean(j1['AP15']),(np.append(np.mean(j1['AP16']),(np.append(np.mean(j1['AP17']),(np.append(np.mean(j1['AP18']),(np.append(np.mean(j1['AP19']),(np.append(np.mean(j1['AP20']),(np.append(np.mean(j1['AP21']),(np.append(np.mean(j1['AP22']),(np.append(np.mean(j1['AP23']),(np.append(np.mean(j1['AP24']),(np.append(np.mean(j1['AP25']),np.mean(j1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean11=(np.append(np.mean(k1['AP1']),np.append(np.mean(k1['AP2']),(np.append(np.mean(k1['AP3']),(np.append(np.mean(k1['AP4']),(np.append(np.mean(k1['AP5']),(np.append(np.mean(k1['AP6']),(np.append(np.mean(k1['AP7']),(np.append(np.mean(k1['AP8']),(np.append(np.mean(k1['AP9']),(np.append(np.mean(k1['AP10']),(np.append(np.mean(k1['AP11']),(np.append(np.mean(k1['AP12']),(np.append(np.mean(k1['AP13']),(np.append(np.mean(k1['AP14']),(np.append(np.mean(k1['AP15']),(np.append(np.mean(k1['AP16']),(np.append(np.mean(k1['AP17']),(np.append(np.mean(k1['AP18']),(np.append(np.mean(k1['AP19']),(np.append(np.mean(k1['AP20']),(np.append(np.mean(k1['AP21']),(np.append(np.mean(k1['AP22']),(np.append(np.mean(k1['AP23']),(np.append(np.mean(k1['AP24']),(np.append(np.mean(k1['AP25']),np.mean(k1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean12=(np.append(np.mean(l1['AP1']),np.append(np.mean(l1['AP2']),(np.append(np.mean(l1['AP3']),(np.append(np.mean(l1['AP4']),(np.append(np.mean(l1['AP5']),(np.append(np.mean(l1['AP6']),(np.append(np.mean(l1['AP7']),(np.append(np.mean(l1['AP8']),(np.append(np.mean(l1['AP9']),(np.append(np.mean(l1['AP10']),(np.append(np.mean(l1['AP11']),(np.append(np.mean(l1['AP12']),(np.append(np.mean(l1['AP13']),(np.append(np.mean(l1['AP14']),(np.append(np.mean(l1['AP15']),(np.append(np.mean(l1['AP16']),(np.append(np.mean(l1['AP17']),(np.append(np.mean(l1['AP18']),(np.append(np.mean(l1['AP19']),(np.append(np.mean(l1['AP20']),(np.append(np.mean(l1['AP21']),(np.append(np.mean(l1['AP22']),(np.append(np.mean(l1['AP23']),(np.append(np.mean(l1['AP24']),(np.append(np.mean(l1['AP25']),np.mean(l1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean13=(np.append(np.mean(m1['AP1']),np.append(np.mean(m1['AP2']),(np.append(np.mean(m1['AP3']),(np.append(np.mean(m1['AP4']),(np.append(np.mean(m1['AP5']),(np.append(np.mean(m1['AP6']),(np.append(np.mean(m1['AP7']),(np.append(np.mean(m1['AP8']),(np.append(np.mean(m1['AP9']),(np.append(np.mean(m1['AP10']),(np.append(np.mean(m1['AP11']),(np.append(np.mean(m1['AP12']),(np.append(np.mean(m1['AP13']),(np.append(np.mean(m1['AP14']),(np.append(np.mean(m1['AP15']),(np.append(np.mean(m1['AP16']),(np.append(np.mean(m1['AP17']),(np.append(np.mean(m1['AP18']),(np.append(np.mean(m1['AP19']),(np.append(np.mean(m1['AP20']),(np.append(np.mean(m1['AP21']),(np.append(np.mean(m1['AP22']),(np.append(np.mean(m1['AP23']),(np.append(np.mean(m1['AP24']),(np.append(np.mean(m1['AP25']),np.mean(m1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean14=(np.append(np.mean(n1['AP1']),np.append(np.mean(n1['AP2']),(np.append(np.mean(n1['AP3']),(np.append(np.mean(n1['AP4']),(np.append(np.mean(n1['AP5']),(np.append(np.mean(n1['AP6']),(np.append(np.mean(n1['AP7']),(np.append(np.mean(n1['AP8']),(np.append(np.mean(n1['AP9']),(np.append(np.mean(n1['AP10']),(np.append(np.mean(n1['AP11']),(np.append(np.mean(n1['AP12']),(np.append(np.mean(n1['AP13']),(np.append(np.mean(n1['AP14']),(np.append(np.mean(n1['AP15']),(np.append(np.mean(n1['AP16']),(np.append(np.mean(n1['AP17']),(np.append(np.mean(n1['AP18']),(np.append(np.mean(n1['AP19']),(np.append(np.mean(n1['AP20']),(np.append(np.mean(n1['AP21']),(np.append(np.mean(n1['AP22']),(np.append(np.mean(n1['AP23']),(np.append(np.mean(n1['AP24']),(np.append(np.mean(n1['AP25']),np.mean(n1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
smean15=(np.append(np.mean(o1['AP1']),np.append(np.mean(o1['AP2']),(np.append(np.mean(o1['AP3']),(np.append(np.mean(o1['AP4']),(np.append(np.mean(o1['AP5']),(np.append(np.mean(o1['AP6']),(np.append(np.mean(o1['AP7']),(np.append(np.mean(o1['AP8']),(np.append(np.mean(o1['AP9']),(np.append(np.mean(o1['AP10']),(np.append(np.mean(o1['AP11']),(np.append(np.mean(o1['AP12']),(np.append(np.mean(o1['AP13']),(np.append(np.mean(o1['AP14']),(np.append(np.mean(o1['AP15']),(np.append(np.mean(o1['AP16']),(np.append(np.mean(o1['AP17']),(np.append(np.mean(o1['AP18']),(np.append(np.mean(o1['AP19']),(np.append(np.mean(o1['AP20']),(np.append(np.mean(o1['AP21']),(np.append(np.mean(o1['AP22']),(np.append(np.mean(o1['AP23']),(np.append(np.mean(o1['AP24']),(np.append(np.mean(o1['AP25']),np.mean(o1['AP26']))))))))))))))))))))))))))))))))))))))))))))))))))
print('The mean of 26 features values in the 15th class:\n',smean15)


# In[117]:


# append N-1 times each pair sequencially
appendrow1=np.vstack((smean1,smean2))
appendrow2=np.vstack((appendrow1,smean3))
appendrow3=np.vstack((appendrow2,smean4))
appendrow4=np.vstack((appendrow3,smean5))
appendrow5=np.vstack((appendrow4,smean6))
appendrow6=np.vstack((appendrow5,smean7))
appendrow7=np.vstack((appendrow6,smean8))
appendrow8=np.vstack((appendrow7,smean9))
appendrow9=np.vstack((appendrow8,smean10))
appendrow10=np.vstack((appendrow9,smean11))
appendrow11=np.vstack((appendrow10,smean12))
appendrow12=np.vstack((appendrow11,smean13))
appendrow13=np.vstack((appendrow12,smean14))
appendrow14=np.vstack((appendrow13,smean15))
# to display all appended rows together with
print('The whole features mean values in each class:\n',appendrow14)


# In[118]:



#Calculate all mean in one
overall=(np.mean(appendrow14,axis=0))
print('The overall mean of the 36,660 datasets of each features is :\n',overall)


# In[119]:


# assign each values from its class to certain variable Xi
x1 = (a1.iloc[0:len(a1), 0:26])
x2 = (b1.iloc[0:len(b1), 0:26])
x3 = (c1.iloc[0:len(c1), 0:26])
x4 = (d1.iloc[0:len(d1), 0:26])
x5 = (e1.iloc[0:len(e1), 0:26])
x6 = (f1.iloc[0:len(f1), 0:26])
x7 = (g1.iloc[0:len(g1), 0:26])
x8 = (h1.iloc[0:len(h1), 0:26])
x9 = (i1.iloc[0:len(i1), 0:26])
x10 = (j1.iloc[0:len(j1), 0:26])
x11 = (k1.iloc[0:len(k1), 0:26])
x12 = (l1.iloc[0:len(l1), 0:26])
x13 = (m1.iloc[0:len(m1), 0:26])
x14 = (n1.iloc[0:len(n1), 0:26])
x15 = (o1.iloc[0:len(o1), 0:26])
print('Assign data values to a variable x15:\n',x15)


# In[120]:


#find mean difference from each class using the following formula, for our case for 
xx1=(x1-mf1).T.dot((x1 - mf1))
xx2=(x2-mf2).T.dot((x2 - mf2))
xx3=(x3-mf3).T.dot((x3 - mf3))
xx4=(x4-mf4).T.dot((x4 - mf4))
xx5=(x5-mf5).T.dot((x5 - mf5))
xx6=(x6-mf6).T.dot((x6 - mf6))
xx7=(x7-mf7).T.dot((x7 - mf7))
xx8=(x8-mf8).T.dot((x8 - mf8))
xx9=(x9-mf9).T.dot((x9 - mf9))
xx10=(x10-mf10).T.dot((x10 - mf10))
xx11=(x11-mf11).T.dot((x11 - mf11))
xx12=(x12-mf12).T.dot((x12 - mf12))
xx13=(x13-mf13).T.dot((x13 - mf13))
xx14=(x14-mf14).T.dot((x14 - mf14))
xx15=(x15-mf15).T.dot((x15 - mf15))




#Adding each mean obtained from each classes  i.e  with in class matrix
S1=xx1+xx2+xx3+xx4+xx5+xx6+xx7+xx8+xx9+xx10+xx11+xx12+xx13+xx14+xx15
print('The mean difference of the datasets are:\n',S1)


# In[121]:


# prepare dataframe from each class means
convert1=pd.DataFrame(smean1)
convert2=pd.DataFrame(smean2)
convert3=pd.DataFrame(smean3)
convert4=pd.DataFrame(smean4)
convert5=pd.DataFrame(smean5)
convert6=pd.DataFrame(smean6)
convert7=pd.DataFrame(smean7)
convert8=pd.DataFrame(smean8)
convert9=pd.DataFrame(smean9)
convert10=pd.DataFrame(smean10)
convert11=pd.DataFrame(smean11)
convert12=pd.DataFrame(smean12)
convert13=pd.DataFrame(smean13)
convert14=pd.DataFrame(smean14)
convert15=pd.DataFrame(smean15)
print('The dataframe for class-15 is:\n',convert15)


# In[122]:


# To Calculate between-class matrix i.e each 15 classes with overall mean 
# here, 50 mean the numbr of data sets in each class. It may be different for different classes so that we should count 
#for each classes accourdingly 
converted1=((4948*(convert1.T)-overall).T.dot((convert1.T)-overall))
converted2=((2460*(convert2.T)-overall).T.dot((convert2.T)-overall))
converted3=((2448*(convert3.T)-overall).T.dot((convert3.T)-overall))
converted4=((1757*(convert4.T)-overall).T.dot((convert4.T)-overall))
converted5=((2448*(convert5.T)-overall).T.dot((convert5.T)-overall))
converted6=((900*(convert6.T)-overall).T.dot((convert6.T)-overall))
converted7=((2894*(convert7.T)-overall).T.dot((convert7.T)-overall))
converted8=((3344*(convert8.T)-overall).T.dot((convert8.T)-overall))
converted9=((4394*(convert9.T)-overall).T.dot((convert9.T)-overall))
converted10=((1914*(convert10.T)-overall).T.dot((convert10.T)-overall))
converted11=((2228*(convert11.T)-overall).T.dot((convert11.T)-overall))
converted12=((1199*(convert12.T)-overall).T.dot((convert12.T)-overall))
converted13=((1783*(convert13.T)-overall).T.dot((convert13.T)-overall))
converted14=((3026*(convert14.T)-overall).T.dot((convert14.T)-overall))
converted15=((915*(convert15.T)-overall).T.dot((convert15.T)-overall))

# sum up the whole classes mean coresponding to the total class together in one i.e between classes 
S2=converted1+converted2+converted3+converted4+converted5+converted6+converted7+converted8+converted9+converted10+converted11+converted12+converted13+converted14+converted15
print('The between class data districutions are:\n',S2)


# In[123]:


#compute x=nv(S1).dot(S2) for between class and inverse of with in class
#ax=np.linalg.inv(S1)
#print(S1)

#ay=ax.dot(S2)
#print(ay)
evalues, evectors = np.linalg.eig(np.linalg.inv(S1).dot(S2)) 
print('The evectors values from dot product of s1 and inverse of s1:',evectors)
print('The evalues obtained from dot product of s1 and inverse of s1',evalues)


# In[124]:


eig_pairs = [(np.abs(evalues[i]), evectors[:,i]) for i in range(len(evalues))]
print('length of evalues is:',len(evalues))
#eig_pairs.sort()


# In[125]:


eig_pairs.sort(key=lambda x: x[0], reverse=True)  
#eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
eig_pairs.reverse()
print('Eigenvalues in descending order:',eig_pairs)
for i in eig_pairs:
	print(i[0])


# In[126]:


#-----------graph of cross product of Sb --------
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 5))
    colors = np.random.rand()
    plt.scatter(S1,S2,  s=60, c='red', marker='^')    
    plt.ylabel('Between-Class  variance')
    plt.xlabel('In-class variance')
    #plt.title('LDA for dimentional reduction')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()
#-----------end ----------------------------------------


# In[127]:


#Graphical representation of each eigen values of each features 
tot = sum(evalues)
print('etotal of sum of eigen values:',tot)
var_exp = [(i / tot)*100 for i in sorted(evalues, reverse=True)]
print('explained variance:',var_exp)
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)


# In[132]:


#=============graphing the calculated values=============
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 5))
    plt.bar(range(26), var_exp, alpha=0.5, align='center', label='individual explained variance')
    #plt.step(range(26), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.step(range(26), var_exp, where='mid', label='variance distribution')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    #plt.title('LDA for dimentional reduction')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
plt.savefig('PREDI2.png', format='png', dpi=1200)
plt.show()
    

