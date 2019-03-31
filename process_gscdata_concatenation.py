
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random 
import csv
from sklearn.utils import shuffle 


# In[2]:


same = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\GSC-Dataset(1)\GSC-Dataset\GSC-Features-Data\same_pairs.csv',usecols=['img_id_A','img_id_B','target'])


# In[4]:


diff = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\GSC-Dataset(1)\GSC-Dataset\GSC-Features-Data\diffn_pairs.csv')


# In[8]:


rindex =  np.array(random.sample(range(len(diff)), 71531))
dfr = diff.loc[rindex]
samediff = [same,dfr]
samediff_neat = pd.concat(samediff)


# In[9]:



samediff_neat.head()
samediff_neat = shuffle(samediff_neat)


# In[10]:


featureset = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\GSC-Dataset(1)\GSC-Dataset\GSC-Features-Data\GSC-Features.csv')


# In[11]:



dic=featureset.set_index('img_id').T.to_dict('list')


# In[14]:


full_feat=[]
t=[]
for index,row in samediff_neat.iterrows() :
    feat = []
    tar_inside=[]
    id_A = row['img_id_A']
    id_B = row['img_id_A']
    tar = row['target']
    for i in dic[id_A]:
        feat.append(i)
    for j in dic[id_B]:
        feat.append(j)
    full_feat.append(feat)
    tar_inside.append(tar)
    t.append(tar_inside)
with open("GSC_X_gsc_concat.csv","w+",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(full_feat)   
with open("GSC_t_gsc_concat.csv","w+",newline='') as my:
    csvWriter = csv.writer(my,delimiter=',')
    csvWriter.writerows(t)   


