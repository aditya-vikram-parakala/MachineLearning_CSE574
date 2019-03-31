
# coding: utf-8

# In[213]:


import numpy as np
import pandas as pd
import random 
import csv
from sklearn.utils import shuffle 


# In[214]:


same = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\HumanObserved-Dataset\HumanObserved-Dataset\HumanObserved-Features-Data\same_pairs.csv',usecols=['img_id_A','img_id_B','target'])


# In[215]:


#same.head()


# In[216]:


diff = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\HumanObserved-Dataset\HumanObserved-Dataset\HumanObserved-Features-Data\diffn_pairs.csv')


# In[217]:


#diff.head()


# In[218]:


rindex =  np.array(random.sample(range(len(diff)), 791))

# get 729 random rows from df
dfr = diff.loc[rindex]
samediff = [same,dfr]
samediff_neat = pd.concat(samediff)


# In[219]:


#samediff_neat(usecols=['img_id_A','t'])
#samediff_neat_B=samediff_neat.set_index('img_id_B').to_dict('list')

samediff_neat.head()
samediff_neat = shuffle(samediff_neat)
#samediff_neat.sample(frac=1).reset_index(drop=True)


# In[220]:


featureset = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\HumanObserved-Dataset\HumanObserved-Dataset\HumanObserved-Features-Data\HumanObserved-Features-Data.csv')


# In[221]:



dic=featureset.set_index('img_id').T.to_dict('list')
#print(dic)


# In[222]:


#if(samediff_neat.iat[0,0]==featureset.key):
    #print('success')


# In[223]:


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
with open("humandata_X_hd_concat.csv","w+",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(full_feat)   
with open("humandata_t_hd_concat.csv","w+",newline='') as my:
    csvWriter = csv.writer(my,delimiter=',')
    csvWriter.writerows(t)   


