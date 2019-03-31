
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import random 
import csv
from sklearn.utils import shuffle 


# In[6]:


same = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\HumanObserved-Dataset\HumanObserved-Dataset\HumanObserved-Features-Data\same_pairs.csv',usecols=['img_id_A','img_id_B','target'])


# In[7]:


same.head()


# In[8]:


diff = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\HumanObserved-Dataset\HumanObserved-Dataset\HumanObserved-Features-Data\diffn_pairs.csv')


# In[9]:


diff.head()


# In[10]:


rindex =  np.array(random.sample(range(len(diff)), 791))

# get 729 random rows from df
dfr = diff.loc[rindex]
samediff = [same,dfr]
samediff_neat = pd.concat(samediff)


# In[11]:


#samediff_neat(usecols=['img_id_A','t'])
#samediff_neat_B=samediff_neat.set_index('img_id_B').to_dict('list')

samediff_neat.head()
samediff_neat = shuffle(samediff_neat)
#samediff_neat.sample(frac=1).reset_index(drop=True)


# In[12]:


featureset = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\HumanObserved-Dataset\HumanObserved-Dataset\HumanObserved-Features-Data\HumanObserved-Features-Data.csv')


# In[13]:



dic=featureset.set_index('img_id').T.to_dict('list')
print(dic)


# In[14]:


#if(samediff_neat.iat[0,0]==featureset.key):
    #print('success')


# In[22]:


full_feat=[]
t=[]
for index,row in samediff_neat.iterrows() :
    feat = []
    tar_inside=[]
    id_A = row['img_id_A']
    id_B = row['img_id_B']
    tar = row['target']
    id_A_value = dic[id_A]
    id_B_value = dic[id_B]
    
    
    #for i in dic[id_A]:
    #    feat.append(i)
    #for j in dic[id_B]:
    #   feat.append(j)
    i=0
    while(i<len(id_A_value)):
        subtract=abs(id_A_value[i]-id_B_value[i])
        feat.append(subtract)
        i=i+1
    full_feat.append(feat)
    tar_inside.append(tar)
    t.append(tar_inside)
with open("humandata_X_hd_sub.csv","w+",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(full_feat)   
with open("humandata_t_hd_sub.csv","w+",newline='') as my:
    csvWriter = csv.writer(my,delimiter=',')
    csvWriter.writerows(t)   


