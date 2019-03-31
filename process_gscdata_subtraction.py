
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random 
import csv
from sklearn.utils import shuffle 


# In[3]:


same = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\GSC-Dataset(1)\GSC-Dataset\GSC-Features-Data\same_pairs.csv',usecols=['img_id_A','img_id_B','target'])


# In[4]:





# In[5]:


diff = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\GSC-Dataset(1)\GSC-Dataset\GSC-Features-Data\diffn_pairs.csv')


# In[6]:





# In[7]:


rindex =  np.array(random.sample(range(len(diff)),71531))
dfr = diff.loc[rindex]
samediff = [same,dfr]
samediff_neat = pd.concat(samediff)


# In[8]:


samediff_neat.head()
samediff_neat = shuffle(samediff_neat)


# In[9]:


featureset = pd.read_csv(r'C:\Users\aditya vikram\Desktop\ASSIGNMENTS\ML\GSC-Dataset(1)\GSC-Dataset\GSC-Features-Data\GSC-Features.csv')


# In[10]:


dic=featureset.set_index('img_id').T.to_dict('list')


# In[12]:


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
    
    
    i=0
    while(i<len(id_A_value)):
        subtract=abs(id_A_value[i]-id_B_value[i])
        feat.append(subtract)
        i=i+1
    full_feat.append(feat)
    tar_inside.append(tar)
    t.append(tar_inside)
with open("GSC_X_gsc_sub.csv","w+",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(full_feat)   
with open("GSC_t_gsc_sub.csv","w+",newline='') as my:
    csvWriter = csv.writer(my,delimiter=',')
    csvWriter.writerows(t)   


