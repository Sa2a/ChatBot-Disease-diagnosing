# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 01:33:46 2022

@author: Sa2a
"""

import pandas as pd
import numpy as np
import seaborn as sns


directory = "D:DEBI/Uottawa/Data Science Applications/project/ChatBot-Disease-diagnosing/dataset/"
dataset = pd.read_csv(directory+'dataset.csv')
print(dataset.head())
#df.describe()
weight_data = pd.read_csv(directory+'Symptom-severity.csv')
print(weight_data.head())

#cleaning

cols = dataset.columns
data = dataset[cols].values.flatten()

s = pd.Series(data)
#s = s.str.strip()

# remove whitespace
s = s.str.replace(' ','')
s = s.values.reshape(dataset.shape)

df = pd.DataFrame(s, columns=dataset.columns)

df = df.fillna(0)
df.head()
print("len before", len(df))
df_dublicated = df.duplicated()
df.drop(df.index[np.where(df_dublicated == True)], inplace= True)
df.reset_index(drop=True,inplace= True)

print("len after", len(df))
sns.countplot(data = df, x = "Disease")

df_encoded = pd.get_dummies(df, columns = ["Disease"], drop_first=True)



weight_data['Symptom'] = weight_data['Symptom'].apply(lambda x: x.replace(' ',''))
weight_data.to_csv(directory+"Symptom-severity.csv",index= False)
# for dublicated values remove
sns.countplot(x="Disease",data=df)
sns.countplot(x="Symptom",data=weight_data)
is_duplicate = weight_data['Symptom'].duplicated()
weight_data.drop(weight_data.index[np.where(is_duplicate == True)], inplace= True)


symptoms = weight_data['Symptom'].unique()

 #Encoding the the symptoms with their severity weight
new_dataset = pd.DataFrame(columns=(np.concatenate((['Disease' ],symptoms),axis=0)))
new_dataset.Disease= df.Disease

#vals = df.values

for r in range(len(df.Disease)):
    for c in range(1,len(df.columns)):
        old_symp = df.iloc[r,c]
        if old_symp == 0:
            continue
        new_dataset.loc[r,old_symp] = weight_data[weight_data['Symptom'] == old_symp]['weight'].values[0]
        
new_dataset = new_dataset.fillna(0)
new_dataset.head()

new_dataset.to_csv(directory+"new_dataset.csv",index= False)

un, cou = np.unique(df.Disease,return_counts=True)


