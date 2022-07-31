# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:15:18 2022

@author: Sa2a
"""

import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

directory = "D:DEBI/Uottawa/Data Science Applications/project/ChatBot-Disease-diagnosing/dataset/"
dataset = pd.read_csv(directory+'new_dataset.csv')
print(dataset.head())

print(dataset.describe(include = 'all'))

df_encoded = pd.get_dummies(dataset,prefix="Disease",prefix_sep='|' ,columns = ["Disease"], drop_first=True)





# 1 if x.item() > 0 else x

df_encoded2 = df_encoded.apply(lambda x : x.apply(lambda y: 1 if y > 0 else y) )
# Building the model
frq_items = apriori(df_encoded2, min_support = 0.05, use_colnames = True)
  
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

def contains_disease(frozen):
    contains = False
    for i in frozen:
        if i.startswith("Disease|"):
            return  True
    return contains
    

filtered_rules  =rules["consequents"].apply(contains_disease)
filtered_rules1  =rules["antecedents"].apply(contains_disease)

print(filtered_rules.unique())
print(filtered_rules1.unique())