# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:56:11 2022

@author: river
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
directory = "D:DEBI/Uottawa/Data Science Applications/project/ChatBot-Disease-diagnosing/dataset/"
dataset = pd.read_csv(directory+'new_dataset.csv')
print(dataset.head())

print(dataset.describe(include = 'all'))


weight_map = {
    "A" :.85,
"B" : 1,
"C": .85,
"D" : 1.55,
"E" : 1,
}

weight_map = {
    "A" : 1,
"B" : 1,
"C": 5,
"D" : 1,
"E" : 4,
}


test_df = pd.read_csv(directory+'weighted_assosiation_test.csv',header= None)
test_df = test_df.fillna(0)

df_new  = pd.DataFrame(columns=(["A","B", "C","D", "E"]))
new_r = 0
for i in range(len(test_df)):
    for c in range(len(test_df.columns)):
        col = test_df.iloc[i,c]
        
        if col == 0 :
            continue
        print(col)
        df_new.loc[i,col] = weight_map[col]
#new_df = new_df.fillna(0)

#new_df.replace(0, np.nan, inplace=True)


'''      
itemset = ["A","B","C","D","E"]


ave_trans_w = []
for i in range(len(new_df)):
    ave_trans_w.append(np.nanmean(new_df.iloc[i,:]))

sum_trans_w = np.sum(ave_trans_w)
print("all w = ",sum_trans_w)

itemset = ["A","B","C","D","E"]

contained_sets= []
for i in range(len(new_df)):
    #print( df_encoded.iloc[i,:])
    mean = cotains_items(itemset, new_df.iloc[i,:])
    
    #if not np.isnan(mean):
    #  mean /=sum_all_wights
    contained_sets.append(mean)
    
print(np.nansum(contained_sets)/sum_trans_w)
'''
def cotains_items(items, row):
    for i in items:
        if np.isnan(row[i]):
             return np.nan
    return np.nanmean(row)
def sum_mean_t(new_df):
    sum_trans_w = 0
    for i in range(len(new_df)):
        sum_trans_w+= np.nanmean(new_df.iloc[i,:])
    return sum_trans_w

def sup(new_df, itemset , sum_trans_w):
    contained_sets= []
    for i in range(len(new_df)):
        #print( df_encoded.iloc[i,:])
        mean = cotains_items(itemset, new_df.iloc[i,:])
        
        #if not np.isnan(mean):
        #  mean /=sum_all_wights
        contained_sets.append(mean)
        
    return np.nansum(contained_sets)/sum_trans_w

def conf(df_encoded , itemset, sum_transactions):
    s_A_B = sup(df_encoded , itemset, sum_transactions)
    s_A = sup(df_encoded , itemset[:-1], sum_transactions)
    return s_A_B/s_A


def lift(df_encoded , itemset, sum_transactions):
    s_A_B = sup(df_encoded , itemset, sum_transactions)
    s_A = sup(df_encoded , itemset[:-1], sum_transactions)
    s_B = sup(df_encoded , itemset[-1:], sum_transactions)
    return s_A_B/(s_A*s_B)

sum_t = sum_mean_t(df_new)
items = ["A"]
s = sup(df_new , items, sum_t)
s


from itertools import combinations

directory = "D:DEBI/Uottawa/Data Science Applications/project/ChatBot-Disease-diagnosing/dataset/"
dataset = pd.read_csv(directory+'new_dataset.csv')

df_encoded = pd.get_dummies(dataset,prefix="Disease",prefix_sep='|' ,columns = ["Disease"])
for c in my_c:
    df_encoded[c]=df_encoded[c].apply(lambda x: .1 if x>0 else x)
    

df_encoded.replace(0, np.nan, inplace=True)

def get_antecedents_consequents(df_encoded):
    cols = df_encoded.columns.values
    mask = df_encoded.gt(0.0).values
    out = [cols[x].tolist() for x in mask]
    
    return out
itemsets = get_antecedents_consequents(df_encoded)


sum_transactions = sum_mean_t(df_encoded)
num_items =[]
diseases =[]
suports = []
confidences = []
lifts = []
for itemset in itemsets:
    suports.append(sup(df_encoded , itemset, sum_transactions))
    confidences.append(conf(df_encoded , itemset, sum_transactions))
    lifts.append(lift(df_encoded , itemset, sum_transactions))
    num_items.append(len(itemset))
    diseases.append(itemset[-1].split('|')[1])



lift_threshold = min(lifts) if min(lifts)>1 else 1

rules = pd.DataFrame({"itemset":itemsets,"Disease":diseases,"support": suports,"confidence": confidences,"lift": lifts,"ItemSetLen":num_items})
rules_detais = rules.describe(include = 'all')

def get_my_rules2(inputset,inputset_no, itemsets):
    if bool(inputset_no):
        return itemsets.apply(lambda item: item if set(inputset).issubset(set(item)) and not set(inputset_no).issubset(set(item)) else None)
    return itemsets.apply(lambda item: item if set(inputset).issubset(set(item)) else None)
# inputset = ["itching","skin_rash", "dischromic_patches"]
inputset = ["high_fever"]
inputset_no = ["extra_marital_contacts"]
inputset_no=[]
max_lift = 0
# while(max_lift < lift_threshold):
    

my_rules = rules.copy()
my_rules.itemset = get_my_rules2(inputset, inputset_no, rules.itemset)

my_rules.drop(my_rules[~pd.notna(my_rules.itemset)].index, inplace = True)

my_rules.sort_values(by=['ItemSetLen','lift'], ascending=[True,False],inplace= True)

my_rules.reset_index(drop=True, inplace=True)

remaining_symptoms = set(my_rules.itemset[0][:-1]).difference(set(inputset))

potential_disease = my_rules.Disease.unique()

# np.concatenate(np.array(['Disease|']*len(potential_disease)),potential_disease)

max_lift = 0
the_one = ""
best_c = 0
best_s = 0
for x in potential_disease:
    d =["Disease|"+x]
    newInputset = np.concatenate((inputset,d))
    s = sup(df_encoded , newInputset, sum_transactions)
    c = conf(df_encoded , newInputset, sum_transactions)
    l = lift(df_encoded , newInputset, sum_transactions)
    if(l>max_lift):
        max_lift = l
        the_one = x
        best_c= c 
        best_s = s
            

print("disease:",the_one)
print("Lift:",max_lift)
print("confidence:",best_c)  
print("weighted support:",best_s)  






# l1 = [1,2,3]
# l2 = [1,2]
# l3 = set(l1).difference(set(l2))


# antecedents= 
# consequents= []
# for i in range(1,len(weight_map)):
#     comb = combinations(list(weight_map.keys()), i+1)
#     print(list(comb))

# col = df_encoded.columns
# my_c = []
# for c in col.values:
#     my_c.append(c) if c.startswith("Disease|") else None

# uniq = np.unique(dataset.Disease)


# np.where(col == "Disase|AIDS")