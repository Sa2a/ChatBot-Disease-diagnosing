# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 10:24:23 2022

@author: river
"""

import pandas as pd
import numpy as np

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

def get_antecedents_consequents(df_encoded):
    cols = df_encoded.columns.values
    mask = df_encoded.gt(0.0).values
    out = [cols[x].tolist() for x in mask]
    return out

def get_my_rules(inputset,inputset_no, itemsets):
    if bool(inputset_no):
        return itemsets.apply(lambda item: item if set(inputset).issubset(set(item)) and not set(inputset_no).issubset(set(item)) else None)
    return itemsets.apply(lambda item: item if set(inputset).issubset(set(item)) else None)


directory = "D:DEBI/Uottawa/Data Science Applications/project/ChatBot-Disease-diagnosing/dataset/"
df_encoded = pd.read_csv(directory+'dataset_encoded.csv')
rules = pd.read_csv(directory+'rules.csv')
rules_detais = rules.describe(include = 'all')

sum_transactions = sum_mean_t(df_encoded)
lift_threshold = min(rules.lift) if min(rules.lift)>1 else 1

from ast import literal_eval


rules.itemset= rules.itemset.apply(literal_eval)

inputset = ["high_fever"]
inputset_no = ["extra_marital_contacts"]
inputset_no=[]

def get_disease(inputset,inputset_no = []):
    max_lift = 0    
    my_rules = rules.copy()
    my_rules.itemset = get_my_rules(inputset, inputset_no, rules.itemset)
    my_rules.drop(my_rules[~pd.notna(my_rules.itemset)].index, inplace = True)
    my_rules.sort_values(by=['lift'], ascending=[False],inplace= True)
    my_rules.reset_index(drop=True, inplace=True)
    
    remaining_symptoms = set(my_rules.itemset[0][:-1]).difference(set(inputset))
    potential_disease = my_rules.Disease.unique()
    
    # np.concatenate(np.array(['Disease|']*len(potential_disease)),potential_disease)
    
    max_lift = 0
    the_one = ""
    # best_c = 0
    # best_s = 0
    for x in potential_disease:
        d =["Disease|"+x]
        newInputset = np.concatenate((inputset,d))
        # s = sup(df_encoded , newInputset, sum_transactions)
        # c = conf(df_encoded , newInputset, sum_transactions)
        l = lift(df_encoded , newInputset, sum_transactions)
        if(l>max_lift):
            max_lift = l
            the_one = x
            # best_c= c 
            # best_s = s
            
    return (max_lift, the_one, remaining_symptoms, potential_disease)


print(get_disease(inputset))




# print("disease:",the_one)
# print("Lift:",max_lift)
# print("confidence:",best_c)  
# print("weighted support:",best_s)  

# if len(potential_disease)>1:
#     print("please refer to a specialized Doctor. You have a potential to have ",potential_disease)
