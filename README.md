<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/FxL5qM0.jpg" alt="Bot logo"></a>
</p>

<h1 align="center">Dr.Bot</h1>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/uOttawa)

</div>

---

<p align="center"> 🤖 Few lines describing what your bot does.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Demo / Working](#demo)
- [How it works](#working)
- [Getting Started](#getting_started)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

Chatbots in the healthcare provides either predictive diagnosis or other assistantships like booking an appointment.

Help doctors and medical staff to deal with patients in better way and reduce their efforts with the help of technology. Chatbots can play a major role in reshaping the healthcare industry by providing either predictive diagnosis or any other assistantships like booking an appointment.


## 🎥 Demo / Working <a name = "demo"></a>
https://bot.dialogflow.com/0e9384e4-68e4-4807-948e-8d1deba06185

![image](https://user-images.githubusercontent.com/34524576/182043459-935c0044-7906-444a-8a81-4dead30dc72f.png)

## The bot succesfully recognized the symptoms and predicted the disease which is "Typhoid"
![image](https://user-images.githubusercontent.com/34524576/182044503-15ab83d4-ff2e-4027-ae24-dd4ab372ca64.png)

## 💭 How it works <a name = "working"></a>

1- DialogFlow integration to provide the interactive interface with the pation through many sourcces like telegrame or through a webage.<br>
2- The conversation is sent to the python code in this repo<br>
3- Extraxt the symptoms from the patents text and map it to our predefined symptoms using syntactic and semantic similarity.<br>

## Now we have a set of the patent's symptoms.<br>
## And we have Three different ML algorithms to predict the potential disease:<br>
* Clasifiations<br>
* Clustering<br>
* Weighted Association rules<br>

Clasification and clustering hase limitation, It just tells the associated disease class. In our situation, we need to dig into the details of the disease and the associated symptoms.
We want to know whether the provided details are sufficient or if we need more details and what are the required details to ask back the patent.<br>

We thought of AR as a way to solve the limitations of the classification adn clustering algorithms.To formulate the rules we considered the symptoms as the antecedents and Diseases as consequents. Each symptom has a severity value. We want to make use of that in evaluating the rules. We’ve found a paper that was talking about weighted association rules and we’ve used it in evaluating the weighted support. https://eprints.soton.ac.uk/257986/1/331.tao.pdf

## Using the Weighted Association rules:
* In order to know whether the input symptoms are sufficient or not. We’ve evaluated the lift and compared it with our lift threshold which has been set to the minimum lift value of our predefined rules. Approximately 4.

* If the input symptoms set is evaluated to be lower than our threshold.
We search for the most important rule which has the highest lift and its item set contains the input symptoms and asks the patent again whether or not he has any of the remaining symptoms from that rule.

After reaching a high level of confidence we respond back to user with the potential disease with its description and precautions.


## 🏁 Getting Started <a name = "getting_started"></a>

## The main functionalities

### Prerequisites

install important python liberaries 
```
pip install pandas
pip install numpy
pip install nltk
pip install spacy
```

## After getting the text from the user
* Syntactic Similarity<br>
Calculate Jaccard Similarity to identify the similarities between sentences<br>
The syntactic similarity is based on the assumption that the similarity between the two texts is proportional to the number of identical words in them (appropriate measures can be adopted here to ensure that the method does not become biased towards the text with a larger word count, 
the syntactic similarity value can be obtained by constructing measures around the word count of the two documents,

file: input_text_process/healthcarechatbot
```
def syntactic_similarity(symp_t,corpus):
    most_sim=[]
    poss_sym=[]
    for symp in corpus:
        d=jaccard_set(symp_t,symp)
        most_sim.append(d)
    order=np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(corpus[i]) :
            return 1,[corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i]!=0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1,poss_sym
    else: return 0,None
```

* Symantic Similarity<br>
semantic similarity focuses more on the meaning and interpretation-based similarity between the two texts.<br>
file: input_text_process/healthcarechatbot
```
def semanticD(doc1,doc2):
    doc1_p=preprocess(doc1).split(' ')
    doc2_p=preprocess_sym(doc2).split(' ')
    score=0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1,doc1)
            syn2 = WSD(tock2,doc2)
            if syn1 is not None and syn2 is not None :
                x=syn1.path_similarity(syn2)
                if x is not None and x>0.25:
                    score+=x
    return score/(len(doc1_p)*len(doc2_p))
```
```
def suggest_syn(sym):
    symp=[]
    synonyms = wordnet.synsets(sym)
    lemmas=[word.lemma_names() for word in synonyms]
    lemmas = list(set(chain(*lemmas)))
    for e in lemmas:
        res,sym1=semantic_similarity(e,all_symp_pr)
        if res!=0:
            symp.append(sym1)
    return list(set(symp))

suggest_syn('puke')
```
```
output: ['vomit','pain','swollen blood vessel', 'dark urine']
```
## AR evaluate support, conf, lift
```
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
```

## Get disease<br>
evaluate the input symptoms and get the maximum lift with its associated disease and the ramaining symptoms to ask the paten back, and the set of potential diseases.
```
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
inputset = ["high_fever","extra_marital_contacts"]
print(get_disease(inputset,inputset_no=[]))
```
![image](https://user-images.githubusercontent.com/34524576/182044057-1cf5e604-35f9-4288-866d-c223b7e5300d.png)



## ✍️ Authors <a name = "authors"></a>

- [@Sa2a]
- [@Hadeer]
- [@NadaAbdellatif]
- [@NadaMontaser]

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
