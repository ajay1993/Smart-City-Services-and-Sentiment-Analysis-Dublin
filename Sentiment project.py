
# coding: utf-8

# In[190]:


import os
from lxml import etree
import re
import sys
f = open('C:/Users/Ajay/outfile2.txt')
lines = f.readlines()
f.close()
from afinn import Afinn
afn = Afinn(emoticons=True)
for sentence in lines:
    print(sentence)
    import nltk
    nltk.download('sentiwordnet')

    from nltk.corpus import sentiwordnet as swn
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords")
    stoplist = str(stopwords.words('english'))
    english_words = []
    nltk.download('wordnet')
    from nltk.corpus import wordnet
#d = enchant.Dict("en_US")
    line= str(sentence)
    print(line)
    for word in line.lower().split():
        if wordnet.synsets(word):
            english_words.append(word)

    " ".join(english_words)
    print (english_words)
    nltk.download('punkt')
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    ps = PorterStemmer()
    texts = ([[word for word in document.lower().split() if word not in stoplist and ps.stem(word)]
             for document in english_words])

    texts = str(list(filter(None, texts)))
    print (texts)
    texts = re.sub('\W+', ' ', str(texts))
    
    print('Predicted Sentiment polarity:', afn.score(texts))
    
    import math
    from textblob import TextBlob
    row = str([cell.encode('utf-8') for cell in sentence])

    from textblob.sentiments import NaiveBayesAnalyzer
    blob = TextBlob(row,analyzer=NaiveBayesAnalyzer())
    blob.sentiment
    aj = blob.sentiment
    print ('aj',aj)
    p_score=0
    n_score=0
    
    for word in texts.lower().split():
        print (word)
        wd = list(swn.senti_synsets(word))[0]
        
#        print('Positive Polarity Score:', wd.pos_score())
#        print('Negative Polarity Score:', wd.neg_score())
#        print('Objective Score:', wd.obj_score())
        p_score += wd.pos_score()
        n_score += wd.neg_score()
    if (p_score == n_score):
        print('neutral')
    elif (abs(p_score) > abs(n_score) ) :
        print('Positive')
    else :
        print('negative')


# In[155]:



from sklearn.feature_extraction.text import CountVectorizer


# In[160]:


import pandas as pd
path='C:/Users/Ajay/Desktop/test.csv'
MD = pd.read_csv(path,encoding = "ISO-8859-1")


# In[161]:


da = MD.data
sen = MD.sen

data = []
data_labels = []
#print(sen)

pdf=MD.filter(items=['data','sen'])
pdf=pdf[pdf['sen']== 1]
pdf=pdf['data']  
positive_list=[]
for pdf in pdf:
    positive_list.append(pdf)
positive=positive_list
positive = str (positive)
for pos in positive.lower().split():  
    data.append(pos) 
    data_labels.append('pos')
    
print (data)    


# In[162]:


ndf=MD.filter(items=['data','sen'])
ndf=ndf[ndf['sen']==2]
ndf=ndf['data']
negative_list=[]
for ndf in ndf:
    negative_list.append(ndf)
negative=negative_list
negative = str(negative)

for neg in negative.lower().split():
    data.append(neg) 
    data_labels.append('neg')
print (data)    


# In[163]:


vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)

#x = v.fit_transform(df['Review'].values.astype('U')) 
data = np.array(data)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage


# In[164]:


from sklearn.cross_validation import train_test_split
 
X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=1234)


# In[165]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[166]:



log_model = log_model.fit(X=X_train, y=y_train)


# In[167]:


y_pred = log_model.predict(X_test)


# In[168]:


import random
j = random.randint(0,len(X_test)-7)
for i in range(j,j+7):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())


# In[169]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[170]:


import pandas as pd
path='C:/Users/Ajay/Desktop/test.csv'
MD = pd.read_csv(path,encoding = "ISO-8859-1")


# In[171]:


da = MD.data
sen = MD.sen

data = []
data_labels = []
#print(sen)

pdf=MD.filter(items=['data','sen'])
pdf=pdf[pdf['sen']== 1]
pdf=pdf['data']  
positive_list=[]
for pdf in pdf:
    data.append(pdf)
    data_labels.append('pos')
#data=positive_list

   
print (data)    


# In[172]:


ndf=MD.filter(items=['data','sen'])
ndf=ndf[ndf['sen']==2]
ndf=ndf['data']
negative_list=[]
for ndf in ndf:
    data.append(ndf)
    data_labels.append('neg')
#data=negative_list
    
print (data)    


# In[173]:


vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)

#x = v.fit_transform(df['Review'].values.astype('U')) 
data = np.array(data)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage


# In[174]:


from sklearn.cross_validation import train_test_split
print(len(features_nd))
print (len(data_labels))
X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.60, 
        random_state=1234)


# In[175]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[176]:


log_model = log_model.fit(X=X_train, y=y_train)


# In[177]:


y_pred = log_model.predict(X_test)


# In[178]:


import random
j = random.randint(0,len(X_test)-7)
for i in range(j,j+7):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())


# In[198]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print (y_test)
print (y_pred)


# In[180]:


Classifierlr = LogisticRegression(solver = 'newton-cg',penalty='l2',multi_class='ovr',C=1, max_iter= 1000, random_state = 0)
Classifierlr.fit(X_train, y_train)


# In[181]:


y_predlr = Classifierlr.predict(X_test) 
accuracy_score(y_test, y_predlr)

#Prediction and Accuracy for Logistic Regression model on training dataset
y_predtr = Classifierlr.predict(X_train)
accuracy_score(y_train, y_predtr)


# In[182]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
precision, recall, fscore, support = score(y_test, y_predlr, average = 'macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[192]:


Classifiersvm = SVC(C= 1, gamma= 0.05, max_iter =1000, kernel='rbf', random_state = 1 )
Classifiersvm.fit(X_train, y_train)

#Grid Search for parameters for SVM model
"""
from sklearn.model_selection import GridSearchCV
parameters = ({'kernel':['linear'], 'C':[1,2,3,4,5,6,10,100,1000]}, {'kernel':['rbf'],'C':[1,2,3,4,5,6,7,8,9,11,12,13,14,15,10,100],'gamma':[0.04,0.05,0.06,0.07,0.03,0.02,0.01,0.6,1,2]})
gridsearch = GridSearchCV( estimator = Classifiersvm, param_grid = parameters, scoring = 'accuracy', cv = 10)
gridsearch = gridsearch.fit(x_train, y_train)
bestacc = gridsearch.best_score_
bestpara = gridsearch.best_params_
"""

#Prediction and Accuracy for SVM model using test dataset 
y_predsv = Classifiersvm.predict(X_test)
print(accuracy_score(y_test, y_predsv))

#Prediction and Accuracy for SVM model using training dataset
y_predsvt = Classifiersvm.predict(X_train)
print (accuracy_score(y_train, y_predsvt))

#Evaluation metrics for SVM model
precision, recall, fscore, support = score(y_test, y_predsv,average = 'macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

#Confusion Matrix for SVM model
cmsvm = confusion_matrix(y_test, y_predsv)


# In[194]:


"""
NAIVE BAYES MODEL
"""
#Implementing Naive Bayes model
Classifiernb = GaussianNB()
Classifiernb.fit(X_train, y_train)

#Prediction and Accuracy for Naive Bayes model using test dataset
y_prednb = Classifiernb.predict(X_test)
print(accuracy_score(y_test, y_prednb))

#Prediction and Accuracy for Naive Bayes model using training dataset
y_prednbt = Classifiernb.predict(X_train)
print(accuracy_score(y_train, y_prednbt))

#Evaluation metrics for Naive Bayes model
precision, recall, fscore, support = score(y_test, y_prednb,average = 'macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

#Confusion matrix for Naive Bayes model
cmnb = confusion_matrix(y_test, y_prednb)


# In[195]:



"""
RANDOM FOREST MODEL
"""
#Implementing Random Forest model
Classifierrf = RandomForestClassifier()
Classifierrf.fit(X_train, y_train)

#Prediction and Accuracy for Random Forest model using test dataset
y_predrf = Classifierrf.predict(X_test)
print(accuracy_score(y_test, y_predrf))

#Prediction and Accuracy for Random Forest model using training dataset
y_predrft = Classifierrf.predict(X_train)
accuracy_score(y_train, y_predrft)

#Evaluation metrics for Random Forest model
precision, recall, fscore, support = score(y_test, y_predrf,average = 'macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

#Confusion matrix for Random Forest model
cmdt = confusion_matrix(y_test, y_predrf)


# In[196]:



#DECISION TREE CLASSIFICATION MODEL
#Implementing Decision Tree model
Classifierdt = DecisionTreeClassifier()
Classifierdt.fit(X_train, y_train)

#Prediction and Accuracy for Decision Tree model using test dataset
y_preddt = Classifierdt.predict(X_test)
print(accuracy_score(y_test, y_preddt))

#Prediction and Accuracy for Decision Tree model using training dataset
y_preddtt = Classifierdt.predict(X_train)
accuracy_score(y_train, y_preddtt)

#Evaluation metrics for Decision Tree model
precision, recall, fscore, support = score(y_test, y_preddt,average = 'macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

#Confusion matrix for Decision Tree model
cmdt = confusion_matrix(y_test, y_preddt)


"""
ENSEMBLE APPROACH USING VOTING CLASSIFIER
"""
#Implementing Ensemble model using Voting classifier
ensemble = VotingClassifier(estimators=[('rf',Classifierrf),('dt',Classifierdt),('lr', Classifierlr), ('svm', Classifiersvm), ('gnb', Classifiernb)], voting='hard')
ensemble = ensemble.fit(X_train, y_train)

#Prediction and Accuracy for Ensemble model using test dataset
y_predensemble = ensemble.predict(X_test)
print(accuracy_score(y_test, y_predensemble))

#Prediction and Accuracy for Ensemble model using training dataset
y_predensemblet = ensemble.predict(X_train)
accuracy_score(y_train, y_predensemblet)

#Evaluation metrics for Ensemble model
precision, recall, fscore, support = score(y_test, y_predensemble,average = 'macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

#Confusion matrix for Ensemble model
cm_en = confusion_matrix(y_test, y_predensemble)

#Applying K fold for validation
folds = cross_val_score(ensemble, X_train, y_train, cv=3)
print(folds)
print (folds.mean())

print (folds.std())


# In[207]:


import glob
from textblob import TextBlob

itr = 0
ovraccur = 0
f = open('positive-vader-blob-tag.txt', 'w',encoding='utf8')
n = open('negative-vader-blob-tag.txt', 'w',encoding='utf8')

import pandas as pd
path='C:/Users/Ajay/Desktop/test.csv'
MD = pd.read_csv(path,encoding = "ISO-8859-1")


da = MD.data
sen = MD.sen

data = []
data_labels = []
#print(sen)

pdf=MD.filter(items=['data','sen'])
pdf=pdf[pdf['sen']== 1]
pdf=pdf['data']  

for pdf in pdf:
    f.write(pdf)
    f.write(',')
    
pdf=MD.filter(items=['data','sen'])
pdf=pdf[pdf['sen']== 2]
pdf=pdf['data']  

for pdf in pdf:
    n.write(pdf)  
    n.write(',') 


f.close()
n.close()
neg_count = 0
neg_correct = 0
pos_count = 0
pos_correct = 0

with open("positive-vader-blob-tag.txt","r",encoding='utf8') as f:
        for line in f.read().split(','):
            analysis = TextBlob(line)
            
            if analysis.sentiment.subjectivity > 0:
                if analysis.sentiment.polarity > 0.9:
                    pos_correct += 1
                pos_count +=1
with open("negative-vader-blob-tag.txt","r",encoding='utf8') as f:
        for line in f.read().split(','):
            analysis = TextBlob(line)
            if analysis.sentiment.subjectivity > 0:
                if analysis.sentiment.polarity <= 0.9:
                    neg_correct += 1
                neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))  
ovr_cor= (pos_correct+neg_correct)
ovr_count= neg_count+pos_count
accur = ovr_cor/ovr_count*100.0
print("Overall accuracy = ", accur)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer= SentimentIntensityAnalyzer()
pos_count = 0
pos_correct = 0
neg_count = 0
neg_correct = 0

with open("positive-vader-blob-tag.txt", "r",encoding='utf8') as f:
    for line in f.read().split(','):
        vs = analyzer.polarity_scores(line)
        if not vs['neg'] > 0.1:
            if vs['pos']-vs['neg'] > 0:
                pos_correct += 1
            pos_count +=1

with open("negative-vader-blob-tag.txt","r",encoding='utf8') as f:
    for line in f.read().split(','):
        vs = analyzer.polarity_scores(line)
        if not vs['pos'] > 0.1:
            if vs['pos']-vs['neg'] <= 0:
                neg_correct += 1
            neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

ovr_cor= (pos_correct+neg_correct)
ovr_count= neg_count+pos_count
accur = ovr_cor/ovr_count*100.0
print("Overall accuracy = ", accur)
        

