import os
from lxml import etree
import re

# path = 'C:/Users/Ajay/Downloads/1998/posts/000/000/'
path = 'C:/Users/Ajay/Downloads/2008/posts/000/'
os.listdir(path)
outfile = open('outfile1.txt', 'w',encoding='utf8')
for filename in os.listdir(path):
    fulllist = os.path.join(path, filename)
    for filename1 in os.listdir(fulllist):
        fullname = os.path.join(fulllist, filename1)
        tree = etree.parse(fullname)
        root = tree.getroot()
        root.tag
        import sys

        for child in root:
            for child in (child):
                if child.tag[-7:] == "content":
                    name = (child.text)
                    name = re.sub('\W+', ' ', str(name))
                    name += '\n'
                    outfile.write(name)

outfile.close()
from gensim import corpora
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# stoplist=set('for a of the and to in that the them this we is was where were as can could be on [] ; ''  me you \n you they'.split())
f = open('outfile1.txt',encoding='utf8')
lines = f.readlines()
f.close()
print(lines)
keylist = set(
    'city public transport buses dublin train luas tram late speed road highway m50 airport flights timing terminal1 terminal2 facilities cancellation weather storm education school college university pre primary primary cost free standard area rent morgage interest downpayment bank student hous crisis satisfaction farm fish potato cow bread sheep shop mall industries top10 booming financee tax it information software insurance event election government protest EU Europe ireland electricity garbage power accident live tracking development poverty economy deaths'.split())
import nltk

nltk.download("stopwords")
stoplist = str(stopwords.words('english'))
a = 0
outfile_opt = open('outfile2.txt', 'w',encoding='utf8')
for sentence in lines:
    for word in keylist:
        if word in sentence:
            a = 1
    if a == 1:
        sentence += '\n'
        outfile_opt.write(sentence)
outfile_opt.close()

f = open('outfile2.txt',encoding='utf8')
lines = f.readlines()
f.close()
print(lines)
english_words = []
# import enchant
nltk.download('wordnet')
from nltk.corpus import wordnet

# d = enchant.Dict("en_US")
line = str(lines)
for word in line.lower().split():
    if wordnet.synsets(word):
        english_words.append(word)

" ".join(english_words)
print(english_words)
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
texts = ([[word for word in document.lower().split() if word not in stoplist and ps.stem(word)]
          for document in english_words])
texts = list(filter(None, texts))
from pprint import pprint

pprint(texts)
dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')  # store the dictionary, for future reference
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
corpora.MmCorpus.serialize('deerwester.mm', corpus)  # store to disk, for later use
from gensim import corpora, models, similarities

a = corpus;
print(dictionary.token2id)

import logging, gensim

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, update_every=0, chunksize=1000,
                                      passes=10)
lda.print_topics(4)
print(dictionary[2])
print(lda[corpus[0]])
from gensim.corpora import HashDictionary

dct = HashDictionary(texts)
import pandas as pd
import numpy as np

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from afinn import Afinn
afn = Afinn(emoticons=True)
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


sid = SentimentIntensityAnalyzer()

k = 0
i = 0

for sentence in lines:
    print(sentence)
    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    for text in sentence.split():
        # print(text)
        i = 0
        # print (len(dictionary))
        l = int(len(dictionary))
        print('len', l)
        #        print (corpus[1115])
        #        print (dictionary[1115])
        while i < l:

            a = corpus[i]
            # print ('hi',i,a)
            b = (np.array(a)[:, 0])
            # b = list(map(int, b))
            # lengthOfList = len(b)
            # sum = 0
            # firstDigit = b[0]
            # lastDigit = b[-1]
            # sum = firstDigit + lastDigit
            # b = sum
            b = int(b)
            print('b', b)
            # print (dictionary[int(b)])
            doc = str(dictionary[int(b)])
            print('doc', doc)
            print('text', text)
            text = text.lower()
            doc = doc.lower()

            if text.find(doc) != -1:

                a = lda[corpus[i]]
                print('a', a)
                b = (np.array(a)[:, 1])
                # print (b)

                j = 0
                num_top = int(lda.num_topics)
                # print (num_top)
                value = 0
                while (j < (num_top)):
                    c = np.array(b)[j]
                    c = float(c)
                    print(j)
                    print('c', c)
                    print('value', value)
                    if c > value:
                        value = c
                        topic = j
                    j += 1
                print(topic)
                if topic == 0:
                    i0 = i0 + 1
                elif topic == 1:
                    i1 = i1 + 1
                elif topic == 2:
                    i2 = i2 + 1
                else:
                    i3 = i3 + 1
                print(i0, i1, i2, i3)
                break
            i += 1
    ss = sid.polarity_scores(str(sentence))
    ss = str(ss)

    afns = afn.score(str(sentence))

    textbs = TextBlob(str(sentence))

    txtb = textbs.sentiment.polarity
    txtbsub = textbs.sentiment.subjectivity

    print(ss)

    if i0 > i1 and i0 > i2 and i0 > i3:
        print('Topic 0')
        d = {'sentence ': k, 'topic': ['Topic 0'], 'Sentiment-vader': ss, 'Sentiment-affin': afns, 'Sentiment-blob':  txtb,'Sentiment-blobsub':  txtbsub,  'Sent-Text': [sentence]}
        data_frame = pd.DataFrame(data=d)
        data_frame.to_csv('topic-Sentt-4.csv', sep=',', mode='a', header=False)

    elif i1 > i2 and i1 > i3:
        print('Topic 1')
        d = {'sentence ': k, 'topic': ['Topic 1'], 'Sentiment-vader': ss, 'Sentiment-affin': afns, 'Sentiment-blob':  txtb,'Sentiment-blobsub':  txtbsub, 'Sent-Text': [sentence]}
        data_frame = pd.DataFrame(data=d)
        data_frame.to_csv('topic-Sentt-4.csv', sep=',', mode='a', header=False)

    elif i2 > i3:
        print('Topic 2')
        d = {'sentence ': k, 'topic': ['Topic 2'], 'Sentiment-vader': ss, 'Sentiment-affin': afns, 'Sentiment-blob':  txtb,'Sentiment-blobsub':  txtbsub, 'Sent-Text': [sentence]}
        data_frame = pd.DataFrame(data=d)
        data_frame.to_csv('topic-Sentt-4.csv', sep=',', mode='a', header=False)

    else:

        print('Topic 3')
        d = {'sentence ': k, 'topic': ['Topic 3'], 'Sentiment-vader': ss, 'Sentiment-affin': afns, 'Sentiment-blob':  txtb, 'Sentiment-blobsub':  txtbsub,'Sent-Text': [sentence]}
        data_frame = pd.DataFrame(data=d)
        data_frame.to_csv('topic-Sentt-4.csv', sep=',', mode='a', header=False)

    k = k + 1

    # line=lines.encode('utf-8')

    print('\n')


df = pd.read_csv('topic-Sentt-4.csv', header=None, index_col=None)
df.columns = ['','Number','Topic Number' ,'Vader-sentiment' ,'Affin-sentiment' ,'TextBlob-Sentiment','TextBlob-Subjectivity','Sentence']
df.to_csv('topic-Sentt-4.csv', index=False)

lda[corpus[21]]
print(corpus[21])
i = 0
a = []
import pandas as pd
import numpy as np

print(len(dictionary))
while i < len(dictionary):
    # print (lda[corpus[i]])
    value = 0
    j = 0
    c = 0
    a = lda[corpus[i]]
    print(a)
    i += 1
    b = (np.array(a)[:, 1])
    print(b)
    num_top = int(lda.num_topics)
    print(num_top)
    value = 0
    while (j < (num_top)):
        c = float(np.array(b)[j])
        print(j)
        print(c)
        if c > value:
            value = c
            topic = j
        j += 1
    print('i', i)
    cor = str(corpus[i])
    print('value=', value)
    d = {'corpus': [cor], 'value': value, 'topic': topic}
    data_frame = pd.DataFrame(data=d)
    # pd.DataFrame({'corpus':corpus[i], 'value':value,'topic':topic})
    data_frame.to_csv('topic-4.csv', sep=',', mode='a', header=False)

print(dictionary.token2id)
import numpy as np

for id in corpus:
    print(id)
#    print (np.array(doc)[:,0])
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=8, update_every=0, chunksize=1000,
                                      passes=10)
lda.print_topics(-1)
print(lda[corpus[0]])
from gensim.corpora import HashDictionary

dct = HashDictionary(texts)
import pandas as pd

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

k = 0;
for sentence in lines:
    print(sentence)
    it = [0] * lda.num_topics
    for text in sentence.split():
        # print(text)
        i = 0
        # print (len(dictionary))
        l = int(len(dictionary))

        while i < l:

            a = corpus[i]
            # print ('hi',i,a)
            b = (np.array(a)[:, 0])
            # b = list(map(int, b))
            # lengthOfList = len(b)
            # sum = 0
            # firstDigit = b[0]
            # lastDigit = b[-1]
            # sum = firstDigit + lastDigit
            # b = sum
            b = int(b)
            # print (b)
            # print (dictionary[int(b)])
            doc = str(dictionary[int(b)])
            # print ('doc',doc)
            # print ('text',text)
            text = text.lower()
            doc = doc.lower()

            if text.find(doc) != -1:

                a = lda[corpus[i]]
                print(a)
                b = (np.array(a)[:, 1])
                # print (b)

                j = 0
                num_top = int(lda.num_topics)
                # print (num_top)
                value = 0
                while (j < (num_top)):
                    c = np.array(b)[j]
                    c = float(c)
                    print(j)
                    print('c', c)
                    print('value', value)
                    if c > value:
                        value = c
                        topic = j
                    j += 1
                print(topic)
                it[topic] = it[topic] + 1
            i += 1
    ss = sid.polarity_scores(str(sentence))

    afns = afn.score(str(sentence))

    textbs = TextBlob(str(sentence))

    txtb = textbs.sentiment.polarity
    txtbsub = textbs.sentiment.subjectivity


    ss = str(ss)
    print(ss)
    print(it)
    m = int(max(it))
    print('max', m)
    top = [itr for itr, jtr in enumerate(it) if jtr == m]
    print('top', top)
    if len(top) > 1:
        topic = 'neutral'
    else:
        topic = 'Topic' + str(top)
    d = {'sentence ': k, 'topic': [topic], 'Sentiment-vader': ss, 'Sentiment-affin': afns, 'Sentiment-blob':  txtb, 'Sentiment-blobsub':  txtbsub,'Sent-Text': [sentence]}
    data_frame = pd.DataFrame(data=d)
    data_frame.to_csv('topic-Sent-8.csv', sep=',', mode='a', header=False)

    #    if i0>i1 and i0>i2 and i0>i3 :
    #        print('Topic 0')
    #        d={'sentence ':k, 'topic':['Topic 0'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #    elif  i1>i2 and i1>i3 :
    #        print('Topic 1')
    #        d={'sentence ':k, 'topic':['Topic 1'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #
    #    elif  i2>i3 :
    #        print('Topic 2')
    #        d={'sentence ':k, 'topic':['Topic 2'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #
    #    else :
    #
    #        print ('Topic 3')
    #        d={'sentence ':k, 'topic':['Topic 3'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #
    k = k + 1

# line=lines.encode('utf-8')


#    print ('\n')

i = 0
a = []
import pandas as pd
import numpy as np

print(len(dictionary))
while i < len(dictionary):
    # print (lda[corpus[i]])
    value = 0
    j = 0
    c = 0
    a = lda[corpus[i]]
    print(a)
    i += 1
    b = (np.array(a)[:, 1])
    print(b)
    num_top = int(lda.num_topics)
    print(num_top)
    value = 0
    while (j < (num_top)):
        c = float(np.array(b)[j])
        print(j)
        print(c)
        if c > value:
            value = c
            topic = j
        j += 1
    print('i', i)
    cor = str(corpus[i])
    print('value=', value)
    d = {'corpus': [cor], 'value': value, 'topic': topic}
    data_frame = pd.DataFrame(data=d)
    # pd.DataFrame({'corpus':corpus[i], 'value':value,'topic':topic})
    data_frame.to_csv('topic-8.csv', sep=',', mode='a', header=False)

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=12, update_every=0, chunksize=1000,
                                      passes=10)
lda.print_topics(-1)
print(lda[corpus[0]])
from gensim.corpora import HashDictionary

dct = HashDictionary(texts)
import pandas as pd

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

k = 0;
for sentence in lines:
    print(sentence)
    it = [0] * lda.num_topics
    for text in sentence.split():
        # print(text)
        i = 0
        # print (len(dictionary))
        l = int(len(dictionary))

        while i < l:

            a = corpus[i]
            # print ('hi',i,a)
            b = (np.array(a)[:, 0])
            # b = list(map(int, b))
            # lengthOfList = len(b)
            # sum = 0
            # firstDigit = b[0]
            # lastDigit = b[-1]
            # sum = firstDigit + lastDigit
            # b = sum
            b = int(b)
            # print (b)
            # print (dictionary[int(b)])
            doc = str(dictionary[int(b)])
            # print ('doc',doc)
            # print ('text',text)
            text = text.lower()
            doc = doc.lower()

            if text.find(doc) != -1:

                a = lda[corpus[i]]
                print(a)
                b = (np.array(a)[:, 1])
                # print (b)

                j = 0
                num_top = int(lda.num_topics)
                # print (num_top)
                value = 0
                while (j < (num_top)):
                    c = np.array(b)[j]
                    c = float(c)
                    print(j)
                    print('c', c)
                    print('value', value)
                    if c > value:
                        value = c
                        topic = j
                    j += 1
                print(topic)
                it[topic] = it[topic] + 1
            i += 1
    ss = sid.polarity_scores(str(sentence))

    afns = afn.score(str(sentence))

    textbs = TextBlob(str(sentence))

    txtb = textbs.sentiment.polarity
    txtbsub = textbs.sentiment.subjectivity

    ss = str(ss)
    print(ss)
    print(it)
    m = int(max(it))
    print('max', m)
    top = [itr for itr, jtr in enumerate(it) if jtr == m]
    print('top', top)
    if len(top) > 1:
        topic = 'neutral'
    else:
        topic = 'Topic' + str(top)
    d = {'sentence ': k, 'topic': [topic], 'Sentiment-vader': ss, 'Sentiment-affin': afns, 'Sentiment-blob':  txtb, 'Sentiment-blobsub':  txtbsub, 'Sent-Text': [sentence]}
    data_frame = pd.DataFrame(data=d)
    data_frame.to_csv('topic-Sent-12.csv', sep=',', mode='a', header=False)

    #    if i0>i1 and i0>i2 and i0>i3 :
    #        print('Topic 0')
    #        d={'sentence ':k, 'topic':['Topic 0'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #    elif  i1>i2 and i1>i3 :
    #        print('Topic 1')
    #        d={'sentence ':k, 'topic':['Topic 1'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #
    #    elif  i2>i3 :
    #        print('Topic 2')
    #        d={'sentence ':k, 'topic':['Topic 2'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #
    #    else :
    #
    #        print ('Topic 3')
    #        d={'sentence ':k, 'topic':['Topic 3'],'Sentiment':ss}
    #        data_frame=pd.DataFrame(data=d)
    #        data_frame.to_csv('topic-Sent-8.csv',sep=',',mode='a',header = False)
    #
    k = k + 1

# line=lines.encode('utf-8')


#    print ('\n')

i = 0
a = []
import pandas as pd
import numpy as np

print(len(dictionary))
while i < len(dictionary):
    # print (lda[corpus[i]])
    value = 0
    j = 0
    c = 0
    a = lda[corpus[i]]
    print(a)
    i += 1
    b = (np.array(a)[:, 1])
    print(b)
    num_top = int(lda.num_topics)
    print(num_top)
    value = 0
    while (j < (num_top)):
        c = float(np.array(b)[j])
        print(j)
        print(c)
        if c > value:
            value = c
            topic = j
        j += 1
    print('i', i)
    cor = str(corpus[i])
    print('value=', value)
    d = {'corpus': [cor], 'value': value, 'topic': topic}
    data_frame = pd.DataFrame(data=d)
    # pd.DataFrame({'corpus':corpus[i], 'value':value,'topic':topic})
    data_frame.to_csv('topic-12.csv', sep=',', mode='a', header=False)

df = pd.read_csv('topic-Sent-8.csv', header=None, index_col=None)
df.columns = ['','Number','Topic Number' ,'Vader-sentiment' ,'Affin-sentiment' ,'TextBlob-Sentiment','TextBlob-Sub','Sentence']
df.to_csv('topic-Sent-8.csv', index=False)

df = pd.read_csv('topic-Sent-12.csv', header=None, index_col=None)
df.columns = ['','Number','Topic Number' ,'Vader-sentiment' ,'Affin-sentiment' ,'TextBlob-Sentiment','TextBlob-Sub','Sentence']
df.to_csv('topic-Sent-12.csv', index=False)
