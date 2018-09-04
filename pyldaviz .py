import os
from lxml import etree
import re
import sys

# path = 'C:/Users/Ajay/Downloads/1998/posts/000/000/'
#path = 'C:/Users/Ajay/Downloads/1998/posts/000/'
#os.listdir(path)
#outfile = open('C:/Users/Ajay/outfile1.txt', 'w')
#for filename in os.listdir(path):
#    fulllist = os.path.join(path, filename)
#    for filename1 in os.listdir(fulllist):
#        fullname = os.path.join(fulllist, filename1)
#        tree = etree.parse(fullname)
#        root = tree.getroot()
'''        root.tag
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
f = open('C:/Users/Ajay/outfile1.txt')
lines = f.readlines()
f.close()
print(lines)
keylist = set(
    'city public transport buses dublin train luas tram late speed road highway m50 airport flights timing terminal1 terminal2 facilities cancellation weather storm education school college university pre primary primary cost free standard area rent morgage interest downpayment bank student hous crisis satisfaction farm fish potato cow bread sheep shop mall industries top10 booming financee tax it information software insurance event election government protest EU Europe ireland electricity garbage power accident live tracking development poverty economy deaths'.split())
import nltk

nltk.download("stopwords")
stoplist = str(stopwords.words('english'))
a = 0
outfile_opt = open('C:/Users/Ajay/outfile2.txt', 'w')
for sentence in lines:
    for word in keylist:
        if word in sentence:
            a = 1
    if a == 1:
        sentence += '\n'
        outfile_opt.write(sentence)
outfile_opt.close()'''

boards_stop = set(
    'boards http ip border width hight ie color colour'.split())


f = open('C:/Users/Ajay/outfile2.txt',encoding='utf8')
lines = f.readlines()
f.close()
print(lines)
lines = ''.join([i for i in str(lines) if not i.isdigit()])
english_words = []
#import enchant
from gensim import corpora
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stoplist = str(stopwords.words('english'))

nltk.download('wordnet')
from nltk.corpus import wordnet
#d = enchant.Dict("en_US")
line= str(lines)    
for word in line.lower().split():
    if wordnet.synsets(word):
        english_words.append(word)

" ".join(english_words)
print (english_words)
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
texts = ([[word for word in document.lower().split() if word not in stoplist and word not in boards_stop and ps.stem(word)]
         for document in english_words])

texts = list(filter(None, texts))
print (texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')  # store the dictionary, for future reference
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
corpora.MmCorpus.serialize('deerwester.mm', corpus)  # store to disk, for later use
from gensim import corpora, models, similarities
a = corpus;
import logging, gensim
from gensim.models import LdaModel, HdpModel
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim
#from gensim.models import LdaModel, HdpModel

def test_hdp():
    """Trains a HDP model and tests the html outputs."""
    corpus, dictionary = get_corpus_dictionary()

    #hdp = HdpModel(corpus, dictionary.id2token)

    data = pyLDAvis.gensim.prepare(hdp, corpus, dictionary)
    pyLDAvis.save_html(data, 'C:/Users/Ajay/Downloads/index_hdp.html')
    #os.remove('index_hdp.html')


def test_lda():
    """Trains a LDA model and tests the html outputs."""
    i =4
    while i <= 12:
        corpus, dictionary = get_corpus_dictionary()
        #lda = LdaModel(corpus=corpus,num_topics=4)
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, update_every=0,
                                          chunksize=1000, passes=10)
        data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        path = 'C:/Users/Ajay/Downloads/index_lda'+str(i)+'.html'
        pyLDAvis.save_html(data, path)
#        os.remove(path)
#        pyLDAvis.show(data)
        i+=4


def get_corpus_dictionary():
    """Crafts a toy corpus and the dictionary associated."""
    # Toy corpus.

    dictionary = corpora.Dictionary(texts)
#    dictionary = dictionary()

    # Transforming corpus with dictionary.
    corpus = [dictionary.doc2bow(doc) for doc in texts]

    # Building reverse index.
    for (token, uid) in dictionary.token2id.items():
        dictionary.id2token[uid] = token

    return corpus, dictionary



#import pyLDAvis
#import pyLDAvis.sklearn
#import pyLDAvis.gensim
#from gensim.models import LdaModel, HdpModel
#ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
#from gensim.corpora.dictionary import Dictionary
#print ('after import gensim')
#lda = LdaModel(corpus=corpus,num_topics=1)
#data = (pyLDAvis.gensim.prepare( 1,corpus, dictionary))
#pyLDAvis.show(data)
#pyLDAvis.save_html(data, 'index_lda1.html')
#os.remove('index_lda1.html')
#lda = models.ldamodel.LdaModel.load('lda.model')
#vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary,sort_topics=False)
#print ('after import gensim prep')
#pyLDAvis.display(vis)
#print ('after import gensim-last')
#pyLDAvis.enable_notebook()
#pyLDAvis.gensim.prepare(lda, corpus, dictionary)


 #"""Displaying gensim topic models"""
    ## Load files from "gensim_modeling"
#corpus = corpora.MmCorpus(corpusfile)
#dictionary = corpora.Dictionary.load(dictionaryfile) # for pyLDAvis
#myldamodel = models.ldamodel.LdaModel.load(modelfile)

    ## Interactive visualisation
#import pyLDAvis.gensim
#vis = pyLDAvis.gensim.prepare(myldamodel, corpus, dictionary)
#pyLDAvis.display(vis)
#dictionary = Dictionary(corpus)

    # Transforming corpus with dictionary.

#corpus = [dictionary.doc2bow(doc) for doc in corpus]
#import matplotlib.pyplot as plt
#from gensim.models import CoherenceModel

#c_v = []
#lm_list = []
#for num_topics in range(1, 4):
#    lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
#    lm_list.append(lm)
#    cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
#    c_v.append(cm.get_coherence())

# Show graph
#x = range(1, limit)
#plt.plot(x, c_v)
#plt.xlabel("num_topics")
#plt.ylabel("Coherence score")
#plt.legend(("c_v"), loc='best')
#plt.show()



if __name__ == "__main__":
    test_lda()
#    test_hdp()

