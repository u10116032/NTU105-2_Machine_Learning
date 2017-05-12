#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:31:14 2017

@author: halley
"""


import word2vec
import numpy as np
import nltk



#%%
# train
# DEFINE your parameters for training
MIN_COUNT = 10
WORDVEC_DIM = 500
WINDOW = 5
NEGATIVE_SAMPLES = 5
ITERATIONS = 200
MODEL = 0
LEARNING_RATE = 0.001

# train model
word2vec.word2vec(
    train= 'hp/all.txt',
    output= 'hp/model_test.bin',
    cbow=MODEL,
    size=WORDVEC_DIM,
    min_count=MIN_COUNT,
    window=WINDOW,
    negative=NEGATIVE_SAMPLES,
    iter_=ITERATIONS,
    alpha=LEARNING_RATE,
    verbose=True
    )

#%%
# test   
# load model for plotting
model = word2vec.load('hp/model_test.bin')

vocabs = []                 
vecs = []                   
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:1000]
vocabs = vocabs[:1000]

'''
Dimensionality Reduction
'''
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)


'''
Plotting
'''
import matplotlib.pyplot as plt
from adjustText import adjust_text

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’","Page"]


plt.figure(figsize=(16,12))
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig('hp.png', dpi=600)
plt.show()

