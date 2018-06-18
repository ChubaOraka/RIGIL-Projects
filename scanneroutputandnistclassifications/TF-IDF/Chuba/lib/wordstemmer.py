# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:13:29 2018

@author: Chuba Oraka

From cell [26] in
https://github.com/javedsha/text-classification/blob/master/Text%2BClassification%2Busing%2Bpython%2C%2Bscikit%2Band%2Bnltk.ipynb
"""

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])