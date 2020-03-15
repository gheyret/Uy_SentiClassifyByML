# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:56:53 2016

@author: AJ_XJU
"""

from preprocesser import Preprocesser
from featureSelector import FeatureSelector
from termWeight import TermWeight
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import NuSVC
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from rule import RuleDetector
from util import read_file, compute_aprf
import random


class Tester:
    def __init__(self, pos_file, neg_file, feature_range=[1000]):   # , 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.feature_range = feature_range
        
    def test_clf(self, clf, clf_name):
        pre_train = Preprocesser(self.pos_file, self.neg_file)
        ngrams = ('unigram',) # 'bigram'
        selector_methods = ('df',)
        weight_methods = ('tf_idf',)
        for ngram in ngrams:
            data, target = pre_train.get_ngram(ngram)
            train_data, test_data, train_target, test_target = train_test_split(data, target,test_size=0.70)
            selector = FeatureSelector(train_data, train_target)
            for selector_method in selector_methods:
                all_features = selector.select(selector_method, selector.all_features_size)
                for count in self.feature_range:
                    features = all_features[:count]
                    train_vectorizer = TermWeight(train_data, train_target, features)
                    for weight_method in weight_methods:
                        train_weighted_data = train_vectorizer.weight(weight_method)
                        test_vectorizer = TermWeight(test_data, test_target, features)
                        test_weighted_data = test_vectorizer.weight(weight_method)
                        clf.fit(train_weighted_data, train_target)
                        test_result = clf.predict(test_weighted_data)
                        score = compute_aprf(test_target, test_result)
                        self.write_result(clf_name, ngram, selector_method, count, weight_method, score)
                        
    
    def split_data(self, unigram_data, bigram_data, target, test_size=0.25):
        data = [(unigram_line, bigram_line, label)  for unigram_line, bigram_line, label in zip(unigram_data, bigram_data, target)]
        test_count = int(test_size*len(data))
        random.shuffle(data)
        unigram_train = [item[0] for item in data[:-test_count]]
        unigram_test = [item[0] for item in data[-test_count:]]
        bigram_train = [item[1] for item in data[:-test_count]]
        bigram_test = [item[1] for item in data[-test_count:]]
        target_train = [item[2] for item in data[:-test_count]]
        target_test = [item[2] for item in data[-test_count:]]
        return unigram_train, unigram_test, bigram_train, bigram_test, target_train, target_test
        
    def test_clf_by_percent(self, clf, clf_name, feature_count_range=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], bigram_range=[0.1, 0.3, 0.5, 0.7, 0.9]):
        pre = Preprocesser(self.pos_file, self.neg_file)
        selector_methods = ('df',)
        weight_methods = ('tf_idf',)
        unigram_data, unigram_target = pre.get_unigram(is_shuffle=False)
        bigram_data, bigram_target = pre.get_bigram(is_shuffle=False)
        train_unigram, test_unigram, train_bigram, test_bigram, train_target, \
                                        test_target = self.split_data(unigram_data, bigram_data, unigram_target)
        unigram_selector = FeatureSelector(train_unigram, train_target)
        bigram_selector = FeatureSelector(train_bigram, train_target)
        train_data = [unigram_line + bigram_line for unigram_line, bigram_line in zip(train_unigram, train_bigram)]
        test_data = [unigram_line + bigram_line for unigram_line, bigram_line in zip(test_unigram, test_bigram)]
        for selector_method in selector_methods:
                unigram_features = unigram_selector.select(selector_method, unigram_selector.all_features_size)
                bigram_features = bigram_selector.select(selector_method, bigram_selector.all_features_size)
                for count in feature_count_range:
                    for bigram_size in bigram_range:
                        bigram_count = int(count * bigram_size)
                        bigram_selected_features = bigram_features[:bigram_count]
                        unigram_selected_features = unigram_features[:(count - bigram_count)]
                        features = unigram_selected_features
                        features.extend(bigram_selected_features)
                        train_vectorizer = TermWeight(train_data, train_target, features)
                        for weight_method in weight_methods:
                            train_weighted_data = train_vectorizer.weight(weight_method)
                            test_vectorizer = TermWeight(test_data, test_target, features)
                            test_weighted_data = test_vectorizer.weight(weight_method)
                            clf.fit(train_weighted_data, train_target)
                            test_result = clf.predict(test_weighted_data)
                            score = compute_aprf(test_target, test_result)
                            self.write_result(clf_name+'_uni_bi', 'unigram+bigram', selector_method, 
                                              str(count)+':' + str(bigram_size), weight_method, score)                   
                
        
    def write_result(self, clf_name, ngram, selector_method, count, weight_method, score):
        with open('../MyData/result/'+clf_name+'.txt', 'at', encoding='utf-8') as f:
            f.write('ngram={0}\tselector_method={1}\tcount={2}\tweight_method={3}\tscore={4}\n'.format(
                    ngram, selector_method, count, weight_method, score))
 

if __name__ == '__main__':
    t = Tester('../MyData/basic_pos_stem_train.txt', '../MyData/basic_neg_stem_train.txt', 
               '../MyData/basic_pos_stem_test.txt' , '../MyData/basic_neg_stem_test.txt')
    t.test_clf(NuSVC(), 'svm_basic_stem')
    
#    t=Tester('../data/pos_origin.txt', '../data/neg_origin.txt')
#    t.test_clf(BernoulliNB,'nb-origin')
    
#    t=Tester('../data/pos_stem.txt', '../data/neg_stem.txt')
#    t.test_clf(BernoulliNB,'nb-stem')

