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
    #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
    def __init__(self, pos_file, neg_file, pos_test, neg_test, feature_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]): 
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.pos_test = pos_test
        self.neg_test = neg_test
        self.feature_range = feature_range
        
    def test_clf(self, clf, clf_name):
        pre_train = Preprocesser(self.pos_file, self.neg_file)
        pre_test = Preprocesser(self.pos_test, self.neg_test)
        ngrams = ('unigram', ) # 'unigram', 'bigram', 'dictWords'
        selector_methods = ('df','mi',)
        weight_methods = ('tf_idf',)
        for ngram in ngrams:
            train_data, train_target = pre_train.get_ngram(ngram)
            test_data, test_target = pre_test.get_ngram(ngram)
            selector = FeatureSelector(train_data, train_target)
            for selector_method in selector_methods:
                all_features = selector.select(selector_method, selector.all_features_size) 
                print(len(all_features))
                for count in self.feature_range:
                    print(int(len(all_features)*count))
                    features = all_features[:int(len(all_features)*count)]  #[:count]
                    train_vectorizer = TermWeight(train_data, train_target, features)
                    for weight_method in weight_methods:
                        train_weighted_data = train_vectorizer.weight(weight_method)
                        test_vectorizer = TermWeight(test_data, test_target, features)
                        test_weighted_data = test_vectorizer.weight(weight_method)
                        clf.fit(train_weighted_data, train_target)
                        test_result = clf.predict(test_weighted_data)
                        score = compute_aprf(test_target, test_result)
                        self.write_result(clf_name, ngram, selector_method, len(features), weight_method, score)
                        
    def write_result(self, clf_name, ngram, selector_method, features, weight_method, score):
        with open('../MyData/result/'+clf_name+'.txt', 'at', encoding='utf-8') as f:
            f.write('ngram={0}\tselector_method={1}\tcount={2}\tweight_method={3}\tscore={4}\n'.format(
                    ngram, selector_method, features, weight_method, score))
 

        #1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 
    def test_clf_uni_bi(self, clf, clf_name, feature_count_range=[100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000], uni_bi_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
 #       pre = Preprocesser(self.pos_file, self.neg_file)
        pre_train = Preprocesser(self.pos_file, self.neg_file)
        pre_test = Preprocesser(self.pos_test, self.neg_test)
        selector_methods = ('df', 'ig', 'mi', 'chi',)  # 'df', 'ig', 'mi', 'chi'
        weight_methods = ('tf_idf',)
        
        train_unigram, train_target = pre_train.get_unigram(is_shuffle=False)
        train_bigram, train_target = pre_train.get_bigram(is_shuffle=False)
        
        test_unigram, test_target = pre_test.get_unigram(is_shuffle=False)#, 0.7, 0.9
        test_bigram, test_target = pre_test.get_bigram(is_shuffle=False)
        
        unigram_selector = FeatureSelector(train_unigram, train_target)
        bigram_selector = FeatureSelector(train_bigram, train_target)
        train_data = [unigram_line + bigram_line for unigram_line, bigram_line in zip(train_unigram, train_bigram)]
        test_data = [unigram_line + bigram_line for unigram_line, bigram_line in zip(test_unigram, test_bigram)]
        for selector_method in selector_methods:
                unigram_features = unigram_selector.select(selector_method, unigram_selector.all_features_size)
                bigram_features = bigram_selector.select(selector_method, bigram_selector.all_features_size)
                for count in feature_count_range:
                    for uni_bi_size in uni_bi_range:
                        bi_size = int(count * uni_bi_size)
                        bigram_selected_features = bigram_features[:bi_size]
                        unigram_selected_features = unigram_features[:(count - bi_size)]
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
                                              str(count)+':' + str(uni_bi_size), weight_method, score)
                


    def test_clf_uni_dict(self, clf, clf_name, feature_count_range=[100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000], uni_dict_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]): 
        pre_train = Preprocesser(self.pos_file, self.neg_file)
        pre_test = Preprocesser(self.pos_test, self.neg_test)
        selector_methods = ('df', 'ig', 'mi', 'chi',)
        weight_methods = ('tf_idf',)
        
        train_unigram, train_target = pre_train.get_unigram(is_shuffle=False)
        train_dict, train_target = pre_train.get_dictWords(is_shuffle=False)
        
        test_unigram, test_target = pre_test.get_unigram(is_shuffle=False)
        test_dict, test_target = pre_test.get_dictWords(is_shuffle=False)
        
        unigram_selector = FeatureSelector(train_unigram, train_target)
        dict_selector = FeatureSelector(train_dict, train_target)
        train_data = [unigram_line + dict_line for unigram_line, dict_line in zip(train_unigram, train_dict)]
        test_data = [unigram_line + dict_line for unigram_line, dict_line in zip(test_unigram, test_dict)]
        for selector_method in selector_methods:
                unigram_features = unigram_selector.select(selector_method, unigram_selector.all_features_size)
                dict_features = dict_selector.select(selector_method, dict_selector.all_features_size)
                for count in feature_count_range:
                    for uni_dict_size in uni_dict_range:
                        dict_cout = int(count * uni_dict_size)
                        dict_selected_features = dict_features[:dict_cout]
                        unigram_selected_features = unigram_features[:(count - dict_cout)]
                        features = unigram_selected_features
                        features.extend(dict_selected_features)
                        train_vectorizer = TermWeight(train_data, train_target, features)
                        for weight_method in weight_methods:
                            train_weighted_data = train_vectorizer.weight(weight_method)
                            test_vectorizer = TermWeight(test_data, test_target, features)
                            test_weighted_data = test_vectorizer.weight(weight_method)
                            clf.fit(train_weighted_data, train_target)
                            test_result = clf.predict(test_weighted_data)
                            score = compute_aprf(test_target, test_result)
                            self.write_result(clf_name+'_uni_dict', 'unigram+dict', selector_method, 
                                              str(count) + ':' + str(uni_dict_size), weight_method, score)
    
    
    def test_clf_bi_dict(self, clf, clf_name, feature_count_range=[100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000], bi_dict_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        pre_train = Preprocesser(self.pos_file, self.neg_file)
        pre_test = Preprocesser(self.pos_test, self.neg_test)
        selector_methods = ('df', 'ig', 'mi', 'chi',)  # 'df', 'ig', 'mi', 'chi'
        weight_methods = ('tf_idf',)
        
        train_dict, train_target = pre_train.get_dictWords(is_shuffle=False)
        train_bigram, train_target = pre_train.get_bigram(is_shuffle=False)
        
        test_dict, test_target = pre_test.get_dictWords(is_shuffle=False)
        test_bigram, test_target = pre_test.get_bigram(is_shuffle=False)
        
        dict_selector = FeatureSelector(train_dict, train_target)
        bigram_selector = FeatureSelector(train_bigram, train_target)
        train_data = [bigram_line + dict_line for bigram_line, dict_line in zip(train_bigram, train_dict)]
        test_data = [bigram_line + dict_line for bigram_line, dict_line in zip(test_bigram, test_dict)]
        for selector_method in selector_methods:
                bigram_features = bigram_selector.select(selector_method, bigram_selector.all_features_size)
                dict_features = dict_selector.select(selector_method, dict_selector.all_features_size)
                for count in feature_count_range:
                    for bi_dict_size in bi_dict_range:
                        dict_cout = int(count * bi_dict_size)
                        dict_selected_features = dict_features[:dict_cout]
                        bigram_selected_features = bigram_features[:(count - dict_cout)]
                        features = bigram_selected_features
                        features.extend(dict_selected_features)
                        train_vectorizer = TermWeight(train_data, train_target, features)
                        for weight_method in weight_methods:
                            train_weighted_data = train_vectorizer.weight(weight_method)
                            test_vectorizer = TermWeight(test_data, test_target, features)
                            test_weighted_data = test_vectorizer.weight(weight_method)
                            clf.fit(train_weighted_data, train_target)
                            test_result = clf.predict(test_weighted_data)
                            score = compute_aprf(test_target, test_result)
                            self.write_result(clf_name+'_bi_dict', 'bigram+bigram', selector_method, 
                                              str(count)+':' + str(bi_dict_size), weight_method, score)
                



if __name__ == '__main__':
    t = Tester('../MyData/MovieParallil/no05_pos_origin_train.txt', '../MyData/MovieParallil/no05_neg_origin_train.txt', 
               '../MyData/MovieParallil/no05_pos_origin_test.txt' , '../MyData/MovieParallil/no05_neg_origin_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no05_origin')

    t = Tester('../MyData/MovieParallil/no1_pos_origin_train.txt', '../MyData/MovieParallil/no1_neg_origin_train.txt', 
               '../MyData/MovieParallil/no1_pos_origin_test.txt' , '../MyData/MovieParallil/no1_neg_origin_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no1_origin')

    t = Tester('../MyData/MovieParallil/no15_pos_origin_train.txt', '../MyData/MovieParallil/no15_neg_origin_train.txt', 
               '../MyData/MovieParallil/no15_pos_origin_test.txt' , '../MyData/MovieParallil/no15_neg_origin_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no15_origin')

    t = Tester('../MyData/MovieParallil/no2_pos_origin_train.txt', '../MyData/MovieParallil/no2_neg_origin_train.txt', 
               '../MyData/MovieParallil/no2_pos_origin_test.txt' , '../MyData/MovieParallil/no2_neg_origin_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no2_origin')
    
    t = Tester('../MyData/MovieParallil/no05_pos_stem_train.txt', '../MyData/MovieParallil/no05_neg_stem_train.txt', 
               '../MyData/MovieParallil/no05_pos_stem_test.txt' , '../MyData/MovieParallil/no05_neg_stem_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no05_stem')
    
    t = Tester('../MyData/MovieParallil/no1_pos_stem_train.txt', '../MyData/MovieParallil/no1_neg_stem_train.txt', 
               '../MyData/MovieParallil/no1_pos_stem_test.txt' , '../MyData/MovieParallil/no1_neg_stem_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no1_stem')
    
    t = Tester('../MyData/MovieParallil/no15_pos_stem_train.txt', '../MyData/MovieParallil/no15_neg_stem_train.txt', 
               '../MyData/MovieParallil/no15_pos_stem_test.txt' , '../MyData/MovieParallil/no15_neg_stem_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no15_stem')
    
    t = Tester('../MyData/MovieParallil/no2_pos_stem_train.txt', '../MyData/MovieParallil/no2_neg_stem_train.txt', 
               '../MyData/MovieParallil/no2_pos_stem_test.txt' , '../MyData/MovieParallil/no2_neg_stem_test.txt')
    t.test_clf_bi_dict(BernoulliNB(), 'bnb_no2_stem')
    
    t = Tester('../MyData/MovieParallil/no05_pos_stem_train.txt', '../MyData/MovieParallil/no05_neg_stem_train.txt', 
               '../MyData/MovieParallil/no05_pos_stem_test.txt' , '../MyData/MovieParallil/no05_neg_stem_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no05_stem')
    
    t = Tester('../MyData/MovieParallil/no1_pos_stem_train.txt', '../MyData/MovieParallil/no1_neg_stem_train.txt', 
               '../MyData/MovieParallil/no1_pos_stem_test.txt' , '../MyData/MovieParallil/no1_neg_stem_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no1_stem')
    
    t = Tester('../MyData/MovieParallil/no15_pos_stem_train.txt', '../MyData/MovieParallil/no15_neg_stem_train.txt', 
               '../MyData/MovieParallil/no15_pos_stem_test.txt' , '../MyData/MovieParallil/no15_neg_stem_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no15_stem')
    
    t = Tester('../MyData/MovieParallil/no2_pos_stem_train.txt', '../MyData/MovieParallil/no2_neg_stem_train.txt', 
               '../MyData/MovieParallil/no2_pos_stem_test.txt' , '../MyData/MovieParallil/no2_neg_stem_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no2_stem')
    
    t = Tester('../MyData/MovieParallil/no05_pos_origin_train.txt', '../MyData/MovieParallil/no05_neg_origin_train.txt', 
               '../MyData/MovieParallil/no05_pos_origin_test.txt' , '../MyData/MovieParallil/no05_neg_origin_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no05_origin')
    
    t = Tester('../MyData/MovieParallil/no1_pos_origin_train.txt', '../MyData/MovieParallil/no1_neg_origin_train.txt', 
               '../MyData/MovieParallil/no1_pos_origin_test.txt' , '../MyData/MovieParallil/no1_neg_origin_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no1_origin')
    
    t = Tester('../MyData/MovieParallil/no15_pos_origin_train.txt', '../MyData/MovieParallil/no15_neg_origin_train.txt', 
               '../MyData/MovieParallil/no15_pos_origin_test.txt' , '../MyData/MovieParallil/no15_neg_origin_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no15_origin')
    
    t = Tester('../MyData/MovieParallil/no2_pos_origin_train.txt', '../MyData/MovieParallil/no2_neg_origin_train.txt', 
               '../MyData/MovieParallil/no2_pos_origin_test.txt' , '../MyData/MovieParallil/no2_neg_origin_test.txt')
    t.test_clf_bi_dict(NuSVC(), 'svm_no2_origin')

