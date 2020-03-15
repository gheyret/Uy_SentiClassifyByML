# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 00:16:26 2017

@author: Administrator
"""


from util import read_file
from rule import RuleDetector
from featureSelector import FeatureSelector
from termWeight import TermWeight
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import NuSVC
from sklearn.ensemble.forest import RandomForestClassifier
import os
from preprocesser import Preprocesser
from sklearn.cross_validation import train_test_split

neg_data = read_file('../data/final_neg_stem.txt')
pos_data = read_file('../data/final_pos_stem.txt')
rd = RuleDetector()
neg_origin, neg_stem = rd.get_features(neg_data) 
pos_origin, pos_stem = rd.get_features(pos_data)
neg_test_size = int(len(neg_data)*0.2)
pos_test_size = int(len(pos_data)*0.2)
r_train_neg_origin = neg_origin[neg_test_size:]
r_test_neg_origin = neg_origin[:neg_test_size]
r_train_neg_stem = neg_stem[neg_test_size:]
r_test_neg_stem = neg_stem[:neg_test_size]
def get_data(data, index):
    result = []
    for line in data:
        vec = [(item[index], ) for item in line]
        result.append(vec)
    return result
all_neg_origin = get_data(neg_data, 0)
all_neg_stem = get_data(neg_data, 2)
train_neg_origin = all_neg_origin[pos_test_size:]
train_neg_stem = all_neg_stem[pos_test_size:]
test_neg_origin = all_neg_origin[:pos_test_size]
test_neg_stem = all_neg_stem[:pos_test_size]

r_train_pos_origin = pos_origin[pos_test_size:]
r_test_pos_origin = pos_origin[:pos_test_size]
r_train_pos_stem = pos_stem[pos_test_size:]
r_test_pos_stem = pos_stem[:pos_test_size]
all_pos_origin = get_data(pos_data, 0)
all_pos_stem = get_data(pos_data, 2)
train_pos_origin = all_pos_origin[pos_test_size:]
train_pos_stem = all_pos_stem[pos_test_size:]
test_pos_origin = all_pos_origin[:pos_test_size]
test_pos_stem = all_pos_stem[:pos_test_size]

train_origin_data = train_pos_origin + train_neg_origin
train_stem_data = train_pos_stem + train_neg_stem
test_origin_data = test_pos_origin + test_neg_origin
test_stem_data = test_pos_stem + test_neg_stem
train_target = [1]*len(train_pos_origin) + [0]*len(train_neg_origin)
test_target = [1]*len(test_pos_origin) + [0]*len(test_neg_origin)

r_train_origin_data = r_train_pos_origin + r_train_neg_origin
r_train_stem_data = r_train_pos_stem + r_train_neg_stem
r_test_origin_data = r_test_pos_origin + r_test_neg_origin
r_test_stem_data = r_test_pos_stem + r_test_neg_stem

def get_counts():
    def get_unigram(data):
        result = set()
        for line in data:
            result.update(line)
        return len(result)
    pos_origin_unigram = get_unigram(train_pos_origin)
    neg_origin_unigram=get_unigram(train_neg_origin)
    pos_stem_unigram = get_unigram(train_pos_stem)
    neg_stem_unigram= get_unigram(train_neg_stem)
    def get_bigram(data):
        result = set()
        for line in data:
            if len(line)>1:
                for i in range(len(line)-1):
                    result.add(line[i]+line[i+1])
        return len(result)
    pos_origin_bigram = get_bigram(train_pos_origin)
    neg_origin_bigram= get_bigram(train_neg_origin)
    pos_stem_bigram = get_bigram(train_pos_stem)
    neg_stem_bigram= get_bigram(train_neg_stem)
    def get_rule(data):
        result = set()
        for line in data:
            result.update(line)
        return len(result)
    origin_rule = get_rule(r_train_origin_data)
    stem_rule = get_rule(r_train_stem_data)
    result_dir = '../data/result/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(os.path.join(result_dir,'pos_neg_counts.txt'), 'wt', encoding='utf-8') as f:
        f.write('pos origin unigram features: {0}\n'.format(pos_origin_unigram))
        f.write('neg origin unigram features: {0}\n'.format(neg_origin_unigram))
        f.write('pos stem unigram features: {0}\n'.format(pos_stem_unigram))
        f.write('neg stem unigram features: {0}\n'.format(neg_stem_unigram))
        f.write('pos origin bigram features: {0}\n'.format(pos_origin_bigram))
        f.write('neg origin bigram features: {0}\n'.format(neg_origin_bigram))
        f.write('pos stem bigram features: {0}\n'.format(pos_stem_bigram))
        f.write('neg stem bigram features: {0}\n'.format(neg_stem_bigram))
        f.write('origin rule features: {0}\n'.format(origin_rule))
        f.write('stem rule features: {0}\n'.format(stem_rule))
    
        
def test_rule(clf, clf_name, train_unigram, train_bigram, train_target, test_unigram, test_bigram, test_target, data_type):
    feature_count_range = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000,6000]
    bigram_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    selector_methods = ('mi','df','ig')
    weight_methods = ('tf_idf',)
    
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
                        score = clf.score(test_weighted_data, test_target)
                        result_dir = '../data/result/'
                        if not os.path.exists(result_dir):
                            os.mkdir(result_dir)
                        with open(os.path.join(result_dir,clf_name+'_unigram_and_rule.txt'), 'at', encoding='utf-8') as f:
                            f.write('clf={1}\t data_type={2}\t count={3}\t rule_size={4}\t score={5:.2f}\n'.format(
                                    clf_name, data_type, count, bigram_size, score*100))
                            
def test_rule_all():
    test_rule(NuSVC(), 'SVM', train_origin_data, r_train_origin_data, train_target, 
         test_origin_data, r_test_origin_data, test_target, 'origin')
    test_rule(NuSVC(), 'SVM', train_stem_data, r_train_stem_data, train_target, 
         test_stem_data, r_test_stem_data, test_target, 'stem')
    test_rule(BernoulliNB(), 'BNB', train_origin_data, r_train_origin_data, train_target, 
         test_origin_data, r_test_origin_data, test_target, 'origin')
    test_rule(BernoulliNB(), 'BNB', train_stem_data, r_train_stem_data, train_target, 
         test_stem_data, r_test_stem_data, test_target, 'stem')
    test_rule(RandomForestClassifier(), 'RFC', train_origin_data, r_train_origin_data, train_target, 
         test_origin_data, r_test_origin_data, test_target, 'origin')
    test_rule(RandomForestClassifier(), 'RFC', train_stem_data, r_train_stem_data, train_target, 
         test_stem_data, r_test_stem_data, test_target, 'stem')
    
def test_unigram_all():  
#    test_unigram(NuSVC(), 'SVM', '../data/pos_origin.txt', '../data/neg_origin.txt', 'origin')
#    test_unigram(NuSVC(), 'SVM', '../data/pos_stem.txt', '../data/neg_stem.txt', 'stem')
    test_unigram(BernoulliNB(), 'BNB', '../data/pos_origin.txt', '../data/neg_origin.txt', 'origin')
    test_unigram(BernoulliNB(), 'BNB', '../data/pos_stem.txt', '../data/neg_stem.txt', 'stem')
    test_unigram(RandomForestClassifier(), 'RFC', '../data/pos_origin.txt', '../data/neg_origin.txt', 'origin')
    test_unigram(RandomForestClassifier(), 'RFC', '../data/pos_stem.txt', '../data/neg_stem.txt', 'stem')
                            
def test_unigram(clf, clf_name, pos_file, neg_file, data_type):
        pre = Preprocesser(pos_file, neg_file)
        ngrams = ('unigram',)
        weight_methods = ('tf_idf',)
        for ngram in ngrams:
            data, target = pre.get_ngram(ngram)
            train_data, test_data, train_target, test_target = train_test_split(data, target)
            selector = FeatureSelector(train_data, train_target)
            all_features_count = selector.all_features_size
            features = selector.all_features
            train_vectorizer = TermWeight(train_data, train_target, features)
            for weight_method in weight_methods:
                train_weighted_data = train_vectorizer.weight(weight_method)
                test_vectorizer = TermWeight(test_data, test_target, features)
                test_weighted_data = test_vectorizer.weight(weight_method)
                clf.fit(train_weighted_data, train_target)
                score = clf.score(test_weighted_data, test_target)
                result_dir = '../data/result/'
                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)
                with open(os.path.join(result_dir,clf_name+'_unigram.txt'), 'at', encoding='utf-8') as f:
                    f.write('clf={1}\t data_type={2}\t count={3}\t score={4:.2f}\n'.format(
                            clf_name, data_type, all_features_count, score*100))
                
    
    
    
   