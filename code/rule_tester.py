# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:14:06 2017

@author: Administrator
"""

from termWeight import TermWeight
from featureSelector import FeatureSelector
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import NuSVC
from sklearn.ensemble.forest import RandomForestClassifier
import os
from util import compute_aprf

def read_word_rule(file_name):
    words = []
    rules = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line)>0:
                if '|' in line:
                    word_line, rule_line = line.split('|')
                    words.append(word_line.strip().split())
                    rules.append(rule_line.strip().split())
                else:
                    words.append(line.split())
                    rules.append([])
    return words, rules


def get_all_features(data):
    result = set()
    for line in data:
        result.update(line)
    return list(result)
    

def test_unigram(pos_file, neg_file, clf):
    pos_data,_ = read_word_rule(pos_file)
    neg_data,_ = read_word_rule(neg_file)
    test_size = 0.2
    pos_test_index = int(len(pos_data)*test_size)
    neg_test_index = int(len(neg_data)*test_size)
    pos_test_data = pos_data[:pos_test_index]
    pos_train_data = pos_data[pos_test_index:]
    neg_test_data = neg_data[:neg_test_index]
    neg_train_data = neg_data[neg_test_index:]
    train_data = pos_train_data + neg_train_data
    test_data = pos_test_data + neg_test_data
    train_target = [1]*len(pos_train_data) + [0]*len(neg_train_data)
    test_target = [1]*len(pos_test_data) + [0]*len(neg_test_data)
    features = get_all_features(train_data)
    train_vectorizer = TermWeight(train_data, train_target, features)
    weight_method = 'tf_idf'
    train_weighted_data = train_vectorizer.weight(weight_method)
    test_vectorizer = TermWeight(test_data, test_target, features)
    test_weighted_data = test_vectorizer.weight(weight_method)
    clf.fit(train_weighted_data, train_target)
    score = clf.score(test_weighted_data, test_target)
    return score

def test_unigram_all():
    pos_origin_file = '../data/rule_word_origin_pos.txt'
    neg_origin_file = '../data/rule_word_origin_neg.txt'
    pos_stem_file = '../data/rule_word_stem_pos.txt'
    neg_stem_file = '../data/rule_word_stem_neg.txt'
    svm_origin_score = test_unigram(pos_origin_file, neg_origin_file, NuSVC())
    svm_stem_score = test_unigram(pos_stem_file, neg_stem_file, NuSVC())
    bnb_origin_score = test_unigram(pos_origin_file, neg_origin_file, BernoulliNB())
    bnb_stem_score = test_unigram(pos_stem_file, neg_stem_file, BernoulliNB())
    rfc_origin_score = test_unigram(pos_origin_file, neg_origin_file, RandomForestClassifier())
    rfc_stem_score = test_unigram(pos_stem_file, neg_stem_file, RandomForestClassifier())
    if not os.path.exists('../result'):
        os.mkdir('../result')
    with open('../result/unigram_result.txt', 'wt', encoding='utf-8') as f:
        f.write('original word result:\n')
        f.write('\t\t SVM: {0:.3f}%\n'.format(svm_origin_score*100))
        f.write('\t\t BNB: {0:.3f}%\n'.format(bnb_origin_score*100))
        f.write('\t\t RFC: {0:.3f}%\n'.format(rfc_origin_score*100))
        f.write('\n stem result:\n')
        f.write('\t\t SVM: {0:.3f}%\n'.format(svm_stem_score*100))
        f.write('\t\t BNB: {0:.3f}%\n'.format(bnb_stem_score*100))
        f.write('\t\t RFC: {0:.3f}%\n'.format(rfc_stem_score*100))
        
def write_result(clf_name, data_type, selector_method, count, rule_size, score):
    if not os.path.exists('../data/result'):
        os.mkdir('../data/result')
    with open('../data/result/'+clf_name+'_unigram_rule.txt', 'at', encoding='utf-8') as f:
        f.write('data_type={0}\t selector_method={1}\t count={2}\t rule_size={3:.2f}\t score={4}\n'.format(
                data_type, selector_method, count, rule_size, score))

        
def test_unigram_rule(pos_file, neg_file, clf, clf_name, data_type):
    selector_methods = ('df',)
    weight_methods = ('tf_idf',)
    feature_count_range = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    rule_range = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    pos_data,rule_pos_data = read_word_rule(pos_file)
    neg_data,rule_neg_data = read_word_rule(neg_file)
    test_size = 0.2
    pos_test_index = int(len(pos_data)*test_size)
    neg_test_index = int(len(neg_data)*test_size)
    pos_test_data = pos_data[:pos_test_index]
    pos_train_data = pos_data[pos_test_index:]
    neg_test_data = neg_data[:neg_test_index]
    neg_train_data = neg_data[neg_test_index:]
    train_data = pos_train_data + neg_train_data
    test_data = pos_test_data + neg_test_data
    train_target = [1]*len(pos_train_data) + [0]*len(neg_train_data)
    test_target = [1]*len(pos_test_data) + [0]*len(neg_test_data)
    
    rule_pos_test_data = rule_pos_data[:pos_test_index]
    rule_pos_train_data = rule_pos_data[pos_test_index:]
    rule_neg_test_data = rule_neg_data[:neg_test_index]
    rule_neg_train_data = rule_neg_data[neg_test_index:]
    rule_train_data = rule_pos_train_data + rule_neg_train_data
    rule_test_data = rule_pos_test_data + rule_neg_test_data
    
    def merge_data(data_a, data_b):
        result = []
        for i in range(len(data_a)):
            result.append(data_a[i] + data_b[i])
        return result
    all_train_data = merge_data(train_data, rule_train_data)
    all_test_data = merge_data(test_data, rule_test_data)
    
    unigram_selector = FeatureSelector(train_data, train_target)
    rule_selector = FeatureSelector(rule_train_data, train_target)
    
    for selector_method in selector_methods:
        unigram_features = unigram_selector.select(selector_method, 
                                               unigram_selector.all_features_size)
        rule_features = rule_selector.select(selector_method, 
                                             rule_selector.all_features_size)
        for count in feature_count_range:
            for rule_size in rule_range:
                rule_count = int(count * rule_size)
                rule_selected_features = rule_features[:rule_count]
                unigram_selected_features = unigram_features[:(count - rule_count)]
                features = unigram_selected_features
                features.extend(rule_selected_features)
                train_vectorizer = TermWeight(all_train_data, train_target, features)
                for weight_method in weight_methods:
                    train_weighted_data = train_vectorizer.weight(weight_method)
                    test_vectorizer = TermWeight(all_test_data, test_target, features)
                    test_weighted_data = test_vectorizer.weight(weight_method)
                    clf.fit(train_weighted_data, train_target)
                    test_result = clf.predict(test_weighted_data)
                    score = compute_aprf(test_target, test_result)
                    write_result(clf_name, data_type, selector_method, count, rule_size, score)
                    
                
def test_unigram_rule_all():
    pos_origin_file = '../data/rule_word_origin_pos.txt'
    neg_origin_file = '../data/rule_word_origin_neg.txt'
    pos_stem_file = '../data/rule_word_stem_pos.txt'
    neg_stem_file = '../data/rule_word_stem_neg.txt'
    test_unigram_rule(pos_origin_file, neg_origin_file, NuSVC(), 'SVM', 'origin')
    test_unigram_rule(pos_stem_file, neg_stem_file, NuSVC(), 'SVM', 'stem')
    test_unigram_rule(pos_origin_file, neg_origin_file, BernoulliNB(), 'BNB', 'origin')
    test_unigram_rule(pos_stem_file, neg_stem_file, BernoulliNB(),  'BNB', 'stem')
    test_unigram_rule(pos_origin_file, neg_origin_file, RandomForestClassifier(), 'RFC', 'origin')
    test_unigram_rule(pos_stem_file, neg_stem_file, RandomForestClassifier(), 'RFC','stem')

    
