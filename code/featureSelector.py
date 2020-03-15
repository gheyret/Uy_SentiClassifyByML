# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:11:34 2016

@author: xat
"""

import math
from preprocesser import Preprocesser

class FeatureSelector:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.all_features_size = len(self.get_bag())

    def get_bag(self):
        '''
        返回一个字典， 字典的key为语料库中所有的特征，value是0
        '''
        result = {}
        for line in self.data:
            for item in line:
                result[item] = 0
        return result
            
    def select(self, method, count):
        '''
        特征选择方法，根据指定的选择策略，返回指定数目的特征，
        参数method可以取的值：ig, chi, df, mi
        '''
        if method == 'ig':
            return self.select_ig(count)
        elif method == 'chi':
            return self.select_chi(count)
        elif method == 'df':
            return self.select_df(count)
        elif method == 'mi':
            return self.select_mi(count)
            
    def __get_item_cat_count(self, item, cat):
        '''
        返回指定类别中的特征频数
        '''
        count = 0
        for index, line in enumerate(self.data):
            if item in line and self.target[index] == cat:
                count += 1
        return count 
        
    def __get_item_count(self, item):
        '''
        返回特征item在整个文档中的频数
        '''
        count = 0
        for line in self.data:
            if item in line:
                count += 1
        return count
        
    def __sort_dic(self, dic):
        '''
        按照字典的value值排序，然后只返回从大到小排序的key
        '''
        temp = [(value, key) for key,value in dic.items()]
        temp.sort(reverse=True)
        return [item for _,item in temp]
        
    def select_ig(self, count):
        '''
        计算每一个特征的信息增益，然后排序，返回指定数目的特征，具体公式在统计自然语言处理 P420
        '''
        p_pos = sum(self.target)/len(self.target)
        p_neg = 1- p_pos
        entropy_s = -(math.log2(p_pos)*p_pos + p_neg*math.log2(p_neg))
        bag = self.get_bag()
        def get_item_entropy(item):
            result = 0
            item_count = self.__get_item_count(item)
            p_item = item_count/len(self.data)
            p_not_item = 1 - p_item
            p_pos_item = (self.__get_item_cat_count(item, 1)+1)/(item_count+len(bag))
            p_neg_item = (self.__get_item_cat_count(item, 0)+1)/(item_count+len(bag))
            p_pos_not_item = (sum(self.target)-self.__get_item_cat_count(item, 1)+1)/(len(self.data)-item_count+len(bag))
            p_neg_not_item = (len(self.target)-sum(self.target)-self.__get_item_cat_count(item, 0)+1)/(len(self.data)-item_count+len(bag))
            result += -p_item*(p_pos_item*math.log2(p_pos_item) + p_neg_item*math.log2(p_neg_item))
            result += -p_not_item*(p_pos_not_item*math.log2(p_pos_not_item) + p_neg_not_item*math.log2(p_neg_not_item))
            return result
        for item in bag:
            entropy = entropy_s - get_item_entropy(item)
            bag[item] = entropy
        return self.__sort_dic(bag)[:count]
    
    def select_chi(self, count):
        def get_item_chi(item, cat):
            result = 0
            a = self.__get_item_cat_count(item, cat)
            b = self.__get_item_count(item) - a
            c = self.target.count(cat) - a
            n = len(self.data)
            d = n - (a+b+c)
            result = n*(a*d-c*b)
            fenmu = (a+c)*(b+d)*(a+b)*(c+d)
            return result/fenmu
        bag = self.get_bag()
        for item in bag:
            bag[item] = max(get_item_chi(item, 1), get_item_chi(item, 0))
        return self.__sort_dic(bag)[:count]
    
    def select_df(self, count):
        bag = self.get_bag()
        for line in self.data:
            for item in set(line):
                bag[item] += 1
        temp = self.__sort_dic(bag)
        return temp[:count]
    
    def select_mi(self, count):
        def get_item_mi(item, cat, bag_count):
            result = 0
            a = self.__get_item_cat_count(item, cat) + 1
            b = self.__get_item_count(item) - a
            c = self.target.count(cat) - a
            n = len(self.data)
            result = (a*n)/((a+c)*(a+b)+bag_count)
            result = math.log2(result)
            return result
        bag = self.get_bag()
        p_pos = self.target.count(1)/len(self.target)
        p_neg = 1 - p_pos
        for item in bag:
            bag[item] = max(p_pos*get_item_mi(item, 1,len(bag)), p_neg*get_item_mi(item, 0,len(bag)))
        return self.__sort_dic(bag)[:count]
            

if __name__ == '__main__':
    pre = Preprocesser('..\data\pos_origin.txt', '..\data\neg_origin.txt')
    data, target = pre.get_ngram('unigram')
    sel = FeatureSelector(data, target)
    
