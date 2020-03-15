# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:52:03 2016

@author: AJ_XJU
"""
import random

class Preprocesser:
    def __init__(self, pos_file, neg_file, stop_words='../data/stop_words.txt', dict_words='../data/dict_words.txt'):
        self.pos_corpus = self.read_file(pos_file)
        self.neg_corpus = self.read_file(neg_file)
        self.stop_words = self.read_file(stop_words)
        self.dict_words = self.read_file(dict_words)
        
    def get_ngram(self, ngram, is_shuffle=True):
        if ngram == 'unigram':
            return self.get_unigram(is_shuffle)
        elif ngram == 'bigram':
            return self.get_bigram(is_shuffle)
        elif ngram == 'trigram':
            return self.get_trigram(is_shuffle)
        elif ngram == 'dictWords': 
            return self.get_dictWords(is_shuffle)
            
    def get_unigram(self, is_shuffle=True):
        pos_data, neg_data = self.corpus2unigram_mat()
        return self.merge(pos_data, neg_data, is_shuffle)
    
    def get_bigram(self, is_shuffle=True):
        pos_data, neg_data = self.corpus2bigram_mat()
        return self.merge(pos_data, neg_data, is_shuffle)
    
    def get_trigram(self, is_shuffle=True):
        pos_data, neg_data = self.corpus2trigram_mat()
        return self.merge(pos_data, neg_data, is_shuffle)
    
    def get_dictWords(self, is_shuffle=True):
        pos_data, neg_data = self.corpus2dictSent_mat()
        return self.merge(pos_data, neg_data, is_shuffle)
       
    def read_file(self, file_name):
        result = []
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result.append(line.strip())
        return result
    
    '''   
    def read_file(self, file_name):
        count=10
        result = []
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result.append(line.strip())
        random.shuffle(result)                   
        i=0
        f_result=[]
        for item in result:
            i+=1
            if i>count:
                break
            f_result.append(item)
        print(f_result)
        return f_result
    '''
        
    def corpus2dictSent_mat(self, filter_dict_words=True, filter_stop_words=True):
        def fun(corpus):
            result = []
            for line in corpus:
                vec = []
                line = line.strip()
                for word in line.split():
                    if filter_dict_words and word in self.dict_words:
                        vec.append((word,))
                if len(vec) > 0:
                    result.append(vec)
                else:
                    for word in line.split():
                        if filter_stop_words and word in self.stop_words:
                            continue
                        vec.append((word,))
                    result.append(vec)
            return result
        return fun(self.pos_corpus), fun(self.neg_corpus)
    
    def corpus2unigram_mat(self, filter_stop_words=True):
        def fun(corpus):
            result = []
            for line in corpus:
                vec = []
                line = line.strip()
                for word in line.split():
                    if filter_stop_words and word in self.stop_words:
                        continue
                    vec.append((word,))
                result.append(vec)
            return result
        return fun(self.pos_corpus), fun(self.neg_corpus)
        
    def corpus2bigram_mat(self):
        pos_unigram, neg_unigram = self.corpus2unigram_mat()
        def fun(corpus):
            result = []
            for line in corpus:
                vec = []
                if len(line) >= 2:
                    for i in range(len(line)-1):
                        vec.append(line[i]+line[i+1])
                result.append(vec)
            return result
        return fun(pos_unigram), fun(neg_unigram)
        
    def corpus2trigram_mat(self):
        pos_unigram, neg_unigram = self.corpus2unigram_mat()
        def fun(corpus):
            result = []
            for line in corpus:
                vec = []
                if len(line) >= 3:
                    for i in range(len(line)-2):
                        vec.append(line[i]+line[i+1]+line[i+2])
                result.append(vec)
            return result
        return fun(pos_unigram), fun(neg_unigram)
        
    def merge(self, pos_data, neg_data, is_shuffle=True):
        pos_target = [1]*len(pos_data)
        neg_target = [0]*len(neg_data)
        pos_data = [(line, cat) for line, cat in zip(pos_data, pos_target)]
        neg_data = [(line, cat) for line, cat in zip(neg_data, neg_target)]
        pos_data.extend(neg_data)
        if is_shuffle:
            random.shuffle(pos_data)
        data = [line for line,_ in pos_data]
        target = [cat for _,cat in pos_data]
        return data, target
  
        
        
        
    

    
