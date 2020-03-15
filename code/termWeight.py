# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:26:15 2016

@author: xat
"""

import math
import numpy as np

class TermWeight:
    def __init__(self, data, target, feature):
        self.target = target
        self.feature = feature
        self.data = self.get_data(data)
        
    def get_data(self, data):
        result = []
        for line in data:
            vec = []
            for item in line:
                if item in self.feature:
                    vec.append(item)
            result.append(vec)
        return result
        
    def weight(self, method, **kw):
        if method == 'bool':
            return self.weight_bool()
        elif method == 'tf':
            return self.weight_tf()
        elif method == 'idf':
            return self.weight_idf()
        elif method == 'tf_idf':
            return self.weight_tf_idf()
        elif method == 'tfc':
            return self.weight_tfc()
        elif method == 'itc':
            return self.weight_itc()
        elif method == 'entropy':
            return self.weight_entropy()
        elif method == 'tf_iwf':
            return self.weight_tf_iwf()
        elif method == 'bm25':
            return self.weight_bm25()
        elif method == 'dph':
            return self.weight_dph()
        elif method == 'hlm':
            return self.weight_hlm()
            
    def weight_bool(self):
        result = []
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    vec[self.feature.index(item)] = 1
            result.append(vec)
        return result
    
    def weight_tf(self):
        result = []
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    vec[self.feature.index(item)] += 1
            result.append(vec)
        return result
        
    def __get_item_document_count(self, item):
        count = 0
        for line in self.data:
            if item in line:
                count += 1
        return count
        
    def __get_item_frequency(self, item):
        count = 0
        for line in self.data:
            if item in line:
                count += line.count(item)
        return count
    
    def weight_idf(self):
        result = []
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    count = self.__get_item_document_count(item)
                    count_all = len(self.data)
                    vec[self.feature.index(item)] = math.log2(count_all/count)
            result.append(vec)
        return result
    
    def weight_tf_idf(self):
        tf = np.array(self.weight_tf())
        idf = np.array(self.weight_idf())
        result = tf*idf
        return result.tolist()
        
    def weight_bm25(self, k=1.2, b=0.75):
        result = []
        temp = [len(line) for line in self.data]
        avgdl = sum(temp)/len(temp)
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    if not vec[self.feature.index(item)]:
                        tf = line.count(item)
                        dl = len(line)
                        vec[self.feature.index(item)] = (k+1)/(tf+k*((1-b)+b*dl/avgdl))
            result.append(vec)
        idf = self.weight_tf_idf()
        result = np.array(result)
        idf = np.array(idf)
        result = result*idf
        return result.tolist()
        
    def weight_dph(self):
        result = []
        temp = [len(line) for line in self.data]
        avgdl = sum(temp)/len(temp)
        n = len(self.data)
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    if not vec[self.feature.index(item)]:
                        tf_ij = line.count(item)
                        tf_j = self.__get_item_frequency(item)
                        dl = len(line)
                        value = (1- tf_ij/dl)**2
                        value /= (tf_ij + 1)
                        value2 = tf_ij*math.log2(tf_ij*avgdl*n/(dl*tf_j))
                        value3 = 2*math.pi*tf_ij*(1-(tf_ij/dl))
                        if value3 <= 0:
                            vec[self.feature.index(item)] = 0
                        else:
                            value2 += 0.5 * math.log2(value3)
                            value *= value2
                            vec[self.feature.index(item)] = value 
            result.append(vec)
        return result
        
    def weight_hlm(self, lamda=0.15):
        result = []
        temp = [len(line) for line in self.data]
        sigmatf = sum(temp)
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    if not vec[self.feature.index(item)]:
                        tf_ij = line.count(item)
                        df_j = self.__get_item_document_count(item)
                        dl = len(line)
                        value = math.log2(1 + lamda*tf_ij*sigmatf/((1-lamda)*df_j*dl))
                        vec[self.feature.index(item)] = value
            result.append(vec)
        return result
        
    def weight_tfc(self):
        result = self.weight_tf_idf()
        for line in result:
            fenmu = [item**2 for item in line]
            fenmu = sum(fenmu) ** 0.5
            if fenmu != 0:
                for i in range(len(line)):
                    line[i] /= fenmu
        return result
        
    def weight_itc(self):
        result = []
        n = len(self.data)
        for line in self.data:
            vec = [0] * len(self.feature)
            temp = []
            for item in line:
                if item in self.feature:
                    tf_ij = line.count(item)
                    n_i = self.__get_item_document_count(item)
                    value = math.log2((tf_ij+1)*math.log2(n/n_i))
                    temp.append(value**2)
            fenmu = sum(temp) ** 0.5
            for item in line:
                if item in self.feature:
                    tf_ij = line.count(item)
                    n_i = self.__get_item_document_count(item)
                    value = math.log2(tf_ij+1) * math.log2(n/n_i)
                    vec[self.feature.index(item)] = value/fenmu
            result.append(vec)
        return result
        
    def weight_entropy(self):
        segma = []
        n = len(self.data)
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    if not vec[self.feature.index(item)]:
                        tf_ij = line.count(item)
                        n_i = self.__get_item_document_count(item)
                        value = tf_ij/n_i
                        value *= math.log2(tf_ij/n_i)
                        vec[self.feature.index(item)] = value
            segma.append(line)
        segma = np.array(segma)
        segma_vec = segma.sum(axis=0)
        result = []
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    if not vec[self.feature.index(item)]:
                        tf_ij = line.count(item)
                        segma_value = segma_vec.index(item)
                        value = math.log2(tf_ij + 1)*(1 + 1/math.log2(n) * segma_value)
                        vec[self.feature.index(item)] = value
            result.append(vec)
        return result
        
    def weight_tf_iwf(self):
        result = []
        segma = 0 
        for item in self.feature:
            nt = self.__get_item_frequency(item)
            segma += nt
        for line in self.data:
            vec = [0]*len(self.feature)
            for item in line:
                if item in self.feature:
                    if not vec[self.feature.index(item)]:
                        tf_ij = line.count(item)
                        nt = self.__get_item_frequency(item)
                        value = tf_ij * (math.log2(segma/nt))**2
                        vec[self.feature.index(item)] = value
            result.append(vec)
        return result
        
   