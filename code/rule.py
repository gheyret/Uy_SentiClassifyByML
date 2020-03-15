# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:49:43 2017

@author: xat
"""

from util import read_file

class RuleDetector:
    def __init__(self, stop_words='../data/stop_words.txt'):
        self.stop_words = self.read_stop_words(stop_words)
        
    def get_line_feature(self, line):
        if len(line) < 2:
            return [], []
        origin_line = []
        stem_line = []
        for i in range(len(line)-1):
            curr_item = line[i]
            next_item = line[i+1]
            if curr_item[1] == 'N':
                if next_item[1] == 'A' or next_item[1] == 'V':
                    origin_line.append((curr_item[0], next_item[0]))
                    stem_line.append((curr_item[2], next_item[2]))
            elif curr_item[1] == 'A':
                if next_item[1] == 'A' or next_item[1] == 'N' or next_item[1] == 'V':
                    origin_line.append((curr_item[0], next_item[0]))
                    stem_line.append((curr_item[2], next_item[2]))
            elif curr_item[1] == 'D':
                if next_item[1] == 'A' or next_item[1] == 'V':
                    origin_line.append((curr_item[0], next_item[0]))
                    stem_line.append((curr_item[2], next_item[2]))
            elif curr_item[1] == 'E':
                if next_item[1] == 'E':
                    origin_line.append((curr_item[0], next_item[0]))
                    stem_line.append((curr_item[2], next_item[2]))
        return origin_line, stem_line
        
    def read_stop_words(self, file_name):
        result = set()
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                result.add(line)
        return result
        
    def filter_stop_words(self, line):
        return list(filter(lambda item:item[0] not in self.stop_words, line))
        
    def get_features(self, tagged_sents):
        origin_result = []
        stem_result = []
        for line in tagged_sents:
            new_line = self.filter_stop_words(line)
            origin_line, stem_line = self.get_line_feature(new_line)
            origin_result.append(origin_line)
            stem_result.append(stem_line)
        return origin_result, stem_result
        
    def write_features(self, data, file_name):
        with open(file_name, 'wt', encoding='utf-8') as f:
            for line in data:
                if line:
                    s = ''
                    for item in line:
                        s += item[0] + '-' + item[1] + ' '
                    f.write(s + '\n')
                else:
                    f.write('\n')
                    
    def write_word_and_features(self, sents, feature_sents, is_origin ,file_name):
        if is_origin:
            index = 0
        else:
            index = 2
        with open(file_name, 'wt', encoding='utf-8') as f:
            for i in range(len(sents)):
                if sents[i]:
                    text = ' '.join([item[index] for item in sents[i]])
                if feature_sents[i]:
                    text += ' | ' + ' '.join([item[0] + '-' + item[1] for item in feature_sents[i]])
                f.write(text + '\n')
    

if __name__ == '__main__':
    neg_data = read_file('../data/negative/final_neg_stem.txt')
    pos_data = read_file('../data/negative/final_pos_stem.txt')
    rd = RuleDetector()
    origin_result, stem_result = rd.get_features(neg_data) 
    rd.write_features(origin_result, '../data/rule_neg_origin.txt') 
    rd.write_features(stem_result, '../data/rule_neg_stem.txt') 
    rd.write_word_and_features(neg_data, origin_result, True, '../data/rule_word_origin_neg.txt')
    rd.write_word_and_features(neg_data, stem_result, False, '../data/rule_word_stem_neg.txt')
    origin_result, stem_result = rd.get_features(pos_data) 
    rd.write_features(origin_result, '../data/rule_pos_origin.txt') 
    rd.write_features(stem_result, '../data/rule_pos_stem.txt')  
    rd.write_word_and_features(pos_data, origin_result, True, '../data/rule_word_origin_pos.txt')
    rd.write_word_and_features(pos_data, stem_result, False, '../data/rule_word_stem_pos.txt')          
#                    
#                
#                    





