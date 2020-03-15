# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:31:50 2017

@author: xat
"""

import re



def read_line(line):
    line = line.strip()
    if len(line) > 0:    
        reg = r'(\w*?)\s*\[\w*?=(\w*?)\]\[\w*?=(\w*?)\]'
        new_line = re.findall(reg, line)
        if len(new_line) > 0 and new_line[-1][0].isdigit():
            new_line = new_line[:-1]
        return new_line
        
def read_file(file_name):
    result = [] 
    with open(file_name, encoding='utf-8') as f:
        for index, line in enumerate(f):
            line = line.strip()
            if len(line) > 0:
                new_line = read_line(line)
                if new_line:
                    result.append(new_line)
    return result
    

def write_file(mylist, file_name):
    with open(file_name, 'wt', encoding='utf-8') as f:
        for line in mylist:
            f.write(line+ '\n')
            
def get_value(data, index):
    result = []
    for line in data:
        vec = [item[index] for item in line]
        vec = ' '.join(vec)
        result.append(vec)
    return result
    
def compute_aprf(y_data, y_result):
    pos = compute_aprf_one(y_data, y_result, 1)
    neg = compute_aprf_one(y_data, y_result, 0)
    return 'pos_precision={:.3f}, pos_recall={:.3f}, pos_f1={:.3f}, neg_precision={:.3f}, neg_recall={:.3f}, neg_f1={:.3f}, mean_precision={:.3f}, mean_recall={:.3f}, mean_f1={:.3f}, accuracy={:.3f}'.format(
                pos[1]*100, pos[2]*100, pos[3]*100, neg[1]*100, neg[2]*100, neg[3]*100, (pos[1]+neg[1])*100/2, (pos[2]+neg[2])*100/2,
                (pos[3]+neg[3])*100/2, pos[0]*100 )
    
def compute_aprf_one(y_data, y_result, type=1):
    '''
    a  |  b
    --- ----
    c  | d
    '''
    a, b, c, d = [0] * 4
    for i in range(len(y_result)):
        if y_data[i] == y_result[i]:
            if y_result[i] == type:
                a += 1
            else:
                d += 1
        else:
            if y_result[i] == type:
                c += 1
            else:
                b += 1
    
    if a==0:
        a=1
    if b==0:
        b=1
    if c==0:
        c=1
    if d==0:
        d=1
    accuracy = (a + d)/(a + b + c + d)
    precision = a / (a + c)
    recall = a / (a + b)
    f1 = 2 * precision * recall / ( precision + recall)
    return accuracy, precision, recall, f1
        
        
    
    
            
            
def write_all():
    pos_data = read_file('../data/final_pos_stem.txt')
    neg_data = read_file('../data/final_neg_stem.txt')
    pos_origin = get_value(pos_data, 0)
    neg_origin = get_value(neg_data, 0)
    pos_stem = get_value(pos_data, 2)
    neg_stem = get_value(neg_data, 2)
#    pos_origin_stem = [line1+' | '+line2 for line1, line2 in zip(pos_origin, pos_stem)]
#    neg_origin_stem = [line1+' | '+line2 for line1, line2 in zip(neg_origin, neg_stem)]                
    write_file(pos_origin, '../data/pos_origin.txt')
    write_file(neg_origin, '../data/neg_origin.txt')
    write_file(pos_stem, '../data/pos_stem.txt')
    write_file(neg_stem, '../data/neg_stem.txt')
#    write_file(pos_origin_stem, '../data/pos_origin_stem.txt')
#    write_file(neg_origin_stem, '../data/neg_origin_stem.txt')

    


