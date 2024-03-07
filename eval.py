import networkx as nx
import numpy as np
import os
from collections import defaultdict
import sys
import pdb

def construct_filter_file(file_path, filter_dict):
    with open(file_path) as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            rel = data[0]
            entities = data[1:]
            for i,e in enumerate(entities):
                r_q = rel+'_'+str(i)
                e_q = ' '.join(entities[:i] + entities[i+1:])
                if r_q in filter_dict:
                    if e_q in filter_dict[r_q]:
                        filter_dict[r_q][e_q].append(e)
                    else:
                        filter_dict[r_q][e_q] = [e]
                else:
                    filter_dict[r_q] = dict()
                    filter_dict[r_q][e_q] = [e]
    return filter_dict

def construct_filter(folder_path):
    train_file = os.path.join(folder_path, 'train.txt')
    valid_file = os.path.join(folder_path, 'valid.txt')

    filter_dict = dict()
    filter_dict = construct_filter_file(train_file, filter_dict)
    if os.path.exists(valid_file):
        filter_dict = construct_filter_file(valid_file, filter_dict)
    return filter_dict

def read_tail_file(file_path):
    tail_dict_list = []
    with open(file_path) as f:
        for line in f.readlines():
            tail_dict = dict()
            new_line = line.strip()[9:-2]
            if len(new_line) > 0:
                tail_dict_str = new_line.split(', ')
                for t_d_s in tail_dict_str:
                    [tail, count] = t_d_s.split(': ')
                    tail_dict[tail[1:-1]] = float(count)
            tail_dict_list.append(tail_dict)
    return tail_dict_list

def read_true_tail_file(file_path):
    t_list = []
    with open(file_path) as f:
        for line in f.readlines():
            t_list.append(line.strip())
    return t_list

def read_test_data(test_file_path, option):
    test_data = []
    with open(test_file_path) as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            rel = data[0]
            entities = data[1:]
            for i,_ in enumerate(entities):
                r_q = rel+'_'+str(i)
                e_q = ' '.join(entities[:i] + entities[i+1:])
                if option=='primary' and (i==0 or i==1):
                    test_data.append([r_q, e_q])
                elif option=='all':
                    test_data.append([r_q, e_q])
    return test_data

def get_hits_ranks(datafolder_path, option):
    filter_dict = construct_filter(datafolder_path)

    if option == 'primary':
        tail_ans_path = os.path.join(datafolder_path, 'primary_ans.txt')
        tail_true_path = os.path.join(datafolder_path, 'primary_golden_ans.txt')
        test_data = read_test_data(os.path.join(datafolder_path, 'test.txt'), 'primary')
    elif option == 'all':
        tail_ans_path = os.path.join(datafolder_path, 'all_ans.txt')
        tail_true_path = os.path.join(datafolder_path, 'all_golden_ans.txt')
        test_data = read_test_data(os.path.join(datafolder_path, 'test.txt'), 'all')

    tail_dict_list = read_tail_file(tail_ans_path)
    tail_true_list = read_true_tail_file(tail_true_path)
    
    hits = []
    r_ranks = []
    for i in range(10):
        hits.append([])
    
    for i in range(len(tail_true_list)):
        tail_true = tail_true_list[i]
        tail_ans = tail_dict_list[i]
        if tail_true in tail_ans.keys():
            # filter
            weight = tail_ans[tail_true]
            try:
                [r_q, e_q] = test_data[i]
                for e in filter_dict[r_q][e_q]:
                    tail_ans[e] = 0.0
            except: pass
            tail_ans[tail_true] = weight
            tail_ans_counts = list(tail_ans.values())
            tail_ans_counts.sort(reverse=True)
            rank = tail_ans_counts.index(tail_ans[tail_true]) + 1
            r_ranks.append(1/rank)
            for hits_level in range(1, 11):
                if rank <= hits_level:
                    hits[hits_level-1].append(1.0)
                else:
                    hits[hits_level-1].append(0.0)
        else:
            r_ranks.append(0.0)
            for hits_level in range(10):
                hits[hits_level].append(0.0)

    print(option)
    for i in range(10):
        print('Hits @{}: {}'.format(i+1, np.mean(hits[i])))

    mrr = np.mean(r_ranks)
    print('MRR: {}'.format(mrr))
    return hits, mrr


dataset = sys.argv[1]
print(dataset)
datafolder_path = './data_preprocessed/{}'.format(dataset)
hits, mrr = get_hits_ranks(datafolder_path, option='primary')
hits, mrr = get_hits_ranks(datafolder_path, option='all')
print('\n')

