import networkx as nx
import copy
import os
import pdb
from collections import Counter
import pickle
from tqdm import tqdm
import time
import sys
import re
import numpy as np
import random


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def construct_h_r_t_dict(h_r_t_dict, h, r, t):
    if h in h_r_t_dict:
        if r in h_r_t_dict[h]:
            h_r_t_dict[h][r].append(t)
        else:
            h_r_t_dict[h][r] = [t]
    else:
        h_r_t_dict[h] = dict()
        h_r_t_dict[h][r] = [t]
    return h_r_t_dict

def construct_graph(graph_path):
    # construct original graph
    G = nx.MultiDiGraph()
    h_r_t_dict = dict() # dict[h][r] = [t_list]

    with open(graph_path, 'r') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            G.add_edge(h, t, r=r)
            h_r_t_dict = construct_h_r_t_dict(h_r_t_dict, h, r, t)
    return G, h_r_t_dict

def get_graph_dict(path):
    train_data = []
    with open(path) as f:
        for line in f.readlines():
            train_data.append(line.strip().split('\t'))

    graph_e_dict, graph_r_dict = dict(), dict()
    for i, data in enumerate(train_data):
        rel = data[0]
        if rel not in graph_r_dict:
            graph_r_dict[rel] = []
        graph_r_dict[rel].append(data)

        entities = data[1:]
        for e in entities:
            if e not in graph_e_dict:
                graph_e_dict[e] = []
            graph_e_dict[e].append([i, data])
    return train_data, graph_e_dict, graph_r_dict

def get_entity_rel_path(graph, source, target):
    # get the shortest path between h and t
    try:
        entity_list = nx.shortest_path(graph, source=source, target=target)
    except:
        return [], [], 0
    
    relation_list = []
    for i in range(1, len(entity_list)):
        r_list = []
        for j in graph.get_edge_data(entity_list[i-1],entity_list[i]):
            r_list.append(graph.get_edge_data(entity_list[i-1],entity_list[i])[j]['r'])
        relation_list.append(r_list)

    # TODO: max length
    # relation_list = [l for l in relation_list if len(l)<4]
    return entity_list, relation_list, len(relation_list)

def rule_single(instance, instance2rel, instance2entity_list, ent_q, ans_e):
    rel_rule = instance2rel[instance]
    index_list = []
    for k,e in enumerate(instance2entity_list[instance]):
        if ans_e != e:
            if e in ent_q:
                index_list.append(ent_q.index(e))
            else:
                index_list.append('pass')
        else:
            index_list.append('MASK')
    if len(index_list) > 0:
        return [rel_rule], index_list
    return [], []
'''
def rule_single(data, ent_q, ans_e):
    index_list = []
    for e in data[1:]:
        if e != ans_e:
            if e in ent_q:
                index_list.append(ent_q.index(e))
            else:
                index_list.append('pass')
        else:
            index_list.append('MASK')
    if len(index_list) > 0:
        return [data[0]], index_list
    return [], []
'''
# TODO
def rule_multi(instance2rel, instance2entity_list, rule_list, e_list, ent_q):
    # rule_list: [['make/750/1/0'], ['model/4558/2/0']]
    # e_list: ['m.01d9cv', 'm.04hx4n', 'm.03vnt4']
    rel_rule, index_list = [], []
    for i in range(len(rule_list)):
        [rel, ins_id, start_id, pivot_id] = rule_list[i][0].split('/')
        start_e, end_e = e_list[i], e_list[i+1]
        instance = 'instance{}'.format(ins_id)
        rule_list_step, index_list_step = rule_single(instance, instance2rel, instance2entity_list, ent_q, end_e)
        # rule_list_step: ['make']
        # index_list_step: ['MASK', 1]
        ent_q = instance2entity_list[instance]
        rel_rule.append(rule_list_step)
        index_list.append(index_list_step)
    # rel_rule: [['make'], ['model']
    # index_list: [['MASK', 1], ['MASK', 'pass', 0, 'pass', 'pass']]
    return rel_rule, index_list

def get_rules_dict_jf(train_data, graph, graph_e_dict, entity2instance, instance2entity_list, instance2rel):
    rules_dict = dict()
    for line_id, data in enumerate(tqdm(train_data, desc='Train')):
        rel = data[0]
        entities = data[1:]
        for loc_id, ans_e in enumerate(entities):
            query = rel + '_' + str(loc_id)
            flag = False

            q_ent = entities[:loc_id] + entities[loc_id+1:]
            q_ent_set = set(q_ent)
            for data_j_list in graph_e_dict[ans_e]:
                j, data_j_list = data_j_list[0], data_j_list[1]
                if line_id != j:
                    inter_set = q_ent_set.intersection(set(data_j_list))
                    if len(inter_set) > 0:
                        #rule_list, index_list = rule_single(data_j_list, entities, ans_e)
                        rule_list, index_list = rule_single('instance{}'.format(j), instance2rel, instance2entity_list, entities, ans_e)
                        if len(rule_list) > 0:
                            # rule_list: ['model']
                            # index_list: ['MASK', 1, 2, 3, 'pass']
                            flag = True
                            if query not in rules_dict:
                                rules_dict[query] = []
                            else:
                                rules_dict[query].append((rule_list, index_list))

            #if 1:
            if not flag:
                # multi-step
                # TODO: remove all outgoing edges from start_e in instance line_id
                for s_i, s_e in enumerate(entities):
                    for e_i, e_e in enumerate(entities):
                        if s_i != e_i:
                            r_name = '{}/{}/{}/{}'.format(rel, line_id, s_i, e_i)
                            try: graph.remove_edge(s_e, e_e, 'r'==r_name)
                            except: pass

                shortest_len = 0
                multi_rule_list, multi_e_list = [], []
                for k, start_e in enumerate(entities):
                    if loc_id != k:
                        entity_list, relation_list, rule_len = get_entity_rel_path(graph, start_e, ans_e)
                        # entity_list, relation_list: (['m.01d9cv', 'm.04hx4n', 'm.03vnt4'], [['make/750/1/0'], ['model/4558/2/0']], 2)
                        if rule_len != 0 and (shortest_len == 0 or rule_len <= shortest_len):
                            shortest_len = rule_len
                            multi_rule_list.append(relation_list)
                            multi_e_list.append(entity_list)

                # multi_rule_list: [[['make/750/1/0'], ['model/4558/2/0']], [['make/1454/0/1'], ['model/960/1/0']]]
                # multi_e_list: [['m.01d9cv', 'm.04hx4n', 'm.03vnt4'], ['m.0415dyw', 'm.01yc2q', 'm.03vnt4']]
                for mi in range(len(multi_rule_list)):
                    rule_list, index_list = rule_multi(instance2rel, instance2entity_list, multi_rule_list[mi], multi_e_list[mi], entities)
                    if len(rule_list) > 1:
                        if query not in rules_dict:
                            rules_dict[query] = []
                        rules_dict[query].append((rule_list, index_list))
    # rules_dict: [(['model'], ['MASK', 1, 'pass', 'pass', 'pass']), 
    #              ([['make'], ['model']], [['MASK', 1], ['MASK', 'pass', 0, 'pass', 'pass']])]

    return rules_dict


def aggregate_rules(rules):
    # aggregate and sort rules by their statistics
    new_rules_list = []
    new_rules_dict = dict() # count dict, dict[rule_index_in_list] = 1
    for rule in rules:
        if rule not in new_rules_list:
            new_rules_list.append(rule)
            index = new_rules_list.index(rule)
            w = sum([1 for item in rule[1] if isinstance(item, int)])/(len(rule[1])-1)
            new_rules_dict[index] = w
            #new_rules_dict[index] = 1
        else:
            index = new_rules_list.index(rule)
            w = sum([1 for item in rule[1] if isinstance(item, int)])/(len(rule[1])-1)
            new_rules_dict[index] += w
            #new_rules_dict[index] += 1
    new_rules_dict = {k: v for k, v in sorted(new_rules_dict.items(), key=lambda x:x[1], reverse=True)}
    if new_rules_dict: 
        if sum(new_rules_dict.values()) != 0.0:
            factor = 1.0/sum(new_rules_dict.values())
        else:
            factor = 1.0
    for k in new_rules_dict:
        new_rules_dict[k] = new_rules_dict[k] * factor
    return new_rules_list, new_rules_dict

def aggregate_rules_dict(rules_dict, rule_folder):
    new_rules_list_dict = dict()
    new_rules_dict_dict = dict()
    for r in rules_dict:
        new_rules_list, new_rules_dict = aggregate_rules(rules_dict[r])
        new_rules_list_dict[r] = new_rules_list
        new_rules_dict_dict[r] = new_rules_dict
    write_rules(new_rules_list_dict, new_rules_dict_dict, rule_folder)
    return new_rules_list_dict, new_rules_dict_dict

def write_rules(rules_list_dict, rules_dict_dict, rule_folder):
    for query in rules_list_dict:
        rules_list = rules_list_dict[query]
        rules_dict = rules_dict_dict[query]
        with open(os.path.join(rule_folder, query+'.txt'), 'w') as fw:
            for rule_index in rules_dict.keys():
                (rule, index_list) = rules_list[rule_index]
                weight = rules_dict[rule_index]
                #print(str(weight)+'\t'+'->'.join(rule)+'\t'+' '.join([str(i) for i in index_list]), file=fw)
                print(str(weight)+'\t'+str(rule)+'\t'+str(index_list), file=fw)

def reason_step(current_entity, rule, step, h_r_t_dict, weight):
    ans_list = []
    if len(rule) == step: # if this is the last step
        return (current_entity, weight)
    if current_entity in h_r_t_dict:
        r_dict = h_r_t_dict[current_entity]
        if rule[step] in r_dict.keys():
            current_entity_list = r_dict[rule[step]]
            weight = weight / len(current_entity_list)
            for e in current_entity_list:
                ans = reason_step(e, rule, step+1, h_r_t_dict, weight)
                if ans:
                    if type(ans) is list:
                        ans_list.extend(ans)
                    else:
                        ans_list.append(ans)
            if len(ans_list) > 0: return ans_list
            else: return None
        else:
            return None
    else:
        return None

def match_entity(relation, query, graph_r_dict, weight):
    ans_list = []
    pattern = relation + '\t'
    for q in query:
        if q == 'MASK' or q == 'pass':
            pattern += '([a-zA-Z0-9_.]+)\t'
        else:
            pattern += q
            pattern += '\t'
    pattern = pattern[:-1]
    for data in graph_r_dict[relation]:
        flag = re.match(pattern, '\t'.join(data))
        if flag:
            ans_list.append(data[1:][query.index('MASK')])
    ans_dict = Counter(ans_list)
    for a in ans_dict:
        ans_dict[a] = ans_dict[a] * weight
    return ans_dict

def match_entity2(relation, query, graph_r_dict):
    ans_list = []
    weight = []
    query_e_list = []
    # pattern = relation + '\t'
    pattern = ''
    for q in query:
        if q == 'MASK' or q == 'pass':
            pattern += '([a-zA-Z0-9_.]+)\t'
        else:
            pattern += re.escape(q)
            pattern += '\t'
    pattern = pattern[:-1]
    for data in graph_r_dict[relation]:
        flag = re.match(pattern, '\t'.join(data[1:]))
        if flag:
            mask_id = query.index('MASK')
            ans_list.append(data[1:][mask_id])
            query_e = data[1:]
            query_e_list.append(query_e)
    return ans_list, query_e_list

def match_entity_multi(query_e, ans, rule, index_list, step, graph_r_dict, ans_list):
    # rule: [['model'], ['model'], ['model']]
    # index_list: [['MASK', 1, 2, 3, 4], [0, 'MASK', 'pass', 'pass', 'pass'], ['MASK', 1, 'pass', 'pass', 'pass']]
    # query_e: ['None', 'm.0j6lyq1', 'm.064v_vt', 'm.0mzmdqk', 'm.02wwn3']

    answer_list = []

    if step == len(rule):
        return ans_list
    elif step == 0:
        query = construct_query(index_list[0], query_e)
        ans_list, query_e_list = match_entity2(rule[0][0], query, graph_r_dict)
        if N <= len(ans_list):
            index = random.sample(list(range(len(ans_list))), N)
            ans_list, query_e_list = list(np.asarray(ans_list)[index]), list(np.asarray(query_e_list)[index])
    else:
        #query = index_list[step].copy()
        #query[index_list[step-1].index('MASK')] = ans
        query = construct_query(index_list[step], query_e)
        ans_list, query_e_list = match_entity2(rule[step][0], query, graph_r_dict)
        if N <= len(ans_list):
            index = random.sample(list(range(len(ans_list))), N)
            ans_list, query_e_list = list(np.asarray(ans_list)[index]), list(np.asarray(query_e_list)[index])
    if step != 0 and len(ans_list) == 0:
        return []
    for i, ans in enumerate(ans_list):
        answer_list.extend(match_entity_multi(query_e_list[i], ans_list[i], rule, index_list, step+1, graph_r_dict, ans_list))

    return answer_list

def construct_query(index_list, query_e):
    query = []
    for i in index_list:
        if i == 'MASK': query.append('MASK')
        elif i == 'pass': query.append('pass')
        else: query.append(query_e[i])
    return query

def reason_single_tail_list(query_e, rules_list, rules_dict, graph_r_dict):
    # reason for a single sample
    answer_dict = Counter()
    #count = 0 # only use the first 5 rules when reasoning
    for rule_index in rules_dict.keys():
        #if count >= 10: break
        (rule, index_list) = rules_list[rule_index] # ['model'], ['MASK',1,'pass',3,2]
        weight = rules_dict[rule_index]

        if len(rule) == 0: continue
        if len(rule) == 1: # single rule
            query = construct_query(index_list, query_e)
            ans_dict_single_rule = match_entity(rule[0], query, graph_r_dict, weight)
            for k,v in ans_dict_single_rule.items():
                if k in answer_dict: answer_dict[k] += v
                else: answer_dict[k] = v
        elif len(rule) > 1:
            ans_list_single_rule = match_entity_multi(query_e, None, rule, index_list, 0, graph_r_dict, [])
            ans_dict_single_rule = Counter(ans_list_single_rule)
            for k,v in ans_dict_single_rule.items():
                if k in answer_dict: answer_dict[k] += v*weight
                else: answer_dict[k] = v*weight
        #count += 1
    # answer_dict do not need sort
    return answer_dict

def reason_tail_whole_jf(test_path, rules_list_dict, rules_dict_dict, datafolder, graph_r_dict):
    # reason for a dataset with different query relations
    all_ans = []
    all_golden_ans = []
    primary_ans = []
    primary_golden_ans = []

    test_data = []
    with open(test_path) as f:
        for line in f.readlines():
            test_data.append(line.strip().split('\t'))

    true_ans = [] # [1or0, ..]
    for data in tqdm(test_data, desc='Test'):
        rel = data[0]
        entities = data[1:]
        for i,ans_e in enumerate(entities):
            query = rel+'_'+str(i)
            query_e = entities[:i]+['None']+entities[i+1:]
            all_golden_ans.append(ans_e)
            if i==0 or i==1:
                primary_golden_ans.append(ans_e)

            if query in rules_list_dict:
                tail_entity = reason_single_tail_list(query_e, rules_list_dict[query], rules_dict_dict[query], graph_r_dict)
                all_ans.append(tail_entity)
                if i==0 or i==1:
                    primary_ans.append(tail_entity)
                if ans_e in tail_entity: true_ans.append(1)
            else:
                all_ans.append(Counter(dict()))
                if i==0 or i==1:
                    primary_ans.append(Counter(dict()))
                true_ans.append(0)
    write_ans(all_ans, os.path.join(datafolder, 'all_ans.txt'))
    write_ans(all_golden_ans, os.path.join(datafolder, 'all_golden_ans.txt'))
    write_ans(primary_ans, os.path.join(datafolder, 'primary_ans.txt'))
    write_ans(primary_golden_ans, os.path.join(datafolder, 'primary_golden_ans.txt'))
    acc = sum(true_ans) / len(true_ans)
    return acc

def write_ans(ans, file_path):
    with open(file_path, 'w') as fw:
        for t in ans:
            print(t, file=fw)

def process_whole(dataset):
    datafolder = './data_preprocessed/{}/'.format(dataset)
    graph_path = 'graph.txt'
    train_path = 'train.txt'
    test_path = 'test.txt'
    entity2instance_path = 'entity2instance.pkl'
    instance2entity_list_path = 'instance2entity.pkl'
    instance2rel_path = 'instance2rel.pkl'
    #line_graph_path = 'line_graph.txt'

    rule_folder_path = './rules/{}/'.format(dataset)
    if not os.path.isdir(rule_folder_path):
        os.makedirs(rule_folder_path)
    #rule_list_path = 'rule_list.pickle'
    #rule_dict_path = 'rule_dict.pickle'

    entity2instance = read_pickle(os.path.join(datafolder, entity2instance_path))
    instance2entity_list = read_pickle(os.path.join(datafolder, instance2entity_list_path))
    instance2rel = read_pickle(os.path.join(datafolder, instance2rel_path))
    train_data, graph_e_dict, graph_r_dict = get_graph_dict(os.path.join(datafolder, train_path))
    graph, _ = construct_graph(os.path.join(datafolder, graph_path))
    #line_graph, line_h_r_dict = construct_graph(os.path.join(datafolder, line_graph_path))
    rules_dict = get_rules_dict_jf(train_data, graph, graph_e_dict, entity2instance, instance2entity_list, instance2rel)
    rules_list_dict, rules_dict_dict = aggregate_rules_dict(rules_dict, rule_folder_path)
    acc = reason_tail_whole_jf(os.path.join(datafolder, test_path), rules_list_dict, rules_dict_dict, datafolder, graph_r_dict)
    print('{}: {}'.format(dataset, acc))


#process_whole('FB-AUTO')
#process_whole('M-FB15K')

# datasets: M-FB15K, FB-AUTO
dataset = sys.argv[1]
N = int(sys.argv[2])
print(dataset)
process_whole(dataset)
