import os
import shutil
import sys
import pickle
import json
import pdb

def filter(data):
    new_data = []
    for d in data:
        if 'wikidata' in d:
            new_data.append(d[1:5])
        else:
            new_data.append(d)
    return new_data

def get_graph_file(original_file_path, processed_folder_path):
    new_data = []
    with open(original_file_path) as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            rel = data[0]
            entities = data[1:]
            for i,e1 in enumerate(entities):
                for j,e2 in enumerate(entities):
                    if i != j:
                        new_data.append([e2, rel+'_'+str(i), e1])

    new_data = [list(j) for j in list(set([tuple(i) for i in new_data]))] # remove duplicates

    with open(os.path.join(processed_folder_path, 'graph.txt'), 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')

def process_file(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            data = line.strip().split(' ')
            new_data.append(data)
    with open(processed_path, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')

def process_file2(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            data = []
            data_original = line.strip().split(',')
            data.append(data_original[1][:-1])
            for i in range(0, len(data_original), 2):
                data.append(data_original[i])
            new_data.append(data)
    with open(processed_path, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')

def process_wiki_txt(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            new_data.append(filter(data))
    with open(processed_path, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')

def process_wiki2_txt(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            data = line.strip().split(' ')
            new_data.append(filter(data))
    with open(processed_path, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')

def process_json_file(original_path, processed_path):
    data = []
    with open(original_path) as f:
        for line in f.readlines():
            line_d = json.loads(line.strip())
            for k,v in line_d.items():
                if k[-2:] == '_h' or k[-2:] == '_t':
                    rel = k[:-2]

def process_wd_file(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            data = line.strip().split(',')
            k = data[0]
            data[0] = data[1]
            data[1] = k
            new_data.append(data)
    with open(processed_path, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')


def get_line_graph_file_wd(original_file_path, processed_folder_path):
    all_data = []
    #all_edge = []
    entity2instance = dict()
    instance2entity = dict()
    instance2rel = dict()

    with open(original_file_path) as f:
        for line in f.readlines():
            data = line.strip().split(',')
            #all_edge.append(data[0])
            all_data.append(data)

    all_name = ['instance'+str(i) for i in range(len(all_data))]

    for i in range(len(all_data)):
        rel = all_data[i][1]
        entities = [all_data[i][j] for j in range(len(all_data[i])) if j%2==0]
        name = all_name[i]
        if name not in instance2entity: instance2entity[name] = []
        for e in entities:
            if e not in entity2instance: entity2instance[e] = set()
            entity2instance[e].add(name)
            instance2entity[name].append(e)
        instance2rel[name] = rel


    with open(os.path.join(processed_folder_path, 'entity2instance.pkl'), 'wb') as fw:
        pickle.dump(entity2instance, fw)

    with open(os.path.join(processed_folder_path, 'instance2entity.pkl'), 'wb') as fw:
        pickle.dump(instance2entity, fw)

    with open(os.path.join(processed_folder_path, 'instance2rel.pkl'), 'wb') as fw:
        pickle.dump(instance2rel, fw)


def get_line_graph_file(original_file_path, processed_folder_path):
    all_data = []
    all_edge = []
    entity2instance = dict()
    instance2entity = dict()
    instance2rel = dict()

    with open(original_file_path) as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            all_edge.append(data[0])
            all_data.append(data)

    all_name = ['instance'+str(i) for i in range(len(all_data))]

    for i in range(len(all_data)):
        rel = all_data[i][0]
        entities = all_data[i][1:]
        name = all_name[i]
        if name not in instance2entity: instance2entity[name] = []
        for e in entities:
            if e not in entity2instance: entity2instance[e] = set()
            entity2instance[e].add(name)
            instance2entity[name].append(e)
        instance2rel[name] = rel


    with open(os.path.join(processed_folder_path, 'entity2instance.pkl'), 'wb') as fw:
        pickle.dump(entity2instance, fw)

    with open(os.path.join(processed_folder_path, 'instance2entity.pkl'), 'wb') as fw:
        pickle.dump(instance2entity, fw)

    with open(os.path.join(processed_folder_path, 'instance2rel.pkl'), 'wb') as fw:
        pickle.dump(instance2rel, fw)


def process_test_file(original_file, processed_file):
    new_data = []
    with open(original_file) as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            new_data.append(data[1:])

    with open(processed_file, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')


def process_folder(dataset_name):
    # single reason do not need graph.txt
    if dataset_name in ['M-FB15K', 'FB-AUTO', 'JF17K-4']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        shutil.copyfile(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        shutil.copyfile(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        shutil.copyfile(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(original_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(original_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['WikiPeople-4']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_wiki_txt(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_wiki_txt(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_wiki_txt(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['JF17K-3']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_file(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_file(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_file(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['WikiPeople-3']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_wiki2_txt(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_wiki2_txt(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_wiki2_txt(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['JF17K']:
        original_folder = './data_original/{}/instances'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        shutil.copyfile(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_test_file(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['JF17K_filter', 'JF17K_filter2', 'FB-AUTO_filter2']:
        # should run filter_jf.py first
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['WikiPeople']:
        original_folder = './data_original/{}/{}'.format(dataset_name, dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_json_file(os.path.join(original_folder, 'n-ary_train.json'), os.path.join(processed_folder, 'train.txt'))
        process_json_file(os.path.join(original_folder, 'n-ary_dev.json'), os.path.join(processed_folder, 'valid.txt'))
        process_json_file(os.path.join(original_folder, 'n-ary_test.json'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['WD50K', 'WD50K_33', 'WD50K_66', 'WD50K_100']:
        original_folder = './data_original/WD50K/{}/statements'.format(dataset_name.lower())
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_wd_file(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_wd_file(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_wd_file(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file_wd(os.path.join(original_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['JF17K_new']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_file2(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_file2(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_line_graph_file(os.path.join(processed_folder, 'train.txt'), processed_folder)




dataset = sys.argv[1]
process_folder(dataset)

