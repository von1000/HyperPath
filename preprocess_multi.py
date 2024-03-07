import os
import shutil
import sys
import pickle

def filter(data):
    new_data = []
    for d in data:
        if 'wikidata' in d:
            new_data.append(d[1:5])
        else:
            new_data.append(d)
    return new_data

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

def get_graph_file(dataset_name, original_file_path, processed_folder_path):
    new_data = []
    if dataset_name in ['M-FB15K', 'FB-AUTO', 'FB-AUTO-2', 'FB-AUTO-4', 'FB-AUTO-5', 'JF17K-4', 'JF17K-2', 'JF17K-5', 'JF17K-6', 'JF17K', 'WikiPeople-4', 'JF17K_new', 'WD50K_100-3', 'WD50K_100-4', 'WD50K_100-5', 'WD50K_100-6']:
        with open(original_file_path) as f:
            for index, line in enumerate(f.readlines()):
                data = line.strip().split('\t')
                rel = data[0]
                entities = data[1:]
                for i,e1 in enumerate(entities):
                    for j,e2 in enumerate(entities):
                        if i != j:
                            new_data.append([e2, rel+'/'+str(index)+'/'+str(j)+'/'+str(i), e1])
    elif dataset_name in ['JF17K-3', 'WikiPeople-3']:
        with open(original_file_path) as f:
            for index, line in enumerate(f.readlines()):
                data = line.strip().split(' ')
                rel = data[0]
                entities = data[1:]
                for i,e1 in enumerate(entities):
                    for j,e2 in enumerate(entities):
                        if i != j:
                            new_data.append([e2, rel+'/'+str(index)+'/'+str(j)+'/'+str(i), e1])

    new_data = [list(j) for j in list(set([tuple(i) for i in new_data]))] # remove duplicates

    with open(os.path.join(processed_folder_path, 'graph.txt'), 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')


def get_pickle_file(dataset_name, original_file_path, processed_folder_path):
    all_data = []
    all_edge = []
    entity2instance = dict()
    instance2entity = dict()
    instance2rel = dict()

    if dataset_name in ['M-FB15K', 'FB-AUTO', 'FB-AUTO-2', 'FB-AUTO-4', 'FB-AUTO-5', 'JF17K-4', 'JF17K-2', 'JF17K-5', 'JF17K-6', 'JF17K', 'WikiPeople-4', 'WD50K_100-3', 'WD50K_100-4', 'WD50K_100-5', 'WD50K_100-6']:
        with open(original_file_path) as f:
            for line in f.readlines():
                data = line.strip().split('\t')
                all_edge.append(data[0])
                all_data.append(data)
    elif dataset_name in ['JF17K-3', 'WikiPeople-3']:
        with open(original_file_path) as f:
            for line in f.readlines():
                data = line.strip().split(' ')
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

def format_file(original_file, processed_file):
    new_data = []
    with open(original_file) as f:
        for line in f.readlines():
            data = line.strip().split(' ')
            new_data.append(data)

    with open(processed_file, 'w') as fw:
        for new_d in new_data:
            fw.write('\t'.join(new_d) + '\n')

def process_folder(dataset_name):
    if dataset_name in ['M-FB15K', 'FB-AUTO', 'JF17K-4']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        get_graph_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        shutil.copyfile(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        #shutil.copyfile(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        shutil.copyfile(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
    elif dataset_name in ['WikiPeople-4']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_wiki_txt(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_wiki_txt(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_wiki_txt(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['JF17K-3']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        get_graph_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        format_file(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        format_file(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        format_file(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
    elif dataset_name in ['JF17K']:
        original_folder = './data_original/{}/instances'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        get_graph_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        shutil.copyfile(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_test_file(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
    elif dataset_name in ['WikiPeople-3']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_wiki2_txt(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_wiki2_txt(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_wiki2_txt(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['JF17K_new']:
        original_folder = '../hkg_logic2/data_preprocessed/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        get_graph_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        shutil.copyfile(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        shutil.copyfile(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
    elif dataset_name in ['JF17K-2', 'JF17K-5', 'JF17K-6', 'FB-AUTO-2', 'FB-AUTO-4', 'FB-AUTO-5', 'WD50K_100-3', 'WD50K_100-4', 'WD50K_100-5', 'WD50K_100-6']:
        original_folder = './data_preprocessed/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        get_graph_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)


dataset = sys.argv[1]
process_folder(dataset)

