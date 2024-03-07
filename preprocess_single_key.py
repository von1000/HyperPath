import os
import shutil
import sys
import pickle
import json

def filter_data(data, train_path):
    pass

def get_graph_file(dataset_name, original_file_path, processed_folder_path):
    new_data = []
    if dataset_name in ['M-FB15K', 'FB-AUTO', 'JF17K-4', 'JF17K', 'WikiPeople-4', 'WD50K_100', 'WD50K_66', 'WD50K_33', 'WD50K', 'WikiPeople', 'WikiPeople_minus']:
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

    if dataset_name in ['M-FB15K', 'FB-AUTO', 'JF17K-4', 'JF17K', 'WikiPeople-4', 'WD50K_100', 'WD50K_66', 'WD50K_33', 'WD50K', 'WikiPeople', 'WikiPeople_minus']:
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

def process_wd_file(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            data_d = []
            data = line.strip().split(',')
            h, r, t = data[0], data[1], data[2]
            r = r+'-'+str(len(data))
            data_d.append(r)
            for i in range(0, len(data), 2):
                data_d.append(data[i])
            new_data.append(data_d)
    with open(processed_path, 'w') as fw:
        for d in new_data:
            fw.write('\t'.join(d)+'\n')

def filter_value(v):
    if 'wikidata' in v:
        return v[1:5]
    else:
        return v

def expand_json(json_obj):
    def expand(item, current):
        if not item:
            expanded.append(current)
            return
        key, values = item[0]
        if isinstance(values, list):
            for value in values:
                new_current = current.copy()
                new_current[key] = value
                expand(item[1:], new_current)
        else:
            new_current = current.copy()
            new_current[key] = values
            expand(item[1:], new_current)

    expanded = []
    items = [(key, value) for key, value in json_obj.items()]
    expand(items, {})
    return expanded

def expand_json_rel(json_obj, rel):
    def expand(item, current):
        if not item:
            expanded.append(current)
            return
        key, values = item[0]
        if key == rel:
            new_current = current.copy()
            new_current[key] = values
            expand(item[1:], new_current)
        elif isinstance(values, list):
            for value in values:
                new_current = current.copy()
                new_current[key] = value
                expand(item[1:], new_current)
        else:
            new_current = current.copy()
            new_current[key] = values
            expand(item[1:], new_current)

    expanded = []
    items = [(key, value) for key, value in json_obj.items()]
    expand(items, {})
    return expanded

def process_json_file(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            line_d = json.loads(line.strip())
            expanded_list = expand_json(line_d)
            for expanded_json in expanded_list:
                data_d = []
                for k,v in expanded_json.items():
                    if k == 'N': continue
                    if k[-2:] == '_h' or k[-2:] == '_t':
                        rel = k[:-2]
                    if len(data_d) == 0:
                        data_d.append(rel+'-'+str(expanded_json['N']))
                    data_d.append(filter_value(v))
                new_data.append(data_d)
    with open(processed_path, 'w') as fw:
        for d in new_data:
            fw.write('\t'.join(d) + '\n')

def process_json_file2(original_path, processed_path):
    new_data = []
    with open(original_path) as f:
        for line in f.readlines():
            line_d = json.loads(line.strip())
            rel = line.split(':')[0][2:-1]
            expanded_list = expand_json_rel(line_d, rel)
            for expanded_json in expanded_list:
                data_d = []
                for k,v in expanded_json.items():
                    if len(data_d) == 0:
                        data_d.append(rel+'-'+str(expanded_json['N']))
                        data_d.append(expanded_json[rel][0])
                        data_d.append(expanded_json[rel][1])
                    if k == 'N' or k == rel: continue
                    data_d.append(filter_value(v))
                new_data.append(data_d)
    with open(processed_path, 'w') as fw:
        for d in new_data:
            fw.write('\t'.join(d) + '\n')

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
    if dataset_name in ['M-FB15K', 'FB-AUTO', 'JF17K-4', 'WikiPeople-4']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        get_graph_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(original_folder, 'train.txt'), processed_folder)
        shutil.copyfile(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        shutil.copyfile(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        shutil.copyfile(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
    elif dataset_name in ['JF17K-3', 'WikiPeople-3']:
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
    elif dataset_name in ['WD50K', 'WD50K_33', 'WD50K_66', 'WD50K_100']:
        original_folder = './data_original/WD50K/{}/statements'.format(dataset_name.lower())
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_wd_file(os.path.join(original_folder, 'train.txt'), os.path.join(processed_folder, 'train.txt'))
        process_wd_file(os.path.join(original_folder, 'valid.txt'), os.path.join(processed_folder, 'valid.txt'))
        process_wd_file(os.path.join(original_folder, 'test.txt'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['WikiPeople']:
        original_folder = './data_original/{}/{}'.format(dataset_name, dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_json_file(os.path.join(original_folder, 'n-ary_train.json'), os.path.join(processed_folder, 'train.txt'))
        process_json_file(os.path.join(original_folder, 'n-ary_dev.json'), os.path.join(processed_folder, 'valid.txt'))
        process_json_file(os.path.join(original_folder, 'n-ary_test.json'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
    elif dataset_name in ['WikiPeople_minus']:
        original_folder = './data_original/{}'.format(dataset_name)
        processed_folder = './data_preprocessed/{}'.format(dataset_name)
        if not os.path.isdir(processed_folder):
            os.makedirs(processed_folder)
        process_json_file2(os.path.join(original_folder, 'n-ary_train.json'), os.path.join(processed_folder, 'train.txt'))
        process_json_file2(os.path.join(original_folder, 'n-ary_valid.json'), os.path.join(processed_folder, 'valid.txt'))
        process_json_file2(os.path.join(original_folder, 'n-ary_test.json'), os.path.join(processed_folder, 'test.txt'))
        get_graph_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)
        get_pickle_file(dataset_name, os.path.join(processed_folder, 'train.txt'), processed_folder)



dataset = sys.argv[1]
process_folder(dataset)

