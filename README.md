# HyperPath
Code for paper "Symbolic Knowledge Reasoning on Hyper-Relational Knowledge Graphs"

Datasets: Please place the original datasets in the "data_original" folder.

For single-hop reasoning on datasets with only one main relation
(on datasets JF17K, JF17K-3, JF17K-4, WikiPeople-3, WikiPeople-4, FB-AUTO, M-FB15K):
```python
python preprocess_single.py DATASET_NAME
python reason_single.py DATASET_NAME
python eval.py DATASET_NAME
```

For single-hop reasoning on datasets with key-value pairs
(on datasets WikiPeople, WD50K, WD50K_33, WD50K_66, WD50K_100):
```python
python preprocess_single_key.py DATASET_NAME
python reason_single_key.py DATASET_NAME
python eval.py DATASET_NAME
```

For multi-hop reasoning on all datasets:
```python
python preprocess_multi.py DATASET_NAME
python reason_multi.py DATASET_NAME N
python eval.py DATASET_NAME
```
N is the number of selected neighbors at each step.
