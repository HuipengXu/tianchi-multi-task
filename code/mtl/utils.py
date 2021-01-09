import os
import json
import random
from collections import defaultdict
from typing import Any, Optional, Dict, Iterable, List, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset, BatchSampler, Sampler
from transformers import BertTokenizer
from transformers.trainer_utils import EvalPrediction

task_num_classes = {'0': 3, '1': 7, '2': 15}
task_id_to_name = {'0': 'ocnli', '1': 'ocemotion', '2': 'tnews'}
task_lambda = {'0': 1.0, '1': 1.5, '2': 1.5}
task_data = {
    'ocemotion': {'predict': 'OCEMOTION_a.csv',
                  'train': 'OCEMOTION_train1128.csv',
                  'test_b': 'ocemotion_test_B.csv'},
    'ocnli': {'predict': 'OCNLI_a.csv',
              'train': 'OCNLI_train1128.csv',
              'test_b': 'ocnli_test_B.csv'},
    'tnews': {'predict': 'TNEWS_a.csv',
              'train': 'TNEWS_train1128.csv',
              'test_b': 'tnews_test_B.csv'}
}


def load_json(file_path):
    return json.load(open(file_path, 'r', encoding='utf8'))


def get_df(data_dir: str, data_name: str) -> pd.DataFrame:
    data_path = os.path.join(data_dir, data_name)
    df = pd.read_csv(data_path, sep='\t', header=None)
    return df


def preprocess(args: Any, task: str, test_b: bool = False):
    data_name = task_data[task]
    if test_b:
        print('Inference stage ...')
        data_name['predict'] = data_name['test_b']
    train_df = get_df(args.data_dir, data_name['train'])
    pred_df = get_df(args.data_dir, data_name['predict'])

    pretrained_model_path = args.tokenizer_dir
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    train_precessed, val_precessed, label2id = convert_df_to_inputs(task, tokenizer, train_df,
                                                                    args.train_val_split_ratio, debug=args.debug)
    predict_precessed, = convert_df_to_inputs(task, tokenizer, pred_df, label2id=label2id, debug=args.debug)

    data_save_dir = os.path.join(args.data_save_dir, task)
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    print(f'Saving processed {task} data ...')
    cache_data_path = {
        'train': os.path.join(args.data_save_dir, task, 'train.pt'),
        'val': os.path.join(args.data_save_dir, task, 'val.pt'),
        'predict': os.path.join(args.data_save_dir, task, 'predict.pt')
    }
    json.dump(label2id, open(os.path.join(data_save_dir, 'label2id.json'), 'w'))
    torch.save(train_precessed, cache_data_path['train'])
    torch.save(val_precessed, cache_data_path['val'])
    torch.save(predict_precessed, cache_data_path['predict'])

    return cache_data_path


def convert_label_to_id(targets_series: pd.Series, label2id: Optional[dict] = None) -> tuple:
    labels = np.unique(targets_series.values)
    train = False
    if label2id is None:
        train = True
        label2id = {str(label): i for i, label in enumerate(labels)}
    targets_series = targets_series.apply(lambda label: str(label))
    targets_series = targets_series.apply(lambda label: label2id[label])
    targets = torch.from_numpy(targets_series.values.astype('int64'))
    outputs = (targets,)
    if train:
        outputs += (label2id,)
    return outputs


def convert_df_to_inputs(task: str, tokenizer: BertTokenizer, df: pd.DataFrame,
                         train_val_split_ratio: Optional[float] = None,
                         label2id: Optional[dict] = None, debug: bool = False) -> tuple:
    inputs = defaultdict(list)
    train = False
    if debug:
        df = df.head(1000)
    df.sample(frac=1, replace=True, random_state=32)
    label2id, train = _iter_row(df, inputs, task, tokenizer, train, train_val_split_ratio, label2id)

    if train_val_split_ratio is not None:
        outputs = train_val_split(inputs, train_val_split_ratio)
    else:
        outputs = (inputs,)

    if train:
        outputs += (label2id,)

    return outputs


def _iter_row(df, inputs: dict, task: str, tokenizer: BertTokenizer, train: bool,
              train_val_split_ratio: float, label2id: Optional[dict] = None) -> Tuple[dict, bool]:
    targets = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Preprocess {task}'):
        text_a = row[1]
        if task == 'ocnli':
            target_idx = 3
            text_b = row[2]
            output_ids = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True,
                                               return_token_type_ids=True, return_attention_mask=True)
        else:
            target_idx = 2
            output_ids = tokenizer.encode_plus(text_a, add_special_tokens=True,
                                               return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(output_ids['input_ids'])
        inputs['token_type_ids'].append(output_ids['token_type_ids'])
        inputs['attention_mask'].append(output_ids['attention_mask'])

        if train_val_split_ratio is not None:
            targets.append(row[target_idx])
        else:
            targets.append(list(label2id.keys())[0])
    targets_series = pd.Series(targets)
    if label2id is None:
        train = True
        targets, label2id = convert_label_to_id(targets_series)
    else:
        targets, = convert_label_to_id(targets_series, label2id)
    inputs['targets'] = targets
    return label2id, train


def train_val_split(inputs, train_val_split_ratio):
    num_val = int(len(inputs['input_ids']) * train_val_split_ratio)
    train_data = {}
    val_data = {}
    for key, tensor in inputs.items():
        train_data[key] = tensor[num_val:]
        val_data[key] = tensor[:num_val]
    outputs = (train_data, val_data)
    return outputs


class SingleTaskDataset(Dataset):

    def __init__(self, data_path: str):
        super(SingleTaskDataset, self).__init__()
        self.data_dict = torch.load(data_path)

    def __getitem__(self, index: int) -> tuple:
        return (self.data_dict['input_ids'][index], self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index], self.data_dict['targets'][index])

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class MultiTaskDataset(Dataset):

    def __init__(self, datasets: Dict[str, SingleTaskDataset]):
        super(MultiTaskDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, index: tuple) -> dict:
        task_id, dataset_index = index
        return {'task_id': task_id, 'data': self.datasets[task_id][dataset_index]}

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())


class MultiTaskBatchSampler(BatchSampler):

    def __init__(self, datasets: MultiTaskDataset, batch_size: int, shuffle=True):
        super(MultiTaskBatchSampler, self).__init__(sampler=Sampler(datasets), batch_size=batch_size,
                                                    drop_last=False)
        self.datasets_length = {task_id: len(dataset) for
                                task_id, dataset in datasets.datasets.items()}
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.task_indexes = []
        self.batch_indexes = {}
        self.task_num_batches = {}
        self.total_batches = 0
        self.init()

    def init(self):
        for task_id, dataset_len in self.datasets_length.items():
            num_batches = (dataset_len - 1) // self.batch_size + 1
            self.batch_indexes[task_id] = list(range(dataset_len))
            self.task_num_batches[task_id] = num_batches
            self.total_batches += num_batches
            self.task_indexes.extend([task_id] * num_batches)

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterable:
        batch_generator = self.get_batch_generator()
        for task_id in self.task_indexes:
            current_indexes_gen = batch_generator[task_id]
            batch = next(current_indexes_gen)
            yield [(task_id, index) for index in batch]

    def get_batch_generator(self) -> Dict[str, Iterable]:
        if self.shuffle:
            random.shuffle(self.task_indexes)
        batch_generator = {}
        for task_id, batch_indexes in self.batch_indexes.items():
            if self.shuffle:
                random.shuffle(batch_indexes)
            batch_generator[task_id] = iter([batch_indexes[i * self.batch_size: (i + 1) * self.batch_size]
                                             for i in range(self.task_num_batches[task_id])])
        return batch_generator


def collate_fn(examples: List[dict], max_seq_len: int) -> dict:
    task_ids = []
    data = []
    for example in examples:
        task_ids.append(example['task_id'])
        data.append(example['data'])

    assert (np.array(task_ids) == task_ids[0]).all(), 'batch data must belong to the same task.'

    input_ids_list, token_type_ids_list, attention_mask_list, targets_list = list(zip(*data))

    cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
    max_seq_len = min(cur_max_seq_len, max_seq_len)
    input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(input_ids_list)):
        seq_len = min(len(input_ids_list[i]), max_seq_len)
        if seq_len <= max_seq_len:
            input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
        else:
            input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] + [102], dtype=torch.long)
        token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
    labels = torch.tensor(targets_list, dtype=torch.long)

    return {
        'task_id': task_ids[0],
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def compute_metrics(results: EvalPrediction) -> float:
    f1 = f1_score(results.predictions, results.label_ids, average='macro')
    return f1


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
