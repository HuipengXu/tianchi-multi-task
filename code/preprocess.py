import re
import os
import json
import shutil
from tqdm import tqdm
from collections import defaultdict

import pandas as pd


def clean_duplication(text):
    left_square_brackets_pat = re.compile(r'\[+')
    right_square_brackets_pat = re.compile(r'\]+')
    punct = [',', '\\.', '\\!', '，', '。', '！', '、', '\?', '？']

    def replace(string, char):
        pattern = char + '{2,}'
        if char.startswith('\\'):
            char = char[1:]
        string = re.sub(pattern, char, string)
        return string

    text = left_square_brackets_pat.sub('', text)
    text = right_square_brackets_pat.sub('', text)
    for p in punct:
        text = replace(text, p)
    return text


def emoji2zh(text, inverse_emoji_dict):
    for emoji, ch in inverse_emoji_dict.items():
        text = text.replace(emoji, ch)
    return text


def clean_emotion(data_path, emoji2zh_data, save_dir, train=True):
    data = defaultdict(list)
    filename = os.path.basename(data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        texts = f.readlines()
        for line in tqdm(texts, desc=data_path):
            if train:
                id_, text, label = line.strip().split('\t')
            else:
                id_, text = line.strip().split('\t')
            data['id'].append(id_)
            text = emoji2zh(text, emoji2zh_data)
            text = clean_duplication(text)
            data['text'].append(text)
            if train:
                data['label'].append(label)
    df = pd.DataFrame(data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(os.path.join(save_dir, filename), index=False,
              encoding='utf8', header=False, sep='\t')
    return df


def clean_nli(data_path, save_dir, train=True):
    data = defaultdict(list)
    filename = os.path.basename(data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        texts = f.readlines()
        for line in tqdm(texts, desc=data_path):
            if train:
                id_, text_a, text_b, label = line.strip().split('\t')
            else:
                id_, text_a, text_b = line.strip().split('\t')
            data['id'].append(id_)
            data['premise'].append(text_a)
            data['hypothesis'].append(text_b)
            if train:
                data['label'].append(label)
    df = pd.DataFrame(data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(os.path.join(save_dir, filename), index=False,
              encoding='utf8', header=False, sep='\t')
    return df


def main():
    """只处理了 nli 和 emotion 数据
    """
    emoji2zh_data = json.load(open('../user_data/emoji2zh.json', 'r', encoding='utf8'))
    data_a_dir = '../tcdata/nlp_round1_data'
    data_b_dir = '../tcdata/nlp_round2_data'
    cleaned_a_dir = '../user_data/cleaned_nlp_round1_data'
    cleaned_b_dir = '../user_data/cleaned_nlp_round2_data'
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
    clean_emotion(os.path.join(data_a_dir, task_data['ocemotion']['train']),
                  emoji2zh_data,
                  cleaned_a_dir)
    clean_emotion(os.path.join(data_a_dir, task_data['ocemotion']['predict']),
                  emoji2zh_data,
                  cleaned_a_dir,
                  train=False)
    clean_emotion(os.path.join(data_b_dir, task_data['ocemotion']['test_b']),
                  emoji2zh_data,
                  cleaned_b_dir,
                  train=False)
    clean_nli(os.path.join(data_a_dir, task_data['ocnli']['train']),
              cleaned_a_dir)
    clean_nli(os.path.join(data_a_dir, task_data['ocnli']['predict']),
              cleaned_a_dir,
              train=False)
    clean_nli(os.path.join(data_b_dir, task_data['ocnli']['test_b']),
              cleaned_b_dir,
              train=False)
    print('Start copying tnews data to cleaned directory ...')
    shutil.copy(os.path.join(data_a_dir, task_data['tnews']['train']), cleaned_a_dir)
    shutil.copy(os.path.join(data_a_dir, task_data['tnews']['predict']), cleaned_a_dir)
    shutil.copy(os.path.join(data_b_dir, task_data['tnews']['test_b']), cleaned_b_dir)
    print('Finished copying tnews data to cleaned directory ...')

    print('Start copying train data to test b directory ...')
    shutil.copy(os.path.join(cleaned_a_dir, task_data['ocemotion']['train']), cleaned_b_dir)
    shutil.copy(os.path.join(cleaned_a_dir, task_data['ocnli']['train']), cleaned_b_dir)
    shutil.copy(os.path.join(cleaned_a_dir, task_data['tnews']['train']), cleaned_b_dir)
    print('Finished copying train data to test b directory ...')


if __name__ == '__main__':
    main()
