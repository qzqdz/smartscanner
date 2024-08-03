# -*- encoding: utf-8 -*-
import argparse
import json
import os
import random
from typing import List
import pandas as pd
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model import SimcseModel

from src.modeling.network.rankcse.teachers import *
import copy
from torchinfo import summary


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
python your_script_name.py --do_train --epochs 5 --batch_size 8 --lr 1e-4 --pooling last-avg --model_path "/path/to/model" --use_teacher --use_teacher_embedding --teacher_model "/path/to/teacher/model" --snli_train "/path/to/train/data" --sts_dev "/path/to/dev/data" --sts_test "/path/to/test/data" --acc_train "/path/to/test/data" --acc_val  "/path/to/test/data" --acc_batch_size 4 --acc_maxlen 64 --acc_k 5 --seed 3402
'''


# # 基本参数
# do_train = True
# # do_train = False
# EPOCHS = 5
# BATCH_SIZE = 8

# # CWC rate
# # LR = 5e-5
# LR = 1e-4
# # BERT rate
# # LR = 1e-5

# MAXLEN = 64
# # POOLING = 'cls'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
# POOLING = 'last-avg'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# # 预训练模型目录
# BERT4 = r'E:\model\transformers4\chinse_simcse_roberta'
# BERT3 = r'E:\model\white_model\roberta-chinese-w'
# BERT2 = r'E:\model\white_model\chinesebert'
# BERT = 'pretrained_model/bert_pytorch'
# BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
# model_path = BERT4

# # 教师模型参数
# use_teacher = True
# use_teacher_embedding = False
# teacher_model = r'E:\model\transformers4\chinse_simcse_roberta'
# teacher_pooler = 'cls_before_pooler'
# tau2 = 0.05

# # 微调后参数存放位置
# SAVE_PATH = './saved_model/with_linear/pytorch_model.bin'

# # 数据位置
# # pure snli
# # SNIL_TRAIN = './datasets/cnsd-snli/train.txt'
# # STS_DEV = './datasets/cnsd-snli/dev.txt'
# # STS_TEST = './datasets/cnsd-snli/test.txt'

# # pure sts-B
# SNIL_TRAIN = './datasets/STS-B/cnsd-sts-train.txt'
# STS_DEV = './datasets/STS-B/cnsd-sts-dev.txt'
# STS_TEST = './datasets/STS-B/cnsd-sts-test.txt'


# seed = 3402
# # 固定PyTorch随机种子
# torch.manual_seed(seed)
# # 固定CPU随机种子
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# # 固定numpy随机种子
# np.random.seed(seed)
# # 固定随机种子
# random.seed(seed)


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集
    """

    # TODO: 把lqcmc的数据生成正负样本, 拿来做测试
    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]

    def load_dataset_from_json(path):

        # 打开并读取JSON文件
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # 提取并转换数据
        dataset = [(item['origin'], item['entailment'], item['contradiction']) for item in data]

        return dataset

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            # return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]
            return [(line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]) for line in f]

    def load_sc_dev_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [(item['str1'], item['str2'], str(item['similarity'])) for item in data]


    assert name in ["snli", "lqcmc", "sts", "sc_dev", "sc_train"]
    if name == 'sc_train':
        return load_dataset_from_json(path)
    elif name == 'sc_dev':
        return load_sc_dev_data(path)
    elif name == 'lqcmc':
        return load_lqcmc_data(path)
    elif name == 'sts':
        return load_sts_data(path)
    else:  # snli
        return load_snli_data(path)


class TrainDataset(Dataset):
    """
    训练数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, MAXLEN, data: List):
        self.data = data
        self.MAXLEN = MAXLEN

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]], max_length=self.MAXLEN,
                         truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """

    def __init__(self,MAXLEN, data: List):
        self.data = data
        self.MAXLEN = MAXLEN
    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer(text, max_length=self.MAXLEN, truncation=True,
                         padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        line = self.data[index]
        # return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), float(line[2])

# 对有标签样本进行召回精度检测的数据集
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.texts = dataframe['target_text']
        self.labels = dataframe['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs


def eval(model, dataloader) -> float:
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)

            source_token_type_ids = source.get('token_type_ids', None)
            if source_token_type_ids is not None:
                source_token_type_ids = source_token_type_ids.squeeze(1).to(DEVICE)

            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]

            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)

            target_token_type_ids = target.get('token_type_ids', None)
            if target_token_type_ids is not None:
                target_token_type_ids = target_token_type_ids.squeeze(1).to(DEVICE)

            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
            # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(args, model, train_dl, dev_dl, optimizer, teacher=None) -> None:
    """
    模型训练函数
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):

        real_batch_num = source.get('input_ids').shape[0]
        num_sent = source.get('input_ids').size(1)
        # print(num_sent)

        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        input_ids = source.get('input_ids').view(real_batch_num * num_sent, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * num_sent, -1).to(DEVICE)

        # token_type_ids = source.get('token_type_ids').view(real_batch_num * num_sent, -1).to(DEVICE)

        token_type_ids = source.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(1).to(DEVICE)

        # Encode, unflatten, and pass to student
        teacher_top1_sim_pred = None
        teacher_embeddings = None
        if teacher is not None:
            with torch.no_grad():
                # Flatten input for encoding by the teacher - (bsz * num_sent, len)


                teacher_inputs = copy.deepcopy(source)
                if args.teacher_max_length>0:
                    teacher_input_ids = input_ids[:, :args.teacher_max_length]  # 截断 input_ids
                    teacher_attention_mask = attention_mask[:, :args.teacher_max_length]  # 截断 attention_mask

                teacher_inputs["input_ids"] = teacher_input_ids
                teacher_inputs["attention_mask"] = teacher_attention_mask


                if token_type_ids is None:
                    teacher_inputs["token_type_ids"] = torch.zeros_like(teacher_input_ids)
                else:
                    if args.teacher_max_length > 0:
                        teacher_token_type_ids = token_type_ids[:, :args.teacher_max_length]
                    teacher_inputs["token_type_ids"] = teacher_token_type_ids
                # print(teacher_inputs)

                teacher_inputs.to(DEVICE)
                # 将输入数据送到指定的设备
                # teacher_inputs = {k: v.to(DEVICE) for k, v in teacher_inputs.items()}

                # Single teacher
                embeddings = teacher.encode(teacher_inputs)
                embeddings = embeddings.view((real_batch_num, num_sent, -1))
                z1T, z2T = embeddings[:, 0], embeddings[:, 1]

                cos = nn.CosineSimilarity(dim=-1)
                teacher_top1_sim_pred = cos(z1T.unsqueeze(1), z2T.unsqueeze(0)) / args.tau2

                if args.use_teacher_embedding:
                    teacher_embeddings = z1T

                # print(teacher_top1_sim_pred.shape)
                # print(teacher_top1_sim_pred)

        # print(input_ids.shape)
        # simcse训练 change
        loss, _ = model(input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred, teacher_embeddings)
        # loss = simcse_sup_loss(out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        # change
        eval_round = 100

        if batch_idx % eval_round == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                early_stop_batch = 0
                best = corrcoef

                # 使用os.path.dirname获取路径中的目录部分
                save_dir = os.path.dirname(args.save_path)
                # 检查目录是否存在，如果不存在，则创建它
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(model.state_dict(), args.save_path)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue  # change
                # break
            early_stop_batch += 1
            # 存在问题，只是停了一个epoch
            if early_stop_batch == eval_round:
                logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {(batch_idx - eval_round) * args.batch_size}")
                return








from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
def acc_eval1(train_path, val_path, model, tokenizer, device, batch_size, max_len, k=5):
    def ds2df(data_ds):
        data_df = pd.DataFrame(columns=['target_text', 'label'])
        # 遍历 data_ds 中的每一行
        for index, row in data_ds.iterrows():
            # 根据 slither 的值决定 label 的值
            label = 0 if row['slither'] == 4 else 1
            # 将 source_code 和 label 添加到新的 DataFrame 中
            new_row = {'target_text': row['target_text'], 'label': label}
            data_df = data_df.append(new_row, ignore_index=True)
        return data_df

    def extract_embeddings(dataloader, model, device):
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # If token_type_ids are not part of your batch, create a tensor of zeros
                # This assumes that all input_ids have the same length
                token_type_ids = torch.zeros_like(input_ids).to(device)
                # If your batch already includes token_type_ids, uncomment the following line:
                # token_type_ids = batch['token_type_ids'].to(device)
                # Provide token_type_ids to the model
                output = model(input_ids, attention_mask, token_type_ids)
                embeddings.extend(output.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        return np.array(embeddings), np.array(labels)

    def nearest_neighbor_retrieval(train_embeddings, test_embeddings, k):
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(train_embeddings)
        distances, indices = nn.kneighbors(test_embeddings)
        return indices

    def evaluate_retrieval(indices, test_labels):
        correct = 0
        for idx, neighbors in enumerate(indices):
            query_label = test_labels[idx]
            neighbor_labels = train_labels[neighbors[1:]]  # 排除自身
            if query_label in neighbor_labels:
                correct += 1
        accuracy = correct / len(test_labels)

        # y_true = test_labels
        # y_pred = [train_labels[neighbor] for neighbors in indices for neighbor in neighbors[1:]]
        # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        # print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

        return accuracy

    # 读取数据
    train_df = pd.read_json(train_path)
    val_df = pd.read_json(val_path)

    # 创建Dataset
    train_dataset = TextDataset(train_df, tokenizer, max_len)
    test_dataset = TextDataset(val_df, tokenizer, max_len)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 提取嵌入
    train_embeddings, train_labels = extract_embeddings(train_loader, model, device)
    test_embeddings, test_labels = extract_embeddings(test_loader, model, device)

    # 执行最近邻检索
    indices = nearest_neighbor_retrieval(train_embeddings, test_embeddings, k)

    # 计算准确率
    accuracy = evaluate_retrieval(indices, test_labels)
    print(f"Label-based Retrieval Accuracy: {accuracy}")


import time
# with time counter
def acc_eval(train_path, val_path, model, tokenizer, device, batch_size, max_len, k=5):
    def ds2df(data_ds):
        data_df = pd.DataFrame(columns=['target_text', 'label'])
        for index, row in data_ds.iterrows():
            label = 0 if row['slither'] == 4 else 1
            new_row = {'target_text': row['target_text'], 'label': label}
            data_df = data_df.append(new_row, ignore_index=True)
        return data_df

    def extract_embeddings(dataloader, model, device):
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = torch.zeros_like(input_ids).to(device)
                output = model(input_ids, attention_mask, token_type_ids)
                embeddings.extend(output.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        return np.array(embeddings), np.array(labels)

    def nearest_neighbor_retrieval(train_embeddings, test_embeddings, k):
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(train_embeddings)
        start_time = time.time()
        distances, indices = nn.kneighbors(test_embeddings)
        retrieval_time = time.time() - start_time
        return indices, retrieval_time

    def evaluate_retrieval(indices, test_labels):
        correct = 0
        y_pred = []
        y_true = test_labels
        for idx, neighbors in enumerate(indices):
            query_label = test_labels[idx]
            neighbor_labels = train_labels[neighbors[1:]]  # Exclude self
            if query_label in neighbor_labels:
                correct += 1
            y_pred.append(neighbor_labels[0])  # Taking the closest neighbor's label

        accuracy = correct / len(test_labels)
        # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        # print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

        return accuracy

    # Read data
    train_df = pd.read_json(train_path)
    val_df = pd.read_json(val_path)

    # Create Dataset
    train_dataset = TextDataset(train_df, tokenizer, max_len)
    test_dataset = TextDataset(val_df, tokenizer, max_len)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract embeddings
    train_embeddings, train_labels = extract_embeddings(train_loader, model, device)

    start_encoding_time = time.time()
    test_embeddings, test_labels = extract_embeddings(test_loader, model, device)
    encoding_time = time.time() - start_encoding_time

    # Perform nearest neighbor retrieval and measure retrieval time
    indices, retrieval_time = nearest_neighbor_retrieval(train_embeddings, test_embeddings, k)

    # Calculate accuracy and other metrics
    accuracy = evaluate_retrieval(indices, test_labels)
    print(f"Label-based Retrieval Accuracy: {accuracy:.8f}")
    print(f"Encoding Time: {encoding_time:.4f} seconds")
    print(f"Retrieval Time: {retrieval_time:.4f} seconds")
    print(f"Total Time: {encoding_time + retrieval_time:.4f} seconds")
def knn_eval(train_path, val_path, model, tokenizer, device, batch_size, max_len, k=5):
    def ds2df(data_ds):
        data_df = pd.DataFrame(columns=['target_text', 'label'])
        # 遍历 data_ds 中的每一行
        for index, row in data_ds.iterrows():
            # 根据 slither 的值决定 label 的值
            label = 0 if row['slither'] == 4 else 1
            # 将 source_code 和 label 添加到新的 DataFrame 中
            new_row = {'target_text': row['target_text'], 'label': label}
            data_df = data_df.append(new_row, ignore_index=True)
        return data_df

    def extract_embeddings(dataloader, model, device):
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = torch.zeros_like(input_ids).to(device)
                output = model(input_ids, attention_mask, token_type_ids)
                embeddings.extend(output.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        return np.array(embeddings), np.array(labels)

    # 读取数据
    train_df = pd.read_json(train_path)
    val_df = pd.read_json(val_path)

    # 创建Dataset
    train_dataset = TextDataset(train_df, tokenizer, max_len)
    test_dataset = TextDataset(val_df, tokenizer, max_len)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 提取嵌入
    train_embeddings, train_labels = extract_embeddings(train_loader, model, device)
    test_embeddings, test_labels = extract_embeddings(test_loader, model, device)

    # 使用 KNN 进行分类
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(test_embeddings)

    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')
    print(f"KNN-based Retrieval Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    conf_matrix = confusion_matrix(test_labels, predictions)
    print(conf_matrix)
    # print(f"KNN-based Retrieval Accuracy: {accuracy}")

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 添加打印P\R\F1
def dou_ann_eval(train_path, test_path, model, tokenizer, device, max_len, batch_size=32, k=5):
    # 读取训练数据
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    train_texts = [d['target_text'] for d in train_data]
    train_labels = [d['label'] for d in train_data]

    # 读取测试数据
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # 将训练数据集编码并存储句向量
    train_encodings = tokenizer(train_texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    train_encodings = {k: v.to(device) for k, v in train_encodings.items()}
    with torch.no_grad():
        train_embeddings = model.encoder(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_encodings.get('token_type_ids', None)
        )
        train_embeddings = train_embeddings.view(len(train_texts), -1)

    # 对测试数据进行评估
    predictions, true_labels = [], []
    test_iter = tqdm(test_data, desc='Evaluating', total=len(test_data))
    for test_sample in test_iter:
        test_text = test_sample['target_text']
        true_label = test_sample['label']
        true_labels.append(true_label)

        # 将测试样本编码为模型输入并移动到设备上
        test_encoding = tokenizer([test_text], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        test_encoding = {k: v.to(device) for k, v in test_encoding.items()}
        with torch.no_grad():
            test_embedding = model.encoder(
                test_encoding['input_ids'],
                test_encoding['attention_mask'],
                test_encoding.get('token_type_ids', None)
            )

            # 计算相似度并获取top-k最相似样本
            similarities = model.sim_cal(test_embedding, train_embeddings)
            topk_indices = torch.topk(similarities, k=k).indices
            topk_labels = [train_labels[i] for i in topk_indices]

        # 将预测结果视为最频繁的标签
        predictions.append(max(set(topk_labels), key=topk_labels.count))

    # 计算评价指标
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return precision, recall, f1

def dou_ann_eval3(train_path, test_path, model, tokenizer, device, max_len, batch_size=32, k=5):
    # 读取训练数据
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    train_texts = [d['target_text'] for d in train_data]
    train_labels = [d['label'] for d in train_data]

    # 读取测试数据
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # 将训练数据集编码并存储句向量
    train_encodings = tokenizer(train_texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    train_encodings = {k: v.to(device) for k, v in train_encodings.items()}
    with torch.no_grad():
        train_embeddings = model.encoder(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_encodings.get('token_type_ids', None)
        )
        train_embeddings = train_embeddings.view(len(train_texts), -1)

    # 对测试数据进行评估
    num_correct = 0
    test_iter = tqdm(test_data, desc='Evaluating', total=len(test_data))
    for test_sample in test_iter:
        test_text = test_sample['target_text']
        true_label = test_sample['label']

        # 将测试样本编码为模型输入并移动到设备上
        test_encoding = tokenizer([test_text], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        test_encoding = {k: v.to(device) for k, v in test_encoding.items()}
        with torch.no_grad():
            test_embedding = model.encoder(
                test_encoding['input_ids'],
                test_encoding['attention_mask'],
                test_encoding.get('token_type_ids', None)
            )

            # 计算相似度并获取top-k最相似样本
            similarities = model.sim_cal(test_embedding, train_embeddings)
            # print(similarities.shape)
            topk_indices = torch.topk(similarities, k=k).indices
            # print(topk_indices)
            topk_labels = [train_labels[i] for i in topk_indices]

        # 检查true_label是否在topk_labels中
        if true_label in topk_labels:
            num_correct += 1

    accuracy = num_correct / len(test_data)
    print(accuracy)

    return accuracy

def dou_ann_eval1(train_path, test_path, model, tokenizer, device, max_len, batch_size=1, k=5):
    # 读取训练数据
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    train_texts = [d['target_text'] for d in train_data]
    train_labels = [d['label'] for d in train_data]

    # 读取测试数据
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    num_correct = 0

    # 将训练数据集分批并移动到设备上
    train_encodings = []
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i:i+batch_size]
        batch_encodings = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        batch_encodings = {k: v for k, v in batch_encodings.items()}
        train_encodings.append(batch_encodings)

    # 对每个测试样本进行评估
    test_iter = tqdm(test_data, desc='Evaluating', total=len(test_data))
    for test_sample in test_iter:
        test_text = test_sample['target_text']
        true_label = test_sample['label']

        # 将测试样本编码为模型输入并移动到设备上
        test_encoding = tokenizer(test_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        test_encoding = {k: v.to(device) for k, v in test_encoding.items()}

        # 计算相似度并获取top-k最相似样本
        similarities = []
        for batch_encodings in train_encodings:
            model.eval()
            with torch.no_grad():
                batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}
                batch_similarities = model(
                    src_token_ids=test_encoding['input_ids'],
                    src_mask=test_encoding['attention_mask'],
                    tgt_token_ids=batch_encodings['input_ids'],
                    tgt_mask=batch_encodings['attention_mask'],
                )[:, 1]  # 取相似度分数(第二个值)
                similarities.extend(batch_similarities.cpu().tolist())

        similarities = torch.tensor(similarities)
        topk_indices = torch.topk(similarities, k=k).indices
        topk_labels = [train_labels[i] for i in topk_indices]

        # 检查true_label是否在topk_labels中
        if true_label in topk_labels:
            num_correct += 1

    accuracy = num_correct / len(test_data)
    print(accuracy)
    return accuracy


def get_weight_decay_params(model, cv=False):
    decay_params = set()
    no_decay_params = set()


    if cv:
        # 正常情况下的处理
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
        blacklist_weight_modules = (nn.BatchNorm2d, nn.GroupNorm)

        for module_name, module in model.named_modules():
            for param_name, _ in module.named_parameters():
                fpn = f'{module_name}.{param_name}' if module_name else param_name  # full param name

                if 'bias' in param_name:
                    no_decay_params.add(fpn)
                elif 'weight' in param_name and isinstance(module, whitelist_weight_modules):
                    decay_params.add(fpn)
                elif 'weight' in param_name and isinstance(module, blacklist_weight_modules):
                    no_decay_params.add(fpn)
    # 如果使用预训练语言模型，对权重衰减策略做特殊处理
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        for name, param in model.named_parameters():
            if not any(nd in name for nd in no_decay):
                decay_params.add(name)
            else:
                no_decay_params.add(name)

        # return decay_params, no_decay_params

    # 验证所有参数都被正确分类
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params
    union_params = decay_params | no_decay_params
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
    str(param_dict.keys() - union_params),)

    # 将参数名称转换为参数对象
    decay = [param_dict[pn] for pn in sorted(list(decay_params))]
    no_decay = [param_dict[pn] for pn in sorted(list(no_decay_params))]

    return decay, no_decay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the SimCSE model with command line options.')


    # Training configuration
    parser.add_argument('--do_train', default=False, action='store_true', help='Whether to train the model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--maxlen', type=int, default=64, help='Maximum sequence length')
    # teacher_max_length
    parser.add_argument('--teacher_max_length', type=int, default=-1, help='Maximum sequence length of teacher')

    parser.add_argument('--pooling', type=str, default='last-avg',
                        choices=['cls', 'pooler', 'last-avg', 'first-last-avg'],
                        help='Pooling strategy used in the model')
    parser.add_argument('--cv', default=False, action='store_true', help='Whether to use cv as the encoder')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay rate')

    # Model and Teacher Model Paths
    parser.add_argument('--model_path', type=str, default='E:\\model\\transformers4\\chinse_simcse_roberta',
                        help='Path to the pretrained model directory')
    parser.add_argument('--use_teacher', default=False, action='store_true', help='Whether to use a teacher model for training')
    parser.add_argument('--use_teacher_embedding', action='store_true',
                        help='Whether to use embeddings from a teacher model')
    parser.add_argument('--teacher_model', type=str, default='E:\\model\\transformers4\\chinse_simcse_roberta',
                        help='Path to the teacher model directory')
    parser.add_argument('--teacher_pooler', type=str, default='cls_before_pooler',
                        help='Pooling method for the teacher model')
    parser.add_argument('--tau2', type=float, default=0.05,
                        help='Tau2 parameter for distillation from the teacher model')
    parser.add_argument('--continuous_training', default=False, action='store_true', help='Whether to continue training')

    # Saving and Data Paths
    parser.add_argument('--save_path', type=str, default='./saved_model/with_linear/pytorch_model.bin',
                        help='Path to save the trained model')
    parser.add_argument('--snli_train', type=str, default='./datasets/STS-B/cnsd-sts-train.txt',
                        help='Path to the training data file')
    parser.add_argument('--sts_dev', type=str, default='./datasets/STS-B/cnsd-sts-dev.txt',
                        help='Path to the development data file')
    parser.add_argument('--sts_test', type=str, default='./datasets/STS-B/cnsd-sts-test.txt',
                        help='Path to the test data file')

    # Accuracy Evaluation Parameters
    parser.add_argument('--acc_train', type=str, default='./datasets/zhihu/train.csv',
                        help='Path to the additional training data for accuracy evaluation')
    parser.add_argument('--acc_val', type=str, default='./datasets/zhihu/train.csv',
                        help='Path to the additional validation data for accuracy evaluation')
    parser.add_argument('--acc_k', type=int, default=5, help='The top-k accuracy for retrieval tasks')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=3402,
                        help='Seed for random number generation to ensure reproducibility')


    args = parser.parse_args()

    # Setting the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialization and model preparation steps here
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)


    train_data = load_data('sc_train', args.snli_train)
    random.shuffle(train_data)
    dev_data = load_data('sc_dev', args.sts_dev)
    test_data = load_data('sc_dev', args.sts_test)

    train_dataloader = DataLoader(TrainDataset(args.maxlen, train_data), batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(TestDataset(args.maxlen, dev_data), batch_size=args.batch_size)
    test_dataloader = DataLoader(TestDataset(args.maxlen, test_data), batch_size=args.batch_size)

    teacher_isavailable = False
    teacher = None
    if args.use_teacher:
        teacher = Teacher(model_name_or_path=args.teacher_model, pooler=args.teacher_pooler, device=DEVICE)
        teacher_isavailable = True
    model = SimcseModel(pretrained_model=args.model_path, pooling=args.pooling, teacher_isavailable=teacher_isavailable)


    model.to(DEVICE)



    if args.do_train:

        if args.use_teacher:
            # 获取需要更新的参数
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'embedding2' in name or 'layers' in name:
                        no_decay.append(param)
                    else:
                        decay.append(param)


        else:
            decay, no_decay = get_weight_decay_params(model, args.cv)

        optim_groups = [
            {'params': decay, 'weight_decay': args.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        best = 0

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=args.lr)

        if args.continuous_training:
            model.load_state_dict(torch.load(args.save_path))
        summary(model)

        for epoch in range(args.epochs):
            train(args, model, train_dataloader, dev_dataloader, optimizer, teacher=teacher)
        torch.save(model.state_dict(), args.save_path)

    # Load the best model for evaluation
    model.load_state_dict(torch.load(args.save_path))
    dev_corrcoef = eval(model, dev_dataloader)
    test_corrcoef = eval(model, test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')

    acc_eval(args.acc_train, args.acc_val, model, tokenizer, DEVICE, args.batch_size, args.maxlen,args.acc_k)
    # acc_eval(args.acc_train, args.acc_val, model, tokenizer, DEVICE, args.batch_size, args.maxlen,20)
    knn_eval(args.acc_train, args.acc_val, model, tokenizer, DEVICE, args.batch_size, args.maxlen,5)




