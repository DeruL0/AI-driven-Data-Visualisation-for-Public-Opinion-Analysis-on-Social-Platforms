import pickle
from gensim.models import Word2Vec
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


embedding_dim = 32
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
maxlen = 150

import torch
import pickle
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=516):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2).float() * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).float().reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# transformer模型
class Transformer(nn.Module):
    def __init__(self, max_len, emb_size, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=dim_feedforward,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.FC1 = nn.Linear(max_len * emb_size, 4096)
        self.FC2 = nn.Linear(4096, 1024)
        self.FC3 = nn.Linear(1024, 2)

        self.dropoutLayer = nn.Dropout()
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask):
        src = self.positional_encoding(src)
        # print(src_key_padding_mask.shape)
        # print(src.shape)
        memory = self.transformer_encoder(src)
        memory = memory.reshape(memory.shape[0], -1)
        out = self.FC1(memory)
        # out = self.dropoutLayer(out)
        out = self.activation(out)
        out = self.FC2(out)
        # out = self.dropoutLayer(out)
        out = self.activation(out)
        out = self.FC3(out)
        # out = self.dropoutLayer(out)
        logit = self.activation(out)
        # logit: torch.Size([128, 125, 2])
        return logit


# 将输入的句子的word转化为idx，并进行截断和补-1（因为0已经表示了其他词）,不在字典中的字用-2表示
def prepare_seq(sentence_list, to_ix, device, max_len):
    seq_list = []
    for sentence in sentence_list:
        seq = []
        for word in sentence:
            if word in to_ix:
                seq.append(to_ix[word] if isinstance(word, str) else to_ix[word[0]])
            else:
                seq.append(-2)
        len_seq = len(seq)
        if len_seq > max_len:
            seq = np.array(seq[:max_len])
        else:
            seq = np.concatenate((np.array(seq), np.array([-1] * (max_len - len_seq))))
        seq_list.append(seq)
    seq_list = np.array(seq_list)
    return torch.tensor(seq_list, dtype=torch.long, device=device)


def get_key_padding_mask(tokens, device):
    key_padding_mask = torch.zeros(tokens.size(), device=device)
    key_padding_mask[tokens == -1] = True
    return key_padding_mask


# 重载的数据集
class MyDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def process_raw_data():
    """
    处理原始.csv数据集,返回构建的word2Vec的模型,如果有预训练的word2Vec,则加载;否则,训练后保存一个

    Parameters:
        file_path - 需要处理的.csv数据集，有'content'与'verify_typ'两列,
        pretrained_word2Vec - word2Vec预训练模型

    Returns:
        X: torch.tensor 每一行都是词向量数组的数组

    """

    model = Word2Vec.load('e:/cpp/pythonfiles/AI-ML/game_code/code/models/word2vec_embed32_epoch30.model')

    word2Vec = torch.tensor(model.wv.vectors, device=device)
    pad_vec = torch.tensor([[0.0] * embedding_dim], dtype=torch.float32, device=device)
    unknown_vec = torch.tensor([[-1.0] * embedding_dim], dtype=torch.float32, device=device)
    torch.cat((word2Vec, unknown_vec, pad_vec))
    file = open('e:/cpp/pythonfiles/AI-ML/game_code/code/models/proccess_embed32_epoch30.pkl', 'rb')
    processed_data = pickle.load(file)
    X = processed_data['X']  # 分词之后的数据
    Y = processed_data['Y']
    X = prepare_seq(X, model.wv.key_to_index, device, maxlen)

    return X, Y, word2Vec


@torch.no_grad()
def sentiment_predict(output_path='result_train.txt'):
    r"""
    加载词向量模型、transformer模型，读取file_path，进行情感分类，将结果保存至output.csv

    Args:
        file_path : 需要处理的.csv数据集，有'content'与'verify_typ'两列
        word2Vec_path : word2Vec模型保存路径
        model_path : transformer模型保存路径
        output_path : 输出的文件保存路径
    """
    print("分词+转化为词向量：")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    X, Y, word2Vec = process_raw_data()
    model = Transformer(maxlen, embedding_dim)
    dataset = MyDataset(X, Y)
    test_dataloader = DataLoader(dataset, batch_size=516, shuffle=False, drop_last=False)
    model.load_state_dict(
        torch.load('e:/cpp/pythonfiles/AI-ML/game_code/code/models/transformer_train.model',
                   map_location=device))
    model.to(device)
    y_pred = []
    num_zero = 0
    num_one = 0
    print("开始预测：")
    for i, (sentence, tag) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        model.eval()

        # Step 2.获得padding_mask,将序列转化为词向量
        key_padding_mask = get_key_padding_mask(sentence, device)
        sentence = word2Vec[sentence]

        # Step 3. Run our forward pass.
        logits = model(sentence, key_padding_mask)
        tag = logits.argmax(1)
        num_zero += (tag == 0).sum().item()
        num_one += (tag == 1).sum().item()
        y_pred.extend(tag.cpu().tolist())
    y_pred = np.array(y_pred)
    y_true = Y
    # 调用的是sklearn的api，先转化为np.array
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(y_pred)
    print('accuary', accuracy)
    print('f1_score', f1)
    print(f'总个数：{len(y_pred)}\n预测为0的个数：{num_zero}\n预测为1的个数：{num_one}')
    with open('result_train.txt', 'w') as f:
        i = 0
        for text in y_pred:
            f.write(f"{i}\t{text}\n")
            i = i + 1
    print(f"结果保存至 {output_path}")


sentiment_predict()