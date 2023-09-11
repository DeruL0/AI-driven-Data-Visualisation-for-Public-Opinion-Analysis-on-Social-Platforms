import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import numpy as np
import torch.nn.functional as F
import jieba
import re
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
from gensim.models import Word2Vec
import jieba.posseg as psg
from tqdm import tqdm
# torch.manual_seed(1)
class Config(object):
    data_path = '/dataset/origin_data.csv'        # 训练集csv文本文件存放路径
    processed_data = "/models/proccess_embed32_epoch30.pkl"    # 预处理好的二进制文件,分词后的结果
    lr = 1e-3
    weight_decay = 1e-4
    epoch = 200
    
    batch_size = 516
    maxlen = 150                # 超过这个长度之后的字被丢弃，小于这个长度的在后面补零
    model_path = "/models/transformer_train.model"          # transformer预训练模型路径
    word2Vec_path = "/models/word2vec_embed32_epoch30.model"          # word2Vec模型保存路径
    save_epoch = 1 #一轮保存一次
    epoch_word2vec=7
    embedding_dim = 32
    min_count = 5
    window_size = 3 
opt = Config()

# 读取保存的文件，返回word2idx word2vec,training_data
def read_processed_data():
    file = open(opt.processed_data,'rb')
    processed_data = pickle.load(file)
    return processed_data['word2idx'],processed_data['word2Vec'],processed_data['x_train'],processed_data['x_test'],processed_data['y_train'],processed_data['y_test'], processed_data['verify_train'],processed_data['verify_test'],processed_data['embedding_dim']

#将输入的句子的word转化为idx，并进行截断和补-1（因为0已经表示了其他词）,不在字典中的字用-2表示
def prepare_seq(sentence_list, to_ix, device,max_len):
    seq_list = []
    for sentence in sentence_list:
        seq = []
        for word in sentence:
          if word in to_ix:
            seq.append(to_ix[word] if isinstance(word,str) else to_ix[word[0]])
          else:
            seq.append(-2)
        len_seq = len(seq)
        if len_seq > max_len:
            seq = np.array(seq[:max_len])
        else:
            seq = np.concatenate( (np.array(seq),np.array( [-1]*(max_len-len_seq) ) ) )
        seq_list.append(seq)
    seq_list = np.array(seq_list)
    return torch.tensor(seq_list,dtype=torch.long,device=device)
    

def get_key_padding_mask(tokens,device):
    key_padding_mask = torch.zeros(tokens.size(),device=device)
    key_padding_mask[tokens == -1] = True
    return key_padding_mask

# 重载的数据集
class MyDataset(Dataset):
    def __init__(self,X,y) -> None:
        super().__init__()
        self.X=X
        self.y=y
        self.len=len(self.X)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
class Dataset_for_predict(Dataset):
    def __init__(self,X) -> None:
        super().__init__()
        self.X=X
        self.len = len(self.X)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index]

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
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
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])





# transformer模型
class Transformer(nn.Module):
    def __init__(self, max_len, emb_size, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.FC1 = nn.Linear(max_len*emb_size, 4096)
        self.FC2 = nn.Linear(4096, 1024)
        self.FC3 = nn.Linear(1024, 2)
        
        self.dropoutLayer=nn.Dropout()
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
        memory = memory.reshape(memory.shape[0],-1)
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


    
# 计算score,注意这里直接用了测试集的数据，因为没有划分验证集(doge)
@torch.no_grad()
def compute_test(model, test_dataloader,word2Vec,device):
    model.eval()
    for _,(sentence, y_true) in enumerate(test_dataloader):
        key_padding_mask = get_key_padding_mask(sentence,device)
        sentence = word2Vec[sentence]
        y_pred = model(sentence,key_padding_mask).argmax(1)
        # y_pred = np.array(y_pred.cpu())
        # y_true = np.array(y_true.cpu())
        # # 调用的是sklearn的api，先转化为np.array
        # f1 = f1_score(y_true,y_pred)
        # accuracy = accuracy_score(y_true,y_pred)
        accuracy = (y_pred==y_true).sum().item()/len(y_pred)
        print(
            # "f1_score= {:.4f}".format(f1),
            "accuracy= {:.4f}".format(accuracy))
        return
# 训练所有的训练集，不进行验证
def train_all():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if os.path.exists(opt.word2Vec_path) and os.path.exists(opt.processed_data):
        with open(opt.processed_data,'rb') as f:
            processed_data = pickle.load(f)
            X = processed_data['X']
            Y = processed_data['Y']
            opt.embedding_dim = processed_data['embedding_dim']
        word2Vec_model = Word2Vec.load(opt.word2Vec_path)
        word2Vec = torch.tensor(word2Vec_model.wv.vectors,device=device)
        pad_vec = torch.tensor([[0.0]*opt.embedding_dim],dtype=torch.float32,device=device)
        unknown_vec = torch.tensor([[-1.0]*opt.embedding_dim],dtype=torch.float32,device=device)
        torch.cat((word2Vec,unknown_vec,pad_vec))
        X = prepare_seq(X,word2Vec_model.wv.key_to_index,device,opt.maxlen)
    else:
        X,Y,word2Vec = process_raw_data(opt.data_path,opt.word2Vec_path,True)
    X = X.to(device)
    Y = torch.tensor(Y,device=device)
    # 训练集和测试集
    dataset = MyDataset(X,Y)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,drop_last=True)

    model = Transformer(opt.maxlen,emb_size=opt.embedding_dim)
    if os.path.exists(opt.model_path):
        print("load from pretrained model")
        model.load_state_dict(torch.load(opt.model_path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # 二分类
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    training_dic={
        'epoch':0
    }
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(opt.epoch):
        training_dic['epoch']= epoch + 1
        for _,(sentence, tags) in tqdm(enumerate(dataloader),total=len(dataloader),postfix=training_dic):

            # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
            model.train()
            model.zero_grad()

            # Step 2.获得padding_mask,将序列转化为词向量
            key_padding_mask = get_key_padding_mask(sentence,device)
            sentence = word2Vec[sentence]

            # Step 3. Run our forward pass.
            pred_tag = model(sentence,key_padding_mask)
            loss = criterion(pred_tag,tags)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss.backward()
            optimizer.step()
            
        accuracy = (pred_tag.argmax(1)==tags).sum().item()/len(tags)
        print(f"epoch={epoch+1}, loss={loss.item():.4f}, acc={accuracy:.4f} ")

        if epoch%opt.save_epoch == 0 and epoch :
            #保存模型
            torch.save(model.state_dict(), opt.model_path)
    

def process_raw_data(file_path: str, word2Vec_path: str, train: bool ):
    """
    处理原始.csv数据集,返回构建的word2Vec的模型,如果有预训练的word2Vec,则加载;否则,训练后保存一个

    Parameters:
        file_path - 需要处理的.csv数据集，有'content'与'verify_typ'两列,
        pretrained_word2Vec - word2Vec预训练模型
    
    Returns:
        X: torch.tensor 每一行都是词向量数组的数组
    
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.exists(word2Vec_path) and not train:
        raise Exception(f"word2Vec_path {opt.word2Vec_path} is not exists")
    df = pd.read_csv(file_path, encoding='UTF-8',low_memory=False)
    # 获取某一列的数据
    # column_data = df['content']
    if train:
        Y = torch.tensor(df['sentiment'].to_numpy(),device=device,dtype=torch.float32)
    dic = {
        '没有认证':'〇',
        '蓝V认证':'壹',
        '黄V认证':'贰',
        '红V认证':'叁',
    }
    verify_type = np.array([dic[verify] if verify in dic else '〇' for verify in df['verify_typ']])
    # 分词
    dic_file = "/lda/stop_dic/dict.txt"
    stop_file = "/lda/stop_dic/stopwords.txt"
    mytext = df['content']
    # 剪切中文字符
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    lines = []
    for verify,sentence in zip(verify_type,mytext):
        word_list = []
        word_list.append(verify)
        # jieba分词
        seg_list = psg.cut(sentence)
        for seg_word in seg_list:
            word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
            find = 0
            for stop_word in stop_list:
                if stop_word == word or len(word) < 2:  # this word is stopword
                    find = 1
                    break
            if find == 0 and seg_word.flag in flag_list:
                word_list.append(word)
        lines.append(word_list)
    dfx = pd.DataFrame({'content_cutted': lines})
    # 将DataFrame输出到CSV文件中
    output_path = '/word_cut/word_cutted.csv'
    dfx.to_csv(output_path, index=False, encoding="utf-8")
    if not os.path.exists(opt.word2Vec_path) and train:
        print("not find word2vec model")
        model = Word2Vec(lines,vector_size = opt.embedding_dim, window = opt.window_size , min_count = opt.min_count, epochs=opt.epoch_word2vec, negative=10,sg=1)
    else:
        model = Word2Vec.load(word2Vec_path)
        model.train(lines,total_examples=len(lines),epochs=model.epochs)
    
    word2Vec = torch.tensor(model.wv.vectors,device=device)
    pad_vec = torch.tensor([[0.0]*opt.embedding_dim],dtype=torch.float32,device=device)
    unknown_vec = torch.tensor([[-1.0]*opt.embedding_dim],dtype=torch.float32,device=device)
    torch.cat((word2Vec,unknown_vec,pad_vec))
    X = prepare_seq(lines,model.wv.key_to_index,device,opt.maxlen)
    if train:
        return X,Y,word2Vec
    else:
        return X,word2Vec

@torch.no_grad()
def sentiment_predict(file_path:str, word2Vec_path=opt.word2Vec_path, model_path=opt.model_path,output_path='result.txt'):
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
    X,word2Vec = process_raw_data(file_path,word2Vec_path,False)
    model = Transformer(opt.maxlen,emb_size=opt.embedding_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)
    dataset = Dataset_for_predict(X)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=False,drop_last=False)
    y_pred = []
    num_zero = 0
    num_one = 0
    print("开始预测：")
    for i,sentence in enumerate(dataloader):
        model.eval()

        # Step 2.获得padding_mask,将序列转化为词向量
        key_padding_mask = get_key_padding_mask(sentence,device)
        sentence = word2Vec[sentence]

        # Step 3. Run our forward pass.
        logits = model(sentence,key_padding_mask)
        tag = logits.argmax(1)
        num_zero += (tag==0).sum().item()
        num_one += (tag==1).sum().item()
        for each in tag:
            y_pred.append(each.item())
    print(y_pred)
    print(f'总个数：{len(y_pred)}\n预测为0的个数：{num_zero}\n预测为1的个数：{num_one}')
    with open('result.txt', 'w') as f:
        i = 0
        for text in y_pred:
            f.write(f"{i}\t{text}\n")
            i = i+1
    print(f"结果保存至 {output_path}")
if __name__=="__main__":
    # sentiment_predict('Q2_test_dataset.csv','lba_word2vec.model','transformer_cls.model')
    train_all()