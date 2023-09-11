
import jieba
import os
import re
import pandas as pd
import jieba.posseg as psg
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import multiprocessing
import numpy as np
class Config(object):
    epoch_word2vec = 30
    embedding_dim = 32
    min_count = 5
    window_size = 3
opt = Config()

df = pd.read_csv('dataset\origin_data.csv')
# 获取某一列的数据
column_data = df['content']
column_data.head()
Y_sentiment = df['sentiment'].to_numpy()
dic = {
    '没有认证':'〇',
    '蓝V认证':'壹',
    '黄V认证':'贰',
    '红V认证':'叁',
}
verify_type = np.array([dic[verify] for verify in df['verify_typ']])
# 分词
lines = []
for line,verify in zip(column_data,verify_type): #分别对每段分词
    temp = jieba.lcut(line)  #结巴分词 精确模式
    words = []
    words.append(verify)
    for i in temp:
        #过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*收起)d【】（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    lines.append(words)

Y_sentiment = df['sentiment'].to_numpy()
processed_data ={
    'X':lines,
    'Y':Y_sentiment,
    'embedding_dim':opt.embedding_dim,#词向量的维度
}
with open("/models/proccess_embed32_epoch30.pkl", "wb") as f:
    pickle.dump(processed_data, f)  
print("word2vec start training")
model = Word2Vec(lines,vector_size = opt.embedding_dim, window = opt.window_size , min_count = opt.min_count, epochs=opt.epoch_word2vec,workers=multiprocessing.cpu_count(), negative=10,sg=1)
model.save('/models/word2vec_embed32_epoch30.model')