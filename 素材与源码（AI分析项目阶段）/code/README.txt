使用工具：
Anaconda Navigator 1.9.12
Jupyter notebook 6.0.3
pycharm 2020.1 x64
python 3.8.3

tools version:
torch	2.0.0+cpu
pandas 1.5.3
jieba	0.42.1
re	2.2.1
gensim 4.3.1
sklearn 1.2.2
numpy 1.23.5
matplotlib 3.2.2
BokehJS 2.1.1
tqdm 4.47.0

数据集源文件:
origin_data.csv
train_data.csv
test_data.csv

代码文件：（执行顺序）
gen_word2vec.py 
对数据集进行分词处理+转化为词向量，最终生成预处理过的分词pkl文件和word2vec向量化分词的model文件
↓
transformer_model_gen.py
对处理过后的数据文件读取，读入word2vec的model文件，迭代生成transformer的情感分类预测模型
transformer_train.model保存在models文件夹
↓
sentiment_pred.py(对训练集进行预测的效果）
情感分类主要代码文件，通过前面代码生成的model和pkl文件进行情感分类预测，并把结果输出到result_train.txt中
↓
pred_test.py
transformer预测模型，对测试集进行预测，并将最终的结果输出到result_test.txt中
↓
LDA_topic.ipynb
对数据集源文件进行处理，通过jieba分词精确分词，生成原数据集被分类后的n个话题类型输出到
名为data_topic.csv文件中，
并且输出html文件对分类话题结果进行可视化的处理。
↓
sentiment_time.ipynb
情感分类的时序分析，通过对数据集的分析得到情感的走向和趋势
↓
hotness1.ipynb
热度分析和热度走向预测，生成影响力曲线与热度曲线
