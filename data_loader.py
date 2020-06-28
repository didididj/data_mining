# -*- coding: utf-8 -*-
import os
import math
import torch
import random
import json
from collections import Counter
import jieba

PAD = '<pad>'  # 0
UNK = '<unk>'  # 1
BOS = '<s>'   # 2
EOS = '</s>'  # 3
# 输入： <s> I eat sth .
# 输出： I eat sth  </s>

# encoding=utf-8
# import jieba

# strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
# for str in strs:
#     seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
#     print("Paddle Mode: " + '/'.join(list(seg_list)))

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))


def read_lines(path):
    """
    {"label": "102",
    "label_desc": "news_entertainment",
    "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物",
    "keywords": "江疏影,美少女,经纪人,甜甜圈"}
    """
    with open(path, 'r',encoding = 'utf-8') as f:
        for line in f:

            yield eval(line)
    f.close()


class Vocab(object):
    def __init__(self, specials=[PAD, UNK, BOS, EOS], config=None,  **kwargs):
        self.specials = specials  
        self.counter = Counter()  
        self.stoi = {}
        self.itos = {}            
        self.weights = None
        self.min_freq = config.min_freq  #出现的频数下限

    def make_vocab(self, dataset):   
        for x in dataset:
            if x != [""]:
                self.counter.update(x)      
        if self.min_freq > 1:
            self.counter = {w: i for w, i in filter(lambda x: x[1] >= self.min_freq, self.counter.items())}
        self.vocab_size = 0                   
        for w in self.specials:               
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        for w in self.counter.keys():        
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1
        

        self.itos = {i: w for w, i in self.stoi.items()}
        
        
    def __len__(self):   
        return self.vocab_size


class DataSet(list):     
    def __init__(self, *args, config=None, is_train=True, dataset="train"):
        self.config = config       
        self.is_train = is_train   
        self.dataset = dataset     
        self.data_path = os.path.join(self.config.data_path, dataset + ".json")
        #super()函数是用于调用父类（超类）的一个方法，继承相应函数及属性
        super(DataSet, self).__init__(*args)

    def read(self):   
        for items in read_lines(self.data_path):
            #以句子与标签形成对应的二元组
            sent_pre = tuple(jieba.cut(items["sentence"], cut_all=False))  #将句子利用jieba库函数按词就行划分
            key_pre = list(jieba.cut(items["keywords"]))     #将关键词也进行分词，有的关键词过长，为了更容易找到关联，故分词处理
            j = 0                                        #删去其中的','
            for i in range(len(key_pre)):
                if key_pre[j] == ',':
                    key_pre.pop(j)
                else:
                    j += 1
            key_pre = tuple(key_pre)
            sent = sent_pre + key_pre
            label = items["label_desc"]         #获取相应的标签
            example = [sent, label]             #形成[按字划分的句子，label]的组合，每一句对应一个组合？  
            self.append(example)

    def _numericalize(self, words, stoi):
        return [1 if x not in stoi else stoi[x] for x in words]

    def numericalize(self, w2id, c2id):   #将汉字转换为对应的数字表示
        for i, example in enumerate(self):
            sent, label = example
            sent = self._numericalize(sent, w2id)    #将句子表示为矢量形式，每个字对应的下标代替该字来形成的向量
            label = c2id[label]                      #类别标记由desc转换为数字形式的类别名
            self[i] = (sent, label)


class DataBatchIterator(object):        
    def __init__(self, config, dataset="train",
                 is_train=True,         
                 batch_size=32,         #默认batch_size为32
                 shuffle=False,         
                 batch_first=False,     
                 sort_in_batch=True):
        self.config = config            
        self.examples = DataSet(
            config=config, is_train=is_train, dataset=dataset)
        self.vocab = Vocab(config=config)   #词表
        self.cls_vocab = Vocab(specials=[], config=config)  #类别专用词表
        self.is_train = is_train
        self.max_seq_len = config.max_seq_len   
        self.sort_in_batch = sort_in_batch
        self.is_shuffle = shuffle
        self.batch_first = batch_first  # [batch_size x seq_len x hidden_size]
        self.batch_size = batch_size
        self.num_batches = 0
        self.device = config.device

    def set_vocab(self, vocab):
        self.vocab = vocab

    def load(self, vocab_cache=None):  
        self.examples.read()     #将句子及类别描述简单地提取出来存储

        if not vocab_cache and self.is_train:
            # 0: 分过词的句子， 1: 关键词， 2: 标记
            self.vocab.make_vocab([x[0] for x in self.examples])  
            self.cls_vocab.make_vocab([[x[1]] for x in self.examples]) 
            if not os.path.exists(self.config.save_vocab):
                torch.save(self.vocab, self.config.save_vocab + ".txt")
                torch.save(self.cls_vocab, self.config.save_vocab + ".cls.txt")
        else:  
            self.vocab = torch.load(self.config.save_vocab + ".txt")
            self.cls_vocab = torch.load(self.config.save_vocab + ".cls.txt")
        assert len(self.vocab) > 0 #断言，当条件表达式为false的时候触发异常
        self.examples.numericalize(   #将汉字数字矢量化，根据词表用索引下标替代该词
            w2id=self.vocab.stoi, c2id=self.cls_vocab.stoi)
        
        self.num_batches = math.ceil(len(self.examples)/self.batch_size)

    def _pad(self, sentence, max_L, w2id, add_bos=False, add_eos=False):  
        if add_bos:    
            sentence = [w2id[BOS]] + sentence
        if add_eos:
            sentence = sentence + [w2id[EOS]]
        if len(sentence) < max_L:  
            sentence = sentence + [w2id[PAD]] * (max_L-len(sentence))
        return [x for x in sentence]

    def pad_seq_pair(self, samples):   
        pairs = [pair for pair in samples]  
 
        Ls = [len(pair[0])+2 for pair in pairs]  #将句子部分取出，并且+2=>为BOS与EOS预留两个位置

        max_Ls = max(Ls)    
        sent = [self._pad(  #整理为相同长度
            item[0], max_Ls, self.vocab.stoi, add_bos=True, add_eos=True) for item in pairs]
        label = [item[1] for item in pairs]  #取出类别描述
        batch = Batch()     #以Batch的形式存储
        #为batch对象赋值
        batch.sent = torch.LongTensor(sent).to(device=self.device)  #将句子转换为整型张量，并且搬至相应设备上
        batch.label = torch.LongTensor(label).to(device=self.device)
        
        if not self.batch_first:  
            batch.sent = batch.sent.transpose(1, 0).contiguous()  
        batch.mask = batch.sent.data.clone().ne(0).long().to(device=self.device) #将sent部分克隆一份并将非0位置1，0位保持为0
        #[ne(0) => 不等于0]
        return batch

    def __iter__(self):   
        if self.is_shuffle:    #是否将元素变为乱序，在本场景下没有必要
            random.shuffle(self.examples)
        total_num = len(self.examples)  #总共存有的多少句子，样本
        for i in range(self.num_batches):  #将样本分成num_batches份batch_size数量的样本
            samples = self.examples[i * self.batch_size:
                                    min(total_num, self.batch_size*(i+1))]
            if self.sort_in_batch:    #按降序排列，主要是将长度相近的作为同一batch中，避免补充过多的<pad>
                samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
            yield self.pad_seq_pair(samples)   #同样以占用较低内存的方式生成迭代器

class Batch(object):
    def __init__(self):
        self.sent = None
        self.label = None
        self.mask = None
