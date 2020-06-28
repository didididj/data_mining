# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from model import TextCNN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging
import sys
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins,elapsed_secs

def build_textcnn_model(vocab, config, train=True):
    model = TextCNN(vocab.vocab_size, config)
    if train:
        model.train()  
        #在训练模型时会在前面加上train();
    else:
        model.eval()
        #在测试模型时在前面使用eval()，会将BN和DropOut固定住，不会取平均，而是用训练好的值

    #train()与eval()两个方法是针对网络train和eval时采用不同方式的情况
    #比如Batch Normalization和Dropout
    #BN的作用主要是对网络中间的每层进行归一化处理，并且使用变换重构保证所提取的特征分布不会被破坏；
    #由于训练完毕后参数都是固定的，所有BN的训练和测试时的操作不同
    #Dropopt能够克服过拟合，在每个训练batch中，通过忽略一般的特征检测器，可以明显地减少过拟合现象。
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model


def train_textcnn_model(model, train_data, valid_data, padding_idx, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    mylog = open('result.log', mode = 'a',encoding='utf-8')
    params = [p for k, p in model.named_parameters() if p.requires_grad]  #提取需要求梯度参数
    optimizer = Adam(params, lr=config.lr)  #定义优化器
    #params--可进行迭代优化的包含了所有参数的列表； lr--学习速率
    #构建优化器对象，保存当前的状态，并能够根据计算得到的梯度来更新参数
    #根据网络反向传播的梯度信息来更新网络的参数，从而起到降低loss函数计算值的作用
    criterion = CrossEntropyLoss(reduction="sum")  #定义损失函数
    #以交叉熵作为损失函数，描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近
    model.train()
    best_loss = 999
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        train_data_iter = iter(train_data)

        #生成train_data的迭代器
        total_loss = 0.
        num = 0
        for idx, batch in enumerate(train_data_iter):
            model.zero_grad()
            #将梯度设置为0
            #根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是
            #被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累计，由此需要每个batch设置一遍归零
            ground_truth = batch.label  #统计真实标签
            # batch_first = False
            outputs = model(batch.sent) #统计模型训练得到的模型 
            loss = criterion(outputs, ground_truth) #计算损失函数值
            total_loss += loss
            loss.backward()   #反向传播
            optimizer.step()  #使用当前参数空间对应的梯度更新参数空间
            #mini-batch训练模式假定每一个训练集就只有mini-batch这样大，由此实际上可以将每一次mini-batch看作是依次训练；
            #每一次训练都需要更新参数空间
            num = num + 1
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s', file=mylog)
        train_loss = total_loss/num
        print("Train Set epoch {0:d} [{1:d}/{2:d}], valid loss: {3:.2f}".format(
                    epoch, idx, train_data.num_batches, train_loss), file=mylog)
        #每个eopch训练样本调用一次验证集。
        valid_loss = valid_textcnn_model(model, valid_data, criterion, config)
        # 处理
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model, '%s.pt' % (config.save_model))
        print("Valid Set epoch {0:d} [{1:d}/{2:d}], valid loss: {3:.2f}".format(
                    epoch, idx, train_data.num_batches, valid_loss), file=mylog)
        model.train()
    mylog.close()



def valid_textcnn_model(model,  valid_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()   
    #使用eval()函数，让model变成测试模式
    #dropout和batch normalization的操作在训练和测试的模式下是不一样的。
    total_loss = 0.
    num = 0
    valid_data_iter = iter(valid_data)
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()    #每一轮batch后梯度归零
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        loss = criterion(outputs, batch.label)  #得到损失函数值
        # loss 打印
        # 处理
        total_loss += loss   #计算总体损失值
        num = num + 1
    return total_loss/num


def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    vocab = train_data.vocab

    # 载入测试集合
    valid_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="dev",
        batch_size=config.batch_size)
    valid_data.set_vocab(vocab)
    valid_data.load()

    # 构建textcnn模型
    model = build_textcnn_model(vocab, config, train=True)

    print(model)

    # Do training.
    padding_idx = vocab.stoi[PAD]
    train_textcnn_model(model, train_data,
                        valid_data, padding_idx, config)

    # 测试时
    # checkpoint = torch.load(config.save_model+".pt",
    #                      map_location = config.device)
    # checkpoint
    # model = build_textcnn_model(
    #     vocab, config, train=True)
    # .....
if __name__ == "__main__":
    main()
