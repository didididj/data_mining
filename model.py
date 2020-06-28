import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, config):
        super(TextCNN, self).__init__()
        self.kernel_sizes = config.kernel_sizes
        self.hidden_dim = config.embed_dim   #词嵌入的维度，根据所要表示的元素的复杂度而定
        self.num_channel = config.num_channel
        self.num_class = config.num_class
        self.word_embedding = nn.Embedding(
            vocab_size, config.embed_dim)  #体现词间的某些联系
        #one-hot编码：一个属性如果有N个可取值，它就可以扩充为N个属性，每个样本的这N个属性中
        #只能有一个为1，表示该样本的该属性属于这个类别，其余扩展属性都为0.
        #嵌入层embedding的第一步是通过索引对该句子进行编码，即给每一个不同的单词分配一个索引
        #第二步创建嵌入矩阵，要决定每一个索引需要分配多少个“潜在因子”；通过潜在因子体现词间相似性；同时保持向量更小
        #每个单词用查找嵌入矩阵中向量的索引来表示
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channel, config.num_kernel, (kernel, config.embed_dim))
             for kernel in self.kernel_sizes])  # 卷积层

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_kernel*3, self.num_class)  # 全连接层，采用了三个不同大小的卷积核，故第一维度*3

    def forward(self, x):
        x = self.word_embedding(x)   # [len(sent), batch_size, embed_dim]
        x = x.permute(1, 0, 2).unsqueeze(1)   #将tensor的维度换位，在第二维度增加一个维度
        #[batch_size, 1, len(sent), embed_dim]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs] #将第四维去掉
        #[batch_size, num_kernel, len(sent)-kernel+1, 1]
        #[batch_size, num_kernel, len(sent)-kernel+1]
        x = [torch.max_pool1d(h, h.size(2)).squeeze(2) for h in x] #[batch_size, num_kernel]
        x = torch.cat(x, 1)   #将张量(tensor)列拼接在一起
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
