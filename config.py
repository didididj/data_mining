#将所有的代码和配置都变成模块化可配置化，这样就提高了代码的重用性，不再每次都去修改代码内部，这个就是我们逐步要做的事情，可配置化
#.py的配置文件，在python项目中是作为一个包导入，严格来说不是配置文件，而是扩展包
import os
from argparse import ArgumentParser
from configparser import ConfigParser
import torch


class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()    #创建ConfigParser对象
        config_file = 'cnn.ini'
        raw_config.read(config_file)   #调用read()函数打开【1】配置文件，里面填的参数是地址
        self.cast_values(raw_config)   

    def cast_values(self, raw_config):
        for section in raw_config.sections(): #得到所有的section，并以列表的形式返回
            for key, value in raw_config.items(section): #将配置文件cnn.ini中的情况以键值对的形式存储
                val = None        
                #将[]形式的部分先转换为self.key = val 的形式
                if type(value) is str and value.startswith("[") and value.endswith("]"): 
                    #将value的类型为str，并且以'['开头,']'结尾部分提取出来，形成对应的键值对
                    val = eval(value)  #执行value所表示的字符串表达式，并返回表达式的值；
                    setattr(self, key, val)  #设置属性值 self.key = val
                    continue
                #处理非[]形式的部分；
                for attr in ["getint", "getfloat", "getboolean"]:
                    #在可能出错的代码前加上try,当捕获到错误后在except下处理
                    try:
                        #找出int,float,bool类型的部分赋值
                        val = getattr(raw_config[section], attr)(key)
                        #获取属性 attr 值，并且赋值给val
                        break
                    except:
                        val = value  #若非以上形式，直接进行赋值即可
                setattr(self, key, val) #统一为 self.key = val形式
            

#argparse是python用于解析命令行参数和选项的标准模块，类似于linux中的ls指令，后面可以跟着不同的参数选项以实现不同的功能
#argparse就可以解析命令行然后执行相应的操作。
def parse_config():
    #使用argparse配置命令行参数
    parser = ArgumentParser(description="Text CNN")  #创建ArgumentParser()对象
    #description给出程序做什么以及如何工作的简短描述;
    #ArgumentParser对象包含将命令行解析成python数据类型所需的全部信息；
    parser.add_argument('--config', dest='config', default='CONFIG') #调用add_argument()方法添加参数
    # action='store_true') # for debug
    parser.add_argument('--train', dest="train", default=True)
    # action='store_true') # for debug
    parser.add_argument('--test', dest="test", default=True)
    parser.add_argument('-v', '--verbose', default=False)
    #add_argument 第一个参数指参数名字
    #dest：对于位置参数的动作，被添加到parse_args()所返回对象上的属性名
    #default 指如果命令行参数没有出现时它们应该是什么值；

    args = parser.parse_args()     #使用parse_args解析添加的参数
    config = Config(args.config)

    config.train = args.train
    config.test = args.test
    #设置日志显示
    config.verbose = args.verbose
    
    #config.batch_size = 32   #对数据集随机抽取batch进行训练min_freq
    #config.kernel_sizes = (3,3)
    #卷积核的尺寸，卷积核的深度与当前图像的深度相同，所以指定卷积核时，只需指定其长和宽两个参数
    #config.dropout = 0.5    
    #主要任务是在训练时，通过设置阈值并与某些隐含层节点的权值对比，让特定的权重不工作，即舍弃这些权重
    #主要作用加快运算的同时，还防止过拟合。
    #dropout为完全的随机抛弃，最主要的作用是讲特征信息完全打散，可以避免各个特征之间的相互依赖性；
    #config.data_path = 'E:/2020年春季学期/数据挖掘/新版代码与数据/tnews_public'
    #config.save_vocab = 'E:/2020年春季学期/数据挖掘/新版代码与数据/results/vocab'
    #config.min_freq = 5
    #config.max_seq_len = 25
    #config.num_channel = 1
    #通道数channel，一般的RGB图片channel数为3，灰度图为1，文本处理应该也为单通道情况
    #config.embed_dim = 5000 #设定词嵌入的维度为5000
    #config.num_class = 18 #类别数
    #config.num_kernel = 6
    #config.lr = 0.001  #学习率
    #config.epochs = 10 #使用训练集的全部数据对模型进行完整训练的次数
    #声明使用设备,gpu or cpu，将torch.Tensor分配到的设备的对象。
    config.device = torch.device('cpu')
        # "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    return config
