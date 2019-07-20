# 
# 使用示例：
# from config import Config
# config = Config()
# model = config.model['model2']    # 使用已有的变量或常量
# config.var3 = 'xxx'               # 新增新的变量或常量并赋值  TODO 疑问：新增变量能传递给别的文件么？若不能，Config中事先定义一些global变量呢？


# model是文件夹，model1,2,3对应文件model1.py等，Model1是model1.py中定义的模型类
from model.model1 import Model1
from model.model2 import Model2
from model.model3 import Model3

class Config(object):

    def __init__(self):

        # 定义一般常量
        self.VAR1 = 'xxx'
        self.VAR2 = 'xxx'

        # 常规文件和路径
        self.PATH1 = 'xxx'
        self.PATH2 = 'xxx'

        # 自定义模型变量，尤其适用于多个模型时，这些是自己定义好的模型，方便调用
        self.model = {
            'model1': Model1,
            'model2': Model2,
            'model3': Model3
        }