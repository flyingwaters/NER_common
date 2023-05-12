# NER_common
一个领域NER的通用框架，领域数据迁移，NER训练和多fold集成

(1) pretrain ner的领域数据，
python pretrain.py
默认是进一步做 MLM-wwm任务的loss，做进一步领域数据的预训练迁移 

1. mcbert-医学ali
2. roberta-wwm-large/ext 中文哈工大讯飞

####后续： 
1. 增加smart-span-mask任务 ---》需要重新定义pretrain_utils的动态smart_DataCollator                   


(2) NER的集成和训练
sh run.sh

参考 [https://github.com/z814081807/DeepNER](https://github.com/z814081807/DeepNER)，非常完整的一个基于BERT的NER的开源项目，做高标准的NER项目的baseline非常好，
做了部分修改，每个epoch验证val，打印指标

####后续:
1. 改写accelerate 分布式加速训练框架
2. global pointer 和w2ner等最新的ner model 
3. 显示的数据增强模块 （4）可信机器学习中的分布偏移的loss等模块引入



