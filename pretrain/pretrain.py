import os
import csv
from transformers import  BertTokenizer, WEIGHTS_NAME,TrainingArguments, BertForMaskedLM, BertConfig
import tokenizers
import torch
from datasets import load_dataset,Dataset 
from accelerate import notebook_launcher

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)

## 加载tokenizer和模型
# 
def train_train_trainer_ddp():
    model_path='hfl/chinese-roberta-wwm-ext-large'
    token_path='hfl/chinese-roberta-wwm-ext-large'

    pretrain_batch_size=36
    num_train_epochs=50
    weight_decay=1e-1
    learning_rate=1e-5
    save_strategy="epoch"
    evaluation_strategy = "epoch"
    save_total_limit=2
    output_dir='outputs/'


    tokenizer =  BertTokenizer.from_pretrained(token_path, do_lower_case=True)
    config=BertConfig.from_pretrained(model_path)
    model=BertForMaskedLM.from_pretrained(model_path, config=config)

    # 通过LineByLineTextDataset接口 加载数据 #长度设置为128, # 这里file_path于本文第一部分的语料格式一致
    train_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path='../pretrain.txt',block_size=512)
    dev_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path='../pretrain_dev.txt',block_size=512)
    # MLM模型的数据DataCollator
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
    # 
    # 训练参数

    training_args = TrainingArguments(
        output_dir=output_dir, weight_decay=weight_decay, resume_from_checkpoint=False, 
        overwrite_output_dir=False, num_train_epochs=num_train_epochs, learning_rate=learning_rate, 
        per_device_train_batch_size=pretrain_batch_size, save_strategy=save_strategy, 
        evaluation_strategy = evaluation_strategy, save_total_limit=save_total_limit)# save_steps=10000

    # 通过Trainer接口训练模型
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=dev_dataset)

    # 开始训练
    trainer.train()
    trainer.save_model(output_dir)
    print(f"pretraining process has ended and the model is saved at {output_dir}")
    #clear the cuda
    torch.cuda.empty_cache()


on_jupyter = False
if __name__=="__main__":
    if on_jupyter:
        notebook_launcher(train_trainer_ddp, args=(), num_processes=2)
    else:
        train_trainer_ddp()