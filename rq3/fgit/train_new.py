import datetime
from email.policy import default
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,Trainer,DefaultDataCollator
from datasets import load_dataset,load_from_disk,concatenate_datasets
import torch
import logging
import os
import sys
import json
from data import PairGeneralSampler, prepare_dataset, DataCollatorForDiffSFT
from loss_radio import DiffSFTTrainer, CustomTrainingArguments
import argparse
from accelerate import Accelerator
from accelerate import Accelerator


import numpy as np

accelerator = Accelerator()
    

def setup_logging(output_dir: str):
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "train.log"))
        ]
    )
    return logging.getLogger(__name__)

def load_and_process_data(data_path: str, tokenizer: AutoTokenizer, max_length: int = 2048, train_ratio: float = 0.95, line_weight: int = 1, 
                         token_weight: int = 1):

    dataset = load_from_disk(data_path)

    total_size = len(dataset)
    train_dataset = dataset.select(range(total_size))

    eval_dataset = dataset.select(range(total_size-100, total_size))


     
    train_dataset = prepare_dataset(
        problems=train_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        diff_level=args.diff_level,
        hybrid=args.hybrid,
        line_weight=line_weight,
        token_weight=token_weight,
    )
    
    eval_dataset = prepare_dataset(
        problems=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        hybrid=args.hybrid,
        line_weight=line_weight,
        token_weight=token_weight
    )
 
    return train_dataset, eval_dataset

def model_init():
    
    return AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to use for training")
    parser.add_argument("--max_length", type=int, default=1024) 
    parser.add_argument("--diff_level", type=str, default='token'),
    parser.add_argument("--hybrid", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-6,choices=[5e-6,1e-5])
    parser.add_argument("--warmup_steps",type=int, default=5) 
    
    global args
    args = parser.parse_args()
    args.output_dir = './RLHF_Model/'+ args.output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,model_max_length=args.max_length,use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    
    train_dataset, eval_dataset = load_and_process_data(
        args.train_data_path,
        tokenizer,
        max_length=args.max_length, 
        train_ratio=args.train_ratio
    )
    
    data_collator = DefaultDataCollator()
    
    
    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=128,
        lr_scheduler_type='linear',
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_first_step=True,
        bf16=True,
        report_to="none",
        optim="adafactor",
        run_name="10k-diff-sft-token",
        ddp_find_unused_parameters=False,
    )


    
    trainer = DiffSFTTrainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_init=model_init,
    )

    
    trainer.train()
        
    trainer.save_state()
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))


if __name__ == "__main__":
    main()
