import json
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup

import utils
from DR_Dataset import DR_Dataset
from accelerate import Accelerator
import time
import random
import os

TRAIN = "train"
DEV = "dev"
SPLITS = [TRAIN, DEV]

def main(args):
    context_path = args.data_dir / "context.json"
    contexts = json.loads(context_path.read_text())
    contexts = [context for context in contexts]
    # print(contexts[0])
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # print(data[TRAIN][0]["question"], data[DEV][0]["question"])
    
    if (args.start_from_last):
        print("load from last...")
        DR_Model = BertForMultipleChoice.from_pretrained(args.ckpt_dir).to(device)
    else:
        DR_Model = BertForMultipleChoice.from_pretrained("hfl/chinese-macbert-large").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    
    context_tokenized = tokenizer(contexts, add_special_tokens=False)
    train_questions_tokenized = tokenizer([train_question["question"] for train_question in data[TRAIN]], add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question"] for dev_question in data[DEV]], add_special_tokens=False)
    # print(train_questions_tokenized[0], dev_questions_tokenized[0], test_questions_tokenized[0])
    # print(context_tokenized[0].ids, train_questions_tokenized[0].ids, dev_questions_tokenized[0].ids, test_questions_tokenized[0].ids)
    
    train_set = DR_Dataset(TRAIN, data[TRAIN], train_questions_tokenized, context_tokenized)
    dev_set = DR_Dataset(DEV, data[DEV], dev_questions_tokenized, context_tokenized)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(DR_Model.parameters(), lr=args.lr)
    print(args.lr)
    update_step = args.num_epoch * len(train_loader) // args.gradient_accumulation_step + args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * update_step, update_step)
    
    best_acc = -1
    for epoch in range(args.num_epoch):
        train_size, dev_size = len(train_loader.dataset), len(dev_set)
        print(train_size, dev_size)
        DR_Model.train()
        train_loss = train_acc = 0
        start_time = time.time()
        for i, datas in enumerate(tqdm(train_loader)):
            datas = [data.to(device) for data in datas]
            output = DR_Model(input_ids=datas[0], token_type_ids=datas[1], attention_mask=datas[2], labels=datas[3])
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += (torch.argmax(output.logits, dim=1)==datas[3]).float().mean()
            train_loss += output.loss
            normalized_loss = output.loss / args.gradient_accumulation_step
            
            if fp16_training:
                accelerator.backward(normalized_loss)
            else:
                normalized_loss.backward()
            
            if i % args.gradient_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            # Print training loss and accuracy over past logging step
            if i % args.logging_step == 0 and i != 0:
                print(f"Epoch {epoch + 1}/{args.num_epoch} | loss = {train_loss.item() / args.logging_step:.3f}, acc = {train_acc / args.logging_step:.3f} lr = {optimizer.param_groups[0]['lr']:.6f}")
                train_loss = train_acc = 0
                DR_Model.eval()
                with torch.no_grad():
                    dev_acc = 0
                    randomlist = random.sample(range(0, len(dev_set)), len(dev_set) // 1)
                    dev_subset = Subset(dev_set, randomlist)
                    dev_loader = DataLoader(dev_subset, batch_size=1, shuffle=False)
                    for i, datas in enumerate(dev_loader):
                        datas = [data.to(device) for data in datas]
                        output = DR_Model(input_ids=datas[0], token_type_ids=datas[1], attention_mask=datas[2], labels=datas[3])
                        dev_acc += (torch.argmax(output.logits, dim=1)==datas[3]).float().mean() / len(dev_loader)
                        print(f"Validation | Steps {i}/{len(dev_loader)} | acc = {dev_acc:.3f}", end="\r")
                        # break
                    print(f"Validation | Steps {i}/{len(dev_loader)} | acc = {dev_acc:.3f}")
                    if (dev_acc >= best_acc):
                        print("Saving model...with acc: {}".format(dev_acc))
                        best_acc = dev_acc
                        DR_Model.save_pretrained(args.ckpt_dir)
                # break

            DR_Model.train()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../dataset/model/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="../ckpt/",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--start_from_last", action="store_true")
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--gradient_accumulation_step", type=int, default=40)
    parser.add_argument("--logging_step", type=int, default=871)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    utils.same_seeds(19991210)
    fp16_training = True
    if fp16_training:
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)