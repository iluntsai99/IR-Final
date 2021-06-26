import json
from os import chdir
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup

import utils
from DR_Dataset import DR_Dataset
from accelerate import Accelerator
import time
from utils import eval
import pickle

TEST = "test"

def main(args):
    SPLITS = [TEST]
    context_path = args.context_path
    print("loading context...")
    contexts = json.loads(context_path.read_text())
    print(contexts[0])
    data_paths = {split: args.data_path for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    DR_Model = BertForMultipleChoice.from_pretrained(args.ckpt_DR_dir).to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    context_tokenized = tokenizer(contexts, add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question"] for test_question in data[TEST]], add_special_tokens=False)
    test_DR_set = DR_Dataset(TEST, data[TEST], test_questions_tokenized, context_tokenized)
    test_DR_loader = DataLoader(test_DR_set, batch_size=1, shuffle=False)

    with open("../dataset/model/label2file.json", "r") as f:
	    label2file = json.load(f)

    DR_Model.eval()
    print(args.ranked_list)
    with open("../dataset/model/error_doc.pk", 'rb') as f:
        unpickler = pickle.Unpickler(f)
        error_doc = unpickler.load()
    with torch.no_grad():
        id, predictions = list(), list()
        prev_id = ""
        cur_id, rank = 0, 0
        # softmax = torch.nn.Softmax(dim=1)
        for i, datas in enumerate(tqdm(test_DR_loader)):
            output = DR_Model(input_ids=datas[0].to(device), token_type_ids=datas[1].to(device), attention_mask=datas[2].to(device))
            cur_id = datas[3][0]
            if cur_id != prev_id:
                if prev_id != "":
                    prob, relevant_documents = torch.topk(rank, k=200, dim=1)
                    # print(prob)
                    # print(relevant_documents)
                    # print(prob.shape, relevant_documents.shape)
                    id.append(prev_id)
                    relevant_documents = torch.reshape((relevant_documents), (-1,))
                    relevant_documents = relevant_documents.detach().tolist()
                    relevant_documents[:] = [rel for rel in relevant_documents if rel not in error_doc]
                    prediction = [label2file[str(label)] for label in relevant_documents]
                    print("question:", data[TEST][i-1]["question"][:64])
                    for j in range(5):
                        print("document:", prediction[j], contexts[relevant_documents[j]][:445])
                    predictions.append(prediction)
                    # print(prediction)
                    print(prev_id, len(prediction))
                
                rank = output.logits
                prev_id = cur_id
            else:
                rank = torch.cat((rank, output.logits), dim=1)
                # print(rank.shape)
        # for last query
        _, relevant_documents = torch.topk(rank, k=120, dim=1)
        id.append(prev_id)
        relevant_documents = torch.reshape((relevant_documents), (-1,))
        relevant_documents = relevant_documents.detach().tolist()
        relevant_documents[:] = [rel for rel in relevant_documents if rel < len(label2file)]
        prediction = [label2file[str(label)] for label in relevant_documents]
        predictions.append(prediction)
        print(prev_id, len(prediction))

    with open(args.ranked_list, 'w') as f:
        print("Writing result to {}".format(args.ranked_list))
        f.write("query_id,retrieved_docs\n")
        for i, prediction in enumerate(predictions):
            f.write("{},{}\n".format(id[i], " ".join(prediction)))
    print(f"Completed! Result is in {args.ranked_list}")
    eval(predictions, args.ground_truth, len(id))



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Directory to the dataset.",
    )
    parser.add_argument(
        "--ckpt_DR_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Public or private",
    )
    parser.add_argument(
        "--ranked_list",
        type=Path,
        help="Prediction file",
    )
    parser.add_argument("-g", type=Path, default="../dataset/partial/test/new_topK.csv", dest="ground_truth")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    utils.same_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    main(args)
