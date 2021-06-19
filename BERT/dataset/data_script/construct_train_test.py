from pathlib import Path
import os
import json
import csv
import random
from tqdm import tqdm

with open("../model/questions.json", "r") as f:
	questions_list = json.load(f)
with open("../model/file2relevant.json", "r") as f:
	doc2label = json.load(f)
# print(doc2label)
train_data = list()
with open('../queries/ans_train.csv', newline='') as csvfile:
	questions = csv.reader(csvfile)
	for i, query in enumerate(tqdm(questions)):
		if query[0] == "query_id":
			continue
		query_id = query[0]
		question = questions_list[i - 1]["question"]
		relevents = query[1].split(" ")
		relevents = [doc2label[rel] for rel in relevents]
		all_doc = set(range(len(doc2label)))
		relevent_set = set(relevents)
		for rel in relevents:
			paragraphs = random.sample(all_doc - relevent_set, 299)
			paragraphs.append(rel)
			random.shuffle(paragraphs)
			train_dic = dict()
			train_dic["id"] = query_id
			train_dic["question"] = question
			train_dic["paragraphs"] = list(paragraphs)
			train_dic["relevant"] = rel
			train_data.append(train_dic)

def chunkIt(seq, num):
    out = []
    last = 0
    while last < len(seq):
        out.append(seq[last:min(last + num, len(seq))])
        last += num
    return out

test_list = list()
for question in questions_list[10:]:
	all_doc = list(range(len(doc2label)))
	n_paragraphs = chunkIt(all_doc, 300)
	query_id = question["id"].split("ZH")[1]
	question = question["question"]
	for paragraphs in n_paragraphs:
		test_dic = dict()
		test_dic["id"] = query_id
		test_dic["question"] = question
		test_dic["paragraphs"] = paragraphs
		test_list.append(test_dic)

random.shuffle(train_data)
train_list = train_data[:200]
dev_list = train_data[200:]
print(len(train_list), len(dev_list), len(test_list))

with open("../model/train.json", 'w') as f:
	json.dump(train_list, f, indent=2)
with open("../model/dev.json", 'w') as f:
	json.dump(dev_list, f, indent=2)
with open("../model/test.json", 'w') as f:
	json.dump(test_list, f, indent=2)