from pathlib import Path
import os
import json
import csv
import random
from tqdm import tqdm
import pickle

random.seed(0)
chunk = 200
with open("../model/file2label.json", "r") as f:
	doc2label = json.load(f)
# print(doc2label)
with open("../model/error_doc.pk", 'rb') as f:
	unpickler = pickle.Unpickler(f)
	error_doc = unpickler.load()
print(len(error_doc))

# with open("../model/train_questions.json", "r") as f:
# 	questions_dict = json.load(f)
# train_data = list()
# print("processing training data...")
# with open('../partial/train/topK.csv', newline='') as csvfile:
# 	questions = csv.reader(csvfile)
# 	all_doc = set(range(len(doc2label))) - set(error_doc)
# 	for i, query in enumerate(tqdm(questions)):
# 		question = questions_dict[query[0]]
# 		relevents = query[1].split(" ")[:2]
# 		# print(question, relevents)
# 		relevent_set = set(relevents)
# 		# weight = list(range(2, 0, -1))
# 		# weight += [1]*(len(relevents) - len(weight))
# 		# # print(len(relevents))
# 		new_relevants = list()
# 		for i, rel in enumerate(relevents):
# 			new_relevants += [doc2label[rel]]
# 		# relevents = [[doc2label[rel]]*weight[count] for count, rel in enumerate(relevents)]
# 		# new_relevants = relevents[:2]
# 		# print(len(new_relevants))
# 		# print(len(all_doc))
# 		sample_set = all_doc - relevent_set
# 		for rel in new_relevants:
# 			paragraphs = random.sample(sample_set, chunk-1)
# 			paragraphs.append(rel)
# 			random.shuffle(paragraphs)
# 			train_dic = dict()
# 			train_dic["id"] = query[0]
# 			train_dic["question"] = question
# 			train_dic["paragraphs"] = list(paragraphs)
# 			train_dic["relevant"] = rel
# 			train_data.append(train_dic)
# random.shuffle(train_data)
# print("train size:", len(train_data))
# with open("../model/train.json", 'w') as f:
# 	json.dump(train_data, f, indent=2)


# with open("../model/dev_questions.json", "r") as f:
# 	questions_dict = json.load(f)
# dev_data = list()
# print("processing deving data...")
# with open('../partial/dev/topK.csv', newline='') as csvfile:
# 	questions = csv.reader(csvfile)
# 	all_doc = set(range(len(doc2label))) - set(error_doc)
# 	for i, query in enumerate(tqdm(questions)):
# 		question = questions_dict[query[0]]
# 		relevents = query[1].split(" ")[:1]
# 		# print(question, relevents)
# 		relevent_set = set(relevents)
# 		# weight = list(range(2, 0, -1))
# 		# weight += [1]*(len(relevents) - len(weight))
# 		# # print(len(relevents))
# 		new_relevants = list()
# 		for i, rel in enumerate(relevents):
# 			new_relevants += [doc2label[rel]]
# 		# relevents = [[doc2label[rel]]*weight[count] for count, rel in enumerate(relevents)]
# 		# new_relevants = relevents[:2]
# 		# print(len(new_relevants))
# 		# print(len(all_doc))
# 		sample_set = all_doc - relevent_set
# 		for rel in new_relevants:
# 			paragraphs = random.sample(sample_set, chunk-1)
# 			paragraphs.append(rel)
# 			random.shuffle(paragraphs)
# 			dev_dic = dict()
# 			dev_dic["id"] = query[0]
# 			dev_dic["question"] = question
# 			dev_dic["paragraphs"] = list(paragraphs)
# 			dev_dic["relevant"] = rel
# 			dev_data.append(dev_dic)
# random.shuffle(dev_data)
# print("dev size:", len(dev_data))
# with open("../model/dev.json", 'w') as f:
# 	json.dump(dev_data, f, indent=2)


def chunkIt(seq, num):
    out = []
    last = 0
    while last < len(seq):
        out.append(seq[last:min(last + num, len(seq))])
        last += num
    return out
print("processing testing data...")
with open("../model/new_test_questions.json", "r") as f:
	questions_dict = json.load(f)
test_data = list()
all_doc = list(set(range(len(doc2label))))
for key, question in tqdm(questions_dict.items()):
	n_paragraphs = chunkIt(all_doc, chunk)
	for paragraphs in n_paragraphs:
		test_dic = dict()
		test_dic["id"] = key
		test_dic["question"] = question
		test_dic["paragraphs"] = paragraphs
		test_data.append(test_dic)
print("test size:", len(test_data))
with open("../model/test.json", 'w') as f:
	json.dump(test_data, f, indent=2)