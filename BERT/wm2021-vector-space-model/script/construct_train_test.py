from pathlib import Path
import os
import json
import csv

with open("../model/questions.json", "r") as f:
	questions_list = json.load(f)
with open("../model/file2relevant.json", "r") as f:
	doc2label = json.load(f)
# print(doc2label)
train_list = list()
with open('../queries/ans_train.csv', newline='') as csvfile:
	questions = csv.reader(csvfile)
	train_dic = dict()
	for i, query in enumerate(questions):
		if query[0] == "query_id":
			continue
		train_dic["id"] = query[0]
		train_dic["question"] = questions_list[i - 1]["question"]
		relevents = query[1].split(" ")
		for rel in relevents:
			if "relevant" not in train_dic:
				train_dic["relevant"] = list()
			train_dic["relevant"].append(doc2label[rel])
		train_list.append(train_dic)
test_list = list()
for question in questions_list[10:]:
	test_dic = dict()
	test_dic["id"] = question["id"]
	test_dic["question"] = question["question"]
	test_list.append(test_dic)
print(len(train_list), len(test_list))

with open("../model/train.json", 'w') as f:
	json.dump(train_list, f, indent=2)
with open("../model/test.json", 'w') as f:
	json.dump(test_list, f, indent=2)