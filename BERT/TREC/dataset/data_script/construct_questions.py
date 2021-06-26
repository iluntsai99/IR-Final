from pathlib import Path
import os
import json
import csv

train_queries = open("../partial/train/queries.tsv")
read_tsv = csv.reader(train_queries, delimiter="\t")
train_question = dict()
for id, question in read_tsv:
    train_question[id] = question.strip()
with open("../model/train_questions.json", "w") as f:
	json.dump(train_question, f, indent=2)
print("finish train")

dev_queries = open("../partial/dev/queries.tsv")
read_tsv = csv.reader(dev_queries, delimiter="\t")
dev_question = dict()
for id, question in read_tsv:
    dev_question[id] = question.strip()
with open("../model/dev_questions.json", "w") as f:
	json.dump(dev_question, f, indent=2)
print("finish dev")

test_queries = open("../partial/test/queries.tsv")
read_tsv = csv.reader(test_queries, delimiter="\t")
test_question = dict()
for id, question in read_tsv:
    test_question[id] = question.strip()
with open("../model/test_questions.json", "w") as f:
	json.dump(test_question, f, indent=2)
print("finish test")