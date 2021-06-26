from pathlib import Path
import os
import json
import csv
import random
from tqdm import tqdm

random.seed(6396969)
sample = random.sample(list(range(200)), 10)
sample.sort()
test_queries = open("../partial/test/queries.tsv")
read_tsv = csv.reader(test_queries, delimiter="\t")
test_question = dict()
for i, query in enumerate(read_tsv):
    if i in sample:
        test_question[query[0]] = query[1].strip()
with open("../model/new_test_questions.json", "w") as f:
	json.dump(test_question, f, indent=2)

id, relevant = list(), list()
with open('../partial/test/topK.csv', newline='') as csvfile:
    questions = csv.reader(csvfile)
    for i, query in enumerate(tqdm(questions)):
        id.append(query[0])
        relevant.append(query[1])
with open("../partial/test/new_topK.csv", 'w') as f:
    for i in sample:
        f.write("{},{}\n".format(id[i], relevant[i]))
