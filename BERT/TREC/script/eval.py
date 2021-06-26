import csv
from utils import eval

id, relevant = list(), list()
with open('../dataset/model/prediction.csv', newline='') as csvfile:
    questions = csv.reader(csvfile)
    for i, query in enumerate(questions):
        if query[0] == "query_id":
            continue
        id.append(query[0])
        relevant.append([j for j in query[1].split()] )

print(len(relevant))
eval(relevant, "../dataset/partial/test/new_topK.csv", len(id))