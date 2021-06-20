from pathlib import Path
import os
import json

with open("../model/file-list", 'r') as f:
	all_lines = f.readlines()
label = dict()
for i in range(len(all_lines)):
	label[all_lines[i].strip()] = i
# print(label)
with open("../model/file2label.json", "w") as f:
	json.dump(label, f, indent=2)

with open("../model/file-list", 'r') as f:
	all_lines = f.readlines()
label = dict()
for i in range(len(all_lines)):
	label[i] = all_lines[i].strip().split('/')[-1].lower()
# print(label)
with open("../model/label2file.json", "w") as f:
	json.dump(label, f, indent=2)

	# cdn_loc_0001911
