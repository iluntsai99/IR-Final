from pathlib import Path
import os
import json

with open("../model/file-list", 'r') as f:
	all_lines = f.readlines()
label = dict()
for i in range(len(all_lines)):
	key = all_lines[i].strip().split('/')[-1].lower()
	label[key] = i
# print(label)
with open("../model/file2relevant.json", "w") as f:
	json.dump(label, f, indent=2)
