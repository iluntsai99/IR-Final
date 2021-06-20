from pathlib import Path
import os
import json
import xml.etree.ElementTree as ET

with open("../model/file2label.json", "r") as f:
	label = json.load(f)
# print(label)

context = []
for key, value in label.items():
	with open(f"../CIRB010" / Path(key), 'r', encoding="utf-8") as xml_f:
		document = ET.parse(xml_f)
		Droot = document.getroot()
		doc = Droot.find("doc")
		try:
			title = doc.find("title").text.strip()
		except:
			title = ""
		text = doc.find("text")
		tag_ps = text.findall('p')
		paragraph_count = 0
		ps = [""]*len(tag_ps)
		for i, p in enumerate(tag_ps):
			ps[i] = p.text.strip().split("。")[0]
			start = ps[i].find('【')
			if start != -1:
				end = ps[i].find('】')
				ps[i] = ps[i][:start] + ps[i][end+1:]
		doc = [title] + ps
		context.append("。".join(doc).replace("相關文件內容", "").replace("包括", "").replace("應", "").replace("說明", "").replace("應說明", "").replace("查詢", "").replace("\n", "").replace(" ", "，"))

with open("../model/context.json", "w") as f:
	json.dump(context, f, indent=2)