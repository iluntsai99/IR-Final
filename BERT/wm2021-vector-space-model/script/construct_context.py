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
		p1 = text.find('p').text.strip().split("。")[0]
		context.append("。".join([title, p1]).replace("相關文件內容", "").replace("包括", "").replace("應", "").replace("說明", "").replace("應說明", "").replace("查詢", ""))
with open("../model/context.json", "w") as f:
	json.dump(context, f, indent=2)