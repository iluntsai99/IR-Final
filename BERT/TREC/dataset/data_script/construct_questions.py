from pathlib import Path
import os
import json
import xml.etree.ElementTree as ET

questions_list = list()
with open("../queries/query-train.xml", 'r', encoding="utf-8") as xml_f:
	query_file = ET.parse(xml_f)
	Qroot = query_file.getroot()
	topic_list = Qroot.findall("topic")
	for topic in topic_list:
		queryID = topic.find("number").text.strip()
		title = topic.find("title").text.strip()
		question = topic.find("question").text.strip().replace("。", "？")
		question = "".join([question, title]).replace("相關文件內容", "").replace("包括", "").replace("應", "").replace("說明", "").replace("應說明", "").replace("查詢", "").replace(" ", "")
		query_dic = dict()
		query_dic["id"] = queryID
		query_dic["question"] = question
		questions_list.append(query_dic)
print(len(questions_list))
with open("../queries/query-test.xml", 'r', encoding="utf-8") as xml_f:
	query_file = ET.parse(xml_f)
	Qroot = query_file.getroot()
	topic_list = Qroot.findall("topic")
	for topic in topic_list:
		queryID = topic.find("number").text.strip()
		title = topic.find("title").text.strip()
		question = topic.find("question").text.strip().replace("。", "？")
		question = "".join([question, title]).replace("相關文件內容", "").replace("包括", "").replace("應", "").replace("說明", "").replace("應說明", "").replace("查詢", "").replace(" ", "")
		query_dic = dict()
		query_dic["id"] = queryID
		query_dic["question"] = question
		questions_list.append(query_dic)
print(len(questions_list))

with open("../model/questions.json", "w") as f:
	json.dump(questions_list, f, indent=2)