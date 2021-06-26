import csv
import io
import os
import sys
import numpy as np
import json
from tqdm import tqdm
import pickle
from unidecode import unidecode

def getcontent(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    line = line.strip().split("\t")
    return line

dir_path = "../../"
# In the corpus tsv, each docid occurs at offset docoffset[docid]
docoffset = {}
with open(os.path.join(dir_path, "data/corpus/msmarco-docs-lookup.tsv"), encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        docoffset[docid] = int(offset)

context = []
with open(os.path.join(dir_path, "data/corpus/msmarco-docs.tsv"), encoding="utf8") as f, \
        open("../partial/corpus/docIDs") as partialCorpusID_f:

    corpus_reader = csv.reader(f, delimiter="\t")
    corpusID = list(map(lambda line: line.strip(),
                        partialCorpusID_f.readlines()))
    error_doc = list()
    for i, docid in enumerate(tqdm(corpusID)):
        line = getcontent(docid, f)
        # if len(line) != 4:
        #     continue
        try:
            docid, url, title, body = line
        except:
            title, body = '', ''
            error_doc.append(i)
        # print(docid)
        # print(url)
        # print(title)
        # print(body)
        clean = unidecode(".".join([title.replace("\n", ""), body.replace("\n", "")])[:1000])
        context.append(clean)

with open("../model/error_doc.pk", 'wb') as f:
    pickle.dump(error_doc, f)
print(len(error_doc))
with open("../model/context.json", "w") as f:
	json.dump(context, f, indent=2)