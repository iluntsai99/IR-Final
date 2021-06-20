import csv

def getcontent(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    return line.rstrip()

# In the corpus tsv, each docid occurs at offset docoffset[docid]
docoffset = {}
with open("data/corpus/msmarco-docs-lookup.tsv", encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        docoffset[docid] = int(offset)


with open("data/corpus/msmarco-docs.tsv", encoding="utf8") as f:
    for docid in docoffset.keys():
        print("docid:", docid)
        print("docoffset[docid]:", docoffset[docid])
        print(getcontent(docid, f))
