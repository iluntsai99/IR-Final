import csv
import io
import os
import sys
import numpy as np


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


dir_path = sys.argv[1]

# In the corpus tsv, each docid occurs at offset docoffset[docid]
docoffset = {}
with open(os.path.join(dir_path, "data/corpus/msmarco-docs-lookup.tsv"), encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        docoffset[docid] = int(offset)


# corpus count
with open(os.path.join(dir_path, "data/corpus/msmarco-docs.tsv"), encoding="utf8") as f, \
        open(os.path.join(dir_path, "data/partial/corpus/docIDs")) as partialCorpusID_f:
    vocab2idx = {}
    idx2vocab = {}
    count_corpus = {}

    corpus_reader = csv.reader(f, delimiter="\t")
    corpusID = list(map(lambda line: line.strip(),
                        partialCorpusID_f.readlines()))

    for docid in corpusID:
        line = getcontent(docid, f)
        if len(line) != 4:
            continue

        docid, url, title, body = line
        for term in title + body:
            if term not in vocab2idx:
                vocab2idx[term] = len(vocab2idx)
                idx2vocab[len(idx2vocab)] = term
                count_corpus[term] = 0
            count_corpus[term] += 1

    # save corpus vector
    corpus_vector = np.zeros((len(vocab2idx), 1))
    for i in range(len(vocab2idx)):
        corpus_vector[i][0] = count_corpus[idx2vocab[i]]

    np.save("corpus_vector.pickle", corpus_vector)
    print("save corpus vector done")
