import os
import argparse
import numpy as np
import scipy.sparse as sp
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity

class Query:
    def __init__(self, num: int, title: str, concepts: str, content: str):
        self.num = num
        self.title = title
        self.concepts = concepts
        self.content = content

    def target_tfidf(self, target: str, vocab2idx: dict, term2idx: dict, weight: int):
        # unigram
        for vocab in target:
            if vocab in vocab2idx:
                term = (vocab2idx[vocab], -1)
                if term in term2idx:
                    self.tf[term2idx[term]] += weight
                    self.idf[term2idx[term]] = 1
        # bigram
        for idx in range(len(target) - 1):
            if (target[idx] in vocab2idx) and (target[idx + 1] in vocab2idx):
                vocab1 = vocab2idx[target[idx]]
                vocab2 = vocab2idx[target[idx + 1]]
                term = (vocab1, vocab2)
                if term in term2idx:
                    self.tf[term2idx[term]] += (weight * 2)
                    self.idf[term2idx[term]] = 1
        return

    def calculate_tfidf(self, vocab2idx: dict, term2idx: dict, TITLE: int=5, CONCEPTS: int=2, CONTENT: int=1):
        nbr_term = len(term2idx)
        self.tf = np.zeros(nbr_term)
        self.idf = np.zeros(nbr_term)

        self.target_tfidf(self.title, vocab2idx, term2idx, TITLE)
        self.target_tfidf(self.concepts, vocab2idx, term2idx, CONCEPTS)
        self.target_tfidf(self.content, vocab2idx, term2idx, CONTENT)

        return self.tf, self.idf

def read_model(model_dir: str, ntcir_dir: str):
    print('Reading model directory...', end=' ', flush=True)

    vocab2idx = dict()
    idx2vocab = list()
    f = open(os.path.join(model_dir, 'vocab.all'), 'r')
    for idx, vocab in enumerate(f):
        vocab2idx[vocab.strip()] = idx
        idx2vocab.append(vocab.strip())

    term2idx = dict()
    idx2term = list()
    f = open(os.path.join(model_dir, 'inverted-file'), 'r')
    for idx, line in enumerate(f):
        vocab1, vocab2, N = [int(i) for i in line.split()]
        term2idx[(vocab1, vocab2)] = idx
        idx2term.append((vocab1, vocab2))
        for i in range(N):
            f.readline()

    doc2idx = dict()
    idx2doc = list()
    f = open(os.path.join(model_dir, 'file-list'), 'r')
    for idx, file_nm in enumerate(f):
        fp = open(os.path.join(ntcir_dir, file_nm.strip()), 'r', encoding='utf-8')
        xml = ET.parse(fp).getroot()
        doc_nm = xml.find('doc').find('id').text.lower()
        doc2idx[doc_nm] = idx
        idx2doc.append(doc_nm)

    print('(done)', flush=True)
    return vocab2idx, idx2doc, term2idx

def okapi(tf: csr_matrix, idf: np.ndarray, k: np.float64=1.5, b: np.float64=0.75):
    docLen = tf.sum(axis=1)
    avgLen = docLen.mean()

    vec = tf.copy().tocoo()
    var1 = vec * (k + 1)
    var2 = k * (1 - b + b * docLen/avgLen)
    
    vec.data += np.array(var2[vec.row]).reshape(vec.data.shape)
    vec.data = var1.data / vec.data
    vec.data *= idf[vec.col]
    vec = vec.tocsr()
    return vec

def doc2vec(model_dir: str, nbr_term: int, nbr_doc: int):
    print('Converting documents to vectors...', flush=True)
    
    data, row, col, doc_idf = [], [], [], []
    f = open(os.path.join(model_dir, 'inverted-file'), 'r')
    for idx, line in enumerate(tqdm(f, total=nbr_term)):
        vocab1, vocab2, N = [int(i) for i in line.split()]
        for i in range(N):
            docID, cnt = [int(j) for j in f.readline().split()]
            data.append(cnt)
            row.append(docID)
            col.append(idx) 
        doc_idf.append(np.log((nbr_doc + 1) / (N + 1)) + 1)

    doc_tf = csr_matrix((data, (row, col)), shape=(nbr_doc, nbr_term), dtype=np.float64)
    return doc_tf, np.array(doc_idf)

def query2vec(query_file: str, vocab2idx: dict, term2idx: dict):
    print('Converting querys to vectors...', flush=True)
    query_tf, query_idf, query_list = [], [], []

    f = open(query_file, 'r', encoding='utf-8')
    xml = ET.parse(f).getroot()
    topics = xml.findall('topic')
    to_replace = ['查詢', '相關文件內容']

    for topic in tqdm(topics):
        num = int(topic.find('number').text.split('TopicZH')[-1])
        title = topic.find('title').text.strip()
        question = topic.find('question').text.strip()
        narrative = topic.find('narrative').text.strip()
        concepts = topic.find('concepts').text.strip()
        
        content = ''.join([question, narrative])
        for target in to_replace:
            content = content.replace(target, '')
        
        query = Query(num, title, concepts, content)
        query_list.append(query)

        tf, idf = query.calculate_tfidf(vocab2idx, term2idx)
        query_tf.append(tf)
        query_idf.append(idf)

    query_tf = np.stack(query_tf)
    query_tf = csr_matrix(query_tf)

    query_idf = np.stack(query_idf).sum(axis=0)
    query_idf = np.log((len(topics) + 1) / (query_idf + 1)) + 1

    return query_tf, query_idf, query_list

def predict(doc_vec: csr_matrix, query_vec: csr_matrix, query_list: list, idx2doc: list, feedback: bool, topk: int=100):
    cos_sim = (query_vec * doc_vec.T).toarray()
    # cos_sim = cosine_similarity(query_vec, doc_vec)
    rankings = np.flip(cos_sim.argsort(axis=1), axis=1)

    if feedback:
        print('Start Rocchio relevance feedback...', flush=True)
        # Rocchio relevance feedback parameters
        rel_count, nrel_count = 20, 1
        alpha, beta, gamma = 1, 0.75, 0
        
        iters = 5
        for _ in range(iters):
            # Update query vectors with Rocchio algorithm
            rel_vecs = csr_matrix((0, query_vec.shape[1]))
            for docID_r in rankings[:, :rel_count]:
                rel_vec = csr_matrix((0, query_vec.shape[1]))
                for docID in docID_r:
                    rel_vec = sp.vstack((rel_vec, doc_vec.getrow(docID)))
                rel_vecs = sp.vstack((rel_vecs, rel_vec.mean(axis=0)))
            
            nrel_vecs = csr_matrix((0, query_vec.shape[1]))
            for docID_nr in rankings[:, -nrel_count:]:
                nrel_vec = csr_matrix((0, query_vec.shape[1]))
                for docID in docID_nr:
                    nrel_vec = sp.vstack((nrel_vec, doc_vec.getrow(docID)))
                nrel_vecs = sp.vstack((nrel_vecs, nrel_vec.mean(axis=0)))

            query_vec = alpha * query_vec + beta * rel_vecs - gamma * nrel_vecs
            
            # Rerank documents based on cosine similarity
            cos_sim = (query_vec * doc_vec.T).toarray()
            # cos_sim = cosine_similarity(query_vec, doc_vec)
            rankings = np.flip(cos_sim.argsort(axis=1), axis=1)
    
    prediction = []
    for ranking in rankings:
        rank_doc = []
        for idx in range(topk):
            rank_doc.append(idx2doc[ranking[idx]])
        prediction.append(rank_doc)
    
    return prediction

if __name__ == '__main__':
    # Usage:  python3 main.py -r -i ../final_data_hw1/queries/query-train.xml -o output_train.csv -m ../final_data_hw1/model -d ../final_data_hw1/CIRB010
    # Usage:  python3 main.py -r -i ../final_data_hw1/queries/query-train.xml -o output_train.csv -m ../final_data_hw1/model -d ../final_data_hw1/
    # Usage:  python3 main.py -r -i ../final_data_hw1/queries/query-test.xml -o output_test.csv -m ../final_data_hw1/model -d ../final_data_hw1/CIRB010
    # Usage:  python3 main.py -r -i ../final_data_hw1/queries/query-test.xml -o output_test.csv -m ../final_data_hw1/model -d ../final_data_hw1/
    parser = argparse.ArgumentParser(description='Vector Space Model.')
    parser.add_argument('-r', action='store_true', default=False, dest='feedback', help='Use relevance feedback or not.')
    parser.add_argument('-i', type=str, dest='query_file', help='Path to query file.')
    parser.add_argument('-o', type=str, dest='output_file', help='Path to output ranked file.')
    parser.add_argument('-m', type=str, dest='model_dir', help='Path to model directory.')
    parser.add_argument('-d', type=str, dest='ntcir_dir', help='Path to NTCIR directory.')
    args = parser.parse_args()

    vocab2idx, idx2doc, term2idx = read_model(args.model_dir, args.ntcir_dir)
    nbr_doc, nbr_term = len(idx2doc), len(term2idx)
    print('Shape of doc-term matrix: ({}, {})'.format(nbr_doc, nbr_term), flush=True)

    doc_tf, doc_idf = doc2vec(args.model_dir, nbr_term, nbr_doc)
    query_tf, query_idf, query_list = query2vec(args.query_file, vocab2idx, term2idx)
    
    doc_vec = okapi(doc_tf, doc_idf)
    query_vec = okapi(query_tf, query_idf)

    pred = predict(doc_vec, query_vec, query_list, idx2doc, args.feedback)
    
    print(f'Writing results to output file {args.output_file}...', end=' ', flush=True)
    with open(args.output_file, 'w') as f:
        print('query_id,retrieved_docs', file=f)
        for idx, query in enumerate(query_list):
            print('{},{}'.format(query.num, ' '.join(pred[idx])), file=f)
    print('(done)', flush=True)