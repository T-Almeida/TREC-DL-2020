import pandas as pd
import gc
import json
import tarfile
import codecs
import sys 
import time
import argparse
import os
import math

from collections import defaultdict
from mmnrm.text import TREC_goldstandard_transform, TREC_queries_transform
from mmnrm.evaluation import TREC_Evaluator
from mmnrm.dataset import TestCollectionV2, sentence_splitter_builderV2
from mmnrm.evaluation import BioASQ_Evaluator
from mmnrm.modelsv2 import deep_rank
from mmnrm.utils import set_random_seed, load_model_weights, load_model


import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import pickle

def load_TREC_queries(file):
    df = pd.read_csv(file, sep="\t")
    df.columns = ["id", "query"]
    topics = []
    for _,l in df.iterrows():
        topics.append({"query":str(l["query"]), "id":str(l["id"])})
        
    return TREC_queries_transform(topics, number_parameter="id", fn=lambda x:x["query"])

def load_TREC_qrels(q_rels_file):
    
    with open(q_rels_file) as f:
        goldstandard = defaultdict(list)

        for line in f:
            line = line.strip().split(" ")
            try:
                goldstandard[line[0]].append((line[2], line[3]))
            except :
                print(line)
            
    return TREC_goldstandard_transform(goldstandard)

def load_prerank(file_name, collection, top_k=1000):
    prerank = defaultdict(list)
    min_rank = 999
    with open(file_name) as f:
        for line in f:
            elements = line.split(" ")
            if elements[2] in collection and len(prerank[elements[0]])<top_k:
                article = collection[elements[2]]
                prerank[elements[0]].append({"id":elements[2], 
                                              "score":elements[4],
                                              "text":article["text"],
                                              "title":article["title"]})
            
    print(min_rank)    
    
    # create test collection base on the docs
    #docs_per_topic = [len(docs_topic) for docs_topic in prerank.values()]
    #print("average docs per topic", sum(docs_per_topic)/len(docs_per_topic), "min:",min(docs_per_topic),"max:",max(docs_per_topic))
    
    return prerank

def collection_iterator(file_name, f_map=None):
    return collection_iterator_fn(file_name=file_name, f_map=f_map)()

def collection_iterator_fn(file_name, f_map=None):
    
    reader = codecs.getreader("ascii")
    tar = tarfile.open(file_name)

    print("[CORPORA] Openning tar file", file_name)

    members = tar.getmembers()
    
    def generator():
        for m in members:
            print("[CORPORA] Openning tar file {}".format(m.name))
            f = tar.extractfile(m)
            articles = json.load(reader(f))
            if f_map is not None:
                articles = list(map(f_map, articles))
            yield articles
            f.close()
            del f
            gc.collect()
    return generator

def load_neural_model(path_to_weights):
    
    rank_model = load_model(path_to_weights, change_config={"return_snippets_score":True})
    tk = rank_model.tokenizer
    
    model_cfg = rank_model.savable_config["model"]
    
    max_input_query = model_cfg["max_q_length"]
    max_input_sentence = model_cfg["max_s_length"]
    max_s_per_q_term = model_cfg["max_s_per_q_term"]
    
    # redundant code... replace
    max_sentences_per_query = model_cfg["max_s_per_q_term"]

    pad_query = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                       maxlen=max_input_query,
                                                                                       dtype=dtype, 
                                                                                       padding='post', 
                                                                                       truncating='post', 
                                                                                       value=0)

    pad_sentences = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_input_sentence,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_docs = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))

    idf_from_id_token = lambda x: math.log(tk.document_count/tk.word_docs[tk.index_word[x]])

    train_sentence_generator, test_sentence_generator = sentence_splitter_builderV2(tk, 
                                                                                      max_sentence_size=max_input_sentence,
                                                                                      mode=4)


    def test_input_generator(data_generator):

        data_generator = test_sentence_generator(data_generator)

        for _id, query, docs in data_generator:

            # tokenization
            query_idf = list(map(lambda x: idf_from_id_token(x), query))

            tokenized_docs = []
            ids_docs = []
            offsets_docs = []
            for doc in docs:

                padded_doc = pad_docs(doc["text"], max_lim=max_input_query)
                for q in range(len(padded_doc)):
                    padded_doc[q] = pad_docs(padded_doc[q], max_lim=max_sentences_per_query)
                    padded_doc[q] = pad_sentences(padded_doc[q])
                tokenized_docs.append(padded_doc)
                ids_docs.append(doc["id"])
                offsets_docs.append(doc["offset"])

            # padding
            query = pad_query([query])[0]
            query = [query] * len(tokenized_docs)
            query_idf = pad_query([query_idf], dtype="float32")[0]
            query_idf = [query_idf] * len(tokenized_docs)

            yield _id, [np.array(query), np.array(tokenized_docs), np.array(query_idf)], ids_docs, offsets_docs
    
    return rank_model, test_input_generator

def rank(model, t_collection):

    generator_Y = t_collection.generator()
                
    q_scores = defaultdict(list)

    for i, _out in enumerate(generator_Y):
        query_id, Y, docs_info, offsets_docs = _out
        s_time = time.time()
        
        scores, q_sentence_attention = model.predict(Y)
        scores = scores[:,0].tolist()
            
        print("\rEvaluation {} | time {}".format(i, time.time()-s_time), end="\r")
        #q_scores[query_id].extend(list(zip(docs_ids,scores)))
        for i in range(len(docs_info)):
            q_scores[query_id].append((docs_info[i], scores[i], q_sentence_attention[i], offsets_docs[i]))

    # sort the rankings
    for query_id in q_scores.keys():
        q_scores[query_id].sort(key=lambda x:-x[1])
        q_scores[query_id] = q_scores[query_id][:100]
    
    return q_scores

def save_answers_to_file(answers, prefix = None, out_file = None):
    if out_file is not None:
        _name = out_file
    elif prefix is not None:
        _name = name.split(".")[0]+"_answer.txt"
    else:
        raise ValueError("set prefix or out_file")
        
    with open(_name,"w", encoding="utf-8") as f:
        for line in answers:
            f.write(line+"\n")
        
    return _name

if __name__ == "__main__":
    
    # argparsing
    parser = argparse.ArgumentParser(description='This is program to make neural reranking over for the trec DL')
    
    parser.add_argument('query_file', type=str)
    parser.add_argument('run_file', type=str)
    parser.add_argument('runtag', type=str)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('-out', dest="out", type=str, default=None)
    parser.add_argument('-topk', dest='topk', help='Number of documents returned by the bm25', default=100)

    args = parser.parse_args()
    
    query_file = args.query_file
    run_file = args.run_file
    runtag = args.runtag
    model_path = args.model_path
    out_file = args.out
    
    # load collection
    collection = sum([ articles for articles in collection_iterator("/backup/MS-MARCO/ms-marco-docs.tar.gz")],[])
    collection = {x["id"]:x for x in collection}
    
    # load queries
    queries = load_TREC_queries(query_file)
    # load baseline run
    baseline_run = load_prerank(run_file, collection)
    # build ranking set
    trec_evaluator = TREC_Evaluator("", '/backup/MS-MARCO/trec_eval-9.0.7/trec_eval')
    baseline_reranking = TestCollectionV2(queries, baseline_run, trec_evaluator)
    
    # load neural model
    rank_model, test_input_generator = load_neural_model(model_path)
    baseline_reranking.set_transform_inputs_fn(test_input_generator)
    
    q_scores = rank(rank_model, baseline_reranking)
    
    answers = []
    for q in queries:
        for i,doc_info in enumerate(q_scores[q["id"]]):
            answers.append("{} Q0 {} {} {} {}".format(q["id"],
                                             doc_info[0],
                                             i+1,
                                             doc_info[1],
                                             runtag))
    
    if out_file is None:
        out_file = os.path.basename(run_file)
    
    save_answers_to_file(answers, out_file = out_file)
    
    