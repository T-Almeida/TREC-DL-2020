from mmnrm.utils import set_random_seed
from nir.embeddings import FastText, Word2Vec

set_random_seed()

import io
from nir.tokenizers import Regex, BioCleanTokenizer, BioCleanTokenizer2
import numpy as np
import math
import os 
import json

import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.dataset import TrainCollectionV2, TestCollectionV2, sentence_splitter_builderV2, TrainPairwiseCollection
from mmnrm.modelsv2 import deep_rank
from mmnrm.callbacks import TriangularLR, WandBValidationLogger, LearningRateScheduler
from mmnrm.training import PairwiseTraining, pairwise_cross_entropy
from mmnrm.utils import merge_dicts, load_model

def main():
    
    
    min_freq = 5
    mun_itter = 15
    emb_size = 200
    
    use_triangularLR = False
    use_step_decay = False
    
    LR = 0.01
    base_lr = 0.001
    max_lr = 0.01
    epoch=32
    
    train_batch_size=32
    type_split_mode=4
    use_query_sw = False
    use_docs_sw = False
       
    
    cache_folder = "/backup/MS-MARCO"


    print("build new model")

    # build config
    tokenizer_class = Regex
    tokenizer_cfg = {"class":tokenizer_class,
                    "attr":{
                        "cache_folder":os.path.join(cache_folder, "tokenizers"),
                        "prefix_name":"ms-marco-docs"
                    },
                    "min_freq":min_freq}

    embeddind_class = Word2Vec
    embedding_cfg = {
        "class":embeddind_class,
        "attr":{
            "cache_folder":os.path.join(cache_folder, "embeddings"),
            "prefix_name":"ms-marco-docs",
            "path":os.path.join(cache_folder, "word2vec/msmarco2020_gensim_iter_15_freq5_200_Regex_word2vec.bin"),
        }
    }

    model_cfg = {
        "max_q_length": 30,
        "max_s_per_q_term": 5,
        "max_s_length": 30,
        "filters":16,
        "kernel_size":[3,3],
        "aggregation_size":20,
        "q_term_weight_mode":0,
        "aggregation_mode":3,
        "extraction_mode":2,
        "score_mode":1,
        "train_context_emgeddings": False,
        "activation": "mish"
    }

    cfg = {"model":model_cfg, "tokenizer": tokenizer_cfg, "embedding": embedding_cfg}


    K.clear_session()

    rank_model = deep_rank(**cfg)
        # inspect the model
    #rank_model.summary()  
    
    tk = rank_model.tokenizer
    
    ###########################
    ## Input transformations ##
    ###########################

    pad_query = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                       maxlen=model_cfg['max_q_length'],
                                                                                       dtype=dtype, 
                                                                                       padding='post', 
                                                                                       truncating='post', 
                                                                                       value=0)

    pad_sentences = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=model_cfg['max_s_length'],
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_docs = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))

    idf_from_id_token = lambda x: math.log(tk.document_count/tk.word_docs[tk.index_word[x]])

    
    # use stop words?
    if use_query_sw:
        with open("stop_words.json", "r") as f:
            query_sw = set(tk.texts_to_sequences([" ".join(json.load(f))])[0])
            print(query_sw)
    else:
        query_sw = None
            
    if use_docs_sw:
        with open("stop_words.json", "r") as f:
            docs_sw = set(tk.texts_to_sequences([" ".join(json.load(f))])[0])
    else:
        docs_sw = None
       
    
    
    train_sentence_generator, test_sentence_generator = sentence_splitter_builderV2(tk, 
                                                                                      max_sentence_size=model_cfg['max_s_length'],
                                                                                      mode=type_split_mode,
                                                                                      queries_sw=query_sw,
                                                                                      docs_sw=docs_sw)

    def training_input_generator(data_generator):

        data_generator = train_sentence_generator(data_generator)

        while True:
            query, pos_docs, pos_extra_features, neg_docs, neg_extra_features = next(data_generator)

            query_idf = np.array([list(map(lambda x: idf_from_id_token(x), t_q)) for t_q in query])

            # padding
            for i in range(len(pos_docs)):
                pos_docs[i] = pad_docs(pos_docs[i], max_lim=model_cfg['max_q_length'])
                neg_docs[i] = pad_docs(neg_docs[i], max_lim=model_cfg['max_q_length'])

                for q in range(len(pos_docs[i])):

                    pos_docs[i][q] = pad_docs(pos_docs[i][q], max_lim=model_cfg['max_s_per_q_term'])
                    neg_docs[i][q] = pad_docs(neg_docs[i][q], max_lim=model_cfg['max_s_per_q_term'])

                    pos_docs[i][q] = pad_sentences(pos_docs[i][q])
                    neg_docs[i][q] = pad_sentences(neg_docs[i][q])

            query = pad_query(query)
            query_idf = pad_query(query_idf, dtype="float32")
            
            yield [query, np.array(pos_docs), query_idf], [query,  np.array(neg_docs), query_idf]

    def test_input_generator(data_generator):

        data_generator = test_sentence_generator(data_generator)

        for _id, query, docs in data_generator:

            # tokenization
            query_idf = list(map(lambda x: idf_from_id_token(x), query))

            tokenized_docs = []
            ids_docs = []
            for doc in docs:

                padded_doc = pad_docs(doc["text"], max_lim=model_cfg['max_q_length'])
                for q in range(len(padded_doc)):
                    padded_doc[q] = pad_docs(padded_doc[q], max_lim=model_cfg['max_s_per_q_term'])
                    padded_doc[q] = pad_sentences(padded_doc[q])
                tokenized_docs.append(padded_doc)
                ids_docs.append(doc["id"])

            # padding
            query = pad_query([query])[0]
            query = [query] * len(tokenized_docs)
            query_idf = pad_query([query_idf], dtype="float32")[0]
            query_idf = [query_idf] * len(tokenized_docs)

            yield _id, [np.array(query), np.array(tokenized_docs), np.array(query_idf)], ids_docs

    # Get the training data
    training_data_used = "/backup/MS-MARCO/preprocess/train_collection_k100"


    train_collection = TrainCollectionV2\
                            .load("/backup/MS-MARCO/preprocess/train_collection_k100")\
                            .batch_size(train_batch_size)\
                            .set_transform_inputs_fn(training_input_generator)

    dev = TestCollectionV2\
                            .load("/backup/MS-MARCO/preprocess/dev_collection_k100")\
                            .batch_size(100)\
                            .set_transform_inputs_fn(test_input_generator)\
                            .set_name("dev set")
    
    val = TestCollectionV2\
                            .load("/backup/MS-MARCO/preprocess/val2019_collection_k100")\
                            .batch_size(100)\
                            .set_transform_inputs_fn(test_input_generator)\
                            .set_name("2019 val set")
    

    
    notes = ""
    
    if use_triangularLR:
        _lr = "tlr_"+str(base_lr)+"_"+str(max_lr)
    elif use_step_decay:
        _lr = "step_decay_"+str(LR)
    else:
        _lr = LR
    
    
    
    wandb_config = {"optimizer": "adam",
                     "lr":_lr,
                     "loss":"pairwise_cross_entropy",
                     "train_batch_size":train_batch_size,
                     "epoch":epoch,
                     "type_split_mode": type_split_mode,
                     "name": "deeprank model",
                     "query_sw":use_query_sw,
                     "docs_sw":use_docs_sw,
                     "training_dataset": training_data_used,
                     "notes": notes
                     }
    
    wandb_config = merge_dicts(wandb_config, model_cfg)

    
    project_name = "trec-dl"
    
    ## config wandb
    wandb_args = {"project": project_name, "config": wandb_config}

    # define callbacks
    
    tlr = TriangularLR(base_lr=base_lr, max_lr=max_lr,)
    
    wandb_val_logger = WandBValidationLogger(wandb_args=wandb_args,
                                             steps_per_epoch=train_collection.get_steps(),
                                             validation_collection=[dev, 
                                                                    val
                                                                    ],
                                             test_collection=None,
                                             path_store = "/backup/MS-MARCO/best_validation_models",
                                             output_metrics=[#"map@10",
                                                             #"recall@10",
                                                             "recall_10", 
                                                             #"map_cut_1000",
                                                             "ndcg_cut_10",
                                                             "P_5"])
    
    step_decay_lr = LearningRateScheduler(initial_learning_rate=LR)
    
    train_callbacks = [wandb_val_logger]
    
    if use_triangularLR:
        train_callbacks.append(tlr)
        
    if use_step_decay:
        train_callbacks.append(step_decay_lr)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
    
    @tf.function
    def clip_grads(grads):
        gradients, _ = tf.clip_by_global_norm(grads, 5.0)
        return gradients

    @tf.function
    def custom_loss(positive_score, negative_score, *args):
        positive_exp = K.exp(positive_score)
        loss = K.mean(-K.log(positive_exp/(positive_exp+K.exp(negative_score))))
        
        loss = K.abs(loss - 0.2) + 0.2
        
        return loss
    
    train = PairwiseTraining(model=rank_model,
                             train_collection=train_collection,
                             loss=pairwise_cross_entropy,
                             grads_callback=clip_grads,
                             optimizer=optimizer,
                             callbacks=train_callbacks)

                              
    train.train(epoch, draw_graph=False)                           
                              
if __name__ == "__main__":
    main()