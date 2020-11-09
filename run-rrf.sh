#!/bin/bash
echo "run1"
CUDA_VISIBLE_DEVICES="" python neural_ranking.py /backup/MS-MARCO/msmarco-test2020-queries.tsv /backup/MS-MARCO/bm25_baseline.txt BIT-run3 /backup/MS-MARCO/best_validation_models/expert-music-7_val_collection0_ndcg_cut_10 -out BIT.UA-run3-1.txt
echo "run2"
CUDA_VISIBLE_DEVICES="" python neural_ranking.py /backup/MS-MARCO/msmarco-test2020-queries.tsv /backup/MS-MARCO/bm25_baseline.txt BIT-run3 /backup/MS-MARCO/best_validation_models/expert-music-7_val_collection1_ndcg_cut_10 -out BIT.UA-run3-2.txt
echo "run3"
CUDA_VISIBLE_DEVICES="" python neural_ranking.py /backup/MS-MARCO/msmarco-test2020-queries.tsv /backup/MS-MARCO/bm25_baseline.txt BIT-run3 /backup/MS-MARCO/best_validation_models/expert-music-7_val_collection1_P_5 -out BIT.UA-run3-3.txt
echo "run4"
CUDA_VISIBLE_DEVICES="" python neural_ranking.py /backup/MS-MARCO/msmarco-test2020-queries.tsv /backup/MS-MARCO/bm25_baseline.txt BIT-run3 /backup/MS-MARCO/best_validation_models/expert-music-7_val_collection1_recall -out BIT.UA-run3-4.txt

