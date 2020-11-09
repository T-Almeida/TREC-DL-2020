import argparse
from collections import defaultdict

def load_prerank(file_name, top_k=100):
    prerank = defaultdict(list)
    min_rank = 999
    with open(file_name) as f:
        for line in f:
            elements = line.split(" ")
            if len(prerank[elements[0]])<top_k:
                prerank[elements[0]].append(line)
           
    
    return prerank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is program cuts the topk documetns for TREC formated files')
    
    parser.add_argument('file', type=str)
    parser.add_argument('-topk', dest='topk', help='Number of documents to cut', default=100)

    args = parser.parse_args()
    
    rank = load_prerank(args.file, top_k=args.topk)

    with open(args.file,"w", encoding="utf-8") as f:
        for q, doc_list in rank.items():
            for doc_line in doc_list:
                f.write(doc_line)
