import numpy as np
import os
from transformers import AutoTokenizer
import pickle as pkl

# def convert(tokenizer, dim=300, input_path = "/data/lastness/glove/glove.840B.300d.txt", save_path="./vocab_embedding"):
#     # save_path = input_path+"__"+tokenizer.name_or_path
#     embedding_mat = np.zeros((tokenizer.vocab_size, dim))
#     vocab = tokenizer.get_vocab()
#     fin = open(input_path, 'r', encoding='utf-8',newline='\n',errors='ignore')
#     count = 0
#     for line in fin:
#         tokens = line.rstrip().split()
#         if tokens[0] in vocab.keys():
#             try:
#                 embedding_mat[vocab[tokens[0]]] = np.asarray(tokens[1:],dtype="float32")
#                 count += 1
#             except:
#                 continue
#     pkl.dump({"embedding":embedding_mat,"vocab":vocab}, open(save_path,"wb"))


def convert(vocab, dim=300, input_path = "/data/lastness/glove/glove.840B.300d.txt", save_path="./test_vocab_embedding"):
    # save_path = input_path+"__"+tokenizer.name_or_path
    embedding_mat = np.zeros((len(list(vocab.keys())), dim))
    fin = open(input_path, 'r', encoding='utf-8',newline='\n',errors='ignore')
    count = 0
    for line in fin:
        tokens = line.rstrip().split()
        if tokens[0] in vocab.keys():
            try:
                embedding_mat[vocab[tokens[0]]] = np.asarray(tokens[1:],dtype="float32")
                count += 1
            except:
                continue
    pkl.dump({"embedding":embedding_mat,"vocab":vocab}, open(save_path,"wb"))

def load_kg_vocab(paths):
    concept2id = {'<pad>': 0, '<unk>': 1, "<s>":2, "</s>":3}
    id2concept = {0: '<pad>', 1: '<unk>', 2:"<s>", 3:"</s>"}
    # count = 0
    id = 4
    for path in paths:
        with open(path, 'r') as f:
            for line in f.readlines():
                vocab, _ = line.strip().split()
                if concept2id.get(vocab, None) is None:
                    concept2id[vocab] = id
                    id2concept[id] = vocab
                    id +=1
    return concept2id, id2concept


if __name__ == '__main__':
     paths = []
     for data in ["commonsense", "news", "tweet","movie",]:
         for alg in ["ac", "hc5", "adg"]:
             paths.append(os.path.join("/data/lastness/KE-dataset", data, alg, "kg_vocab.txt"))
     concept2id, id2concept = load_kg_vocab(paths)
     import json
     json.dump(concept2id, open("concept2id.json", "w"))
     json.dump(id2concept, open("id2concept.json", "w"))
     convert(concept2id, save_path="840B.300d.concept_embedding.pkl")


    #paths = []
    #for data in ["news"]:
    #    for alg in ["ac"]:
    #        paths.append(os.path.join("/data/lastness/KE-dataset", data, alg, "kg_vocab.txt"))
    #concept2id, id2concept = load_kg_vocab(paths)
    #import json
    #json.dump(concept2id, open("news_concept2id.json", "w"))
    #json.dump(id2concept, open("news_id2concept.json", "w"))
    #convert(concept2id, save_path="news_840B.300d.concept_embedding.pkl")
    ## tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ## tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ## convert(tokenizer.get_vocab)