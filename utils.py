import json

import numpy as np
import sklearn.metrics as metrics

class MyDict(dict):
    __setattr__ = dict.__setitem__
    # def __setattr__(self, key, value):
    #     try:
    #         self[key] = value
    #     except:
    #         raise  AttributeError(key)
    # __getattr__ = dict.__getitem__
    def __getattr__(self, item):
        try:
            return self[item]
        except:
            raise AttributeError(item)

class Config(object):
    def __init__(self, config_path):
        configs = json.load(open(config_path, "r", encoding="utf-8"))
        self.configs = self.dictobj2obj(configs)
        self.configs.state_dict = configs

    def dictobj2obj(self, dictobj):
        if not isinstance(dictobj, dict):
            return dictobj
        d = MyDict()
        for k, v in dictobj.items():
            d[k] = self.dictobj2obj(v)
        return d



    def get_configs(self):
        return self.configs


def compute_metrics(task_name, preds, labels, stego_label=1):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name in ["steganalysis", "graph_steganalysis", "ss", "gss","KE-steganalysis","KE-graph_steganalysis"]:
        return {"accuracy": metrics.accuracy_score(labels, preds),
                "macro_f1":metrics.f1_score(labels, preds, average="macro"),
                "precision":metrics.precision_score(labels, preds, pos_label=stego_label),
                "recall":metrics.recall_score(labels, preds, pos_label=stego_label),
                "f1_score":metrics.f1_score(labels, preds, pos_label=stego_label)}
    else:
        raise KeyError(task_name)

def bpw_jsonlines(filename, max_num=None):
    import jsonlines
    with open(filename, "r", encoding="utf-8") as f:
        bits = []
        tokens = []
        tokens_num = []
        counter = 0
        for text in jsonlines.Reader(f):
            bits += "".join(text["bits"][2:-1])
            tokens += text["tokens"][2:-1]
            tokens_num.append(len(text["tokens"][1:-1]))
            counter += 1
            if max_num is not None and counter >= max_num:
                break
        print("%s : %s %s %s" % (filename, str(len(bits) / len(tokens)), str(len(bits)), str(np.mean(tokens_num))+"_"+str(np.std(tokens_num))))

def calc_len(filename):
    with open(filename, "r") as f:
        sentences = f.readlines()
        words = []
        words_num = []
        for sentence in sentences:
            words += sentence.strip().split()
            words_num.append( len( sentence.strip().split()))
    print("{}: {} {}".format(filename, np.mean(words_num), np.std(words_num)))

def calc_K(filename):
    import jsonlines
    with open(filename, "r", encoding="utf-8") as f:
        concepts = []
        concepts_num = []
        uni_concepts = []
        counter = 0
        for sentence in jsonlines.Reader(f):
            counter +=1
            concepts+=sentence["qc"]
            concepts_num.append(len(sentence["qc"]))
        for c in concepts:
            if c not in uni_concepts:
                uni_concepts.append(c)
            # print()
            # words += sentence.strip().split()
    print("{}: {} {} {}".format(filename, np.mean(concepts_num), np.std(concepts_num), len(uni_concepts)))

def calc_concepts(filename):
    import jsonlines
    with open(filename, "r", encoding="utf-8") as f:
        concepts = []
        concepts_num = []
        uni_concepts = []
        counter = 0
        for sentence in jsonlines.Reader(f):
            counter +=1
            concepts+=sentence["concepts"]
            concepts_num.append(len(sentence["concepts"]))
        for c in concepts:
            if c not in uni_concepts:
                uni_concepts.append(c)
            # print()
            # words += sentence.strip().split()
    print("{}: {} {} {}".format(filename, np.mean(concepts_num), np.std(concepts_num), len(uni_concepts)))



if __name__ == '__main__':
    # for alg in ["hc5", "ac", "adg"]:
    #     for data in ["tweet", "news", "movie"]:
    #         # calc_len("/data/lastness/KE-dataset/{}/{}/cover.txt".format(data, alg))
    #         # calc_K("/data/lastness/KE-dataset/{}/{}/stego.concepts_nv.json".format(data, alg))
    #         calc_concepts("/data/lastness/KE-dataset/{}/{}/stego.2hops_100_triple.json".format(data, alg))
            # bpw_jsonlines("/data/lastness/KE-dataset/{}/{}/stegos-encoding.jsonl".format(data,alg))
    for data in ["tweet", "news", "movie"]:
        # calc_K("/data/lastness/KE-dataset/{}/hc5/cover.concepts_nv.json".format(data))
        calc_concepts("/data/lastness/KE-dataset/{}/hc5/cover.2hops_100_triple.json".format(data))#