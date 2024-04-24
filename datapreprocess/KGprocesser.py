import torch
import numpy as np
from torch.utils.data import TensorDataset
import os
import json
import copy
from random import sample

def load_kg_vocab(path, tokenizer):
    concept2id = {'<pad>': 1, '<unk>': 3}
    with open(path, 'r') as f:
        for line in f.readlines():
            vocab, _ = line.strip().split()
            tokenized_vocab = tokenizer.encode(
                ' '+vocab, add_special_tokens=False)
            if len(tokenized_vocab) > 1:
                print('not covered vocab: ', vocab, tokenized_vocab)
            if concept2id.get(vocab):
                print('duplicated vocab: ', vocab, tokenized_vocab)
            concept2id[vocab] = tokenized_vocab[0]
    return concept2id


class InputExample(object):
    def __init__(self, sentence, label_id, concepts, concepts_labels, distances, head_ids, tail_ids, relations,triple_labels):
        self.sentence = sentence
        self.label = label_id
        self.concepts = concepts
        self.concepts_labels = concepts_labels
        self.distances = distances
        self.head_ids = head_ids
        self.tail_ids = tail_ids
        self.relations = relations
        self.triple_labels = triple_labels


class SeqInputFeatures(object):
    def __init__(self, input_ids,input_mask, segment_ids, label_ids, concepts_ids, concepts_labels, distances, head_ids, tail_ids, relations_ids,triple_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.concepts_ids = concepts_ids
        self.concepts_labels = concepts_labels
        self.distances = distances
        self.head_ids = head_ids
        self.tail_ids = tail_ids
        self.relations_ids = relations_ids
        self.triple_labels = triple_labels


class KGSteganalysisProcessor(object):
    def __init__(self, tokenizer, split_ratios=[0.8,0.1,0.1]):
        assert (np.sum(split_ratios)==1), "sum of split ratio must equals to 1"
        self.tokenizer = tokenizer
        self.order = 1
        self.max_seq_len = 128
        self.label_list = ["cover", "stego"]
        self.num_labels = 2
        self.split_ratios = split_ratios
        self.max_concept_length = 300
        self.max_oracle_concept_length = 30
        self.max_triple_len = 600
        self.label2id = {}
        self.id2label = {}
        for idx, label in enumerate(self.label_list):
            self.label2id[label] = idx
            self.id2label[idx] = label


    def get_examples(self, dir, type=None):
        return self._create_examples(
            dir=dir, type=type
        )


    def get_train_examples(self, dir):
        return self.get_examples(dir, type="train")


    def get_dev_examples(self, dir):
        return self.get_examples(dir, type="val")


    def get_test_examples(self, dir):
        return self.get_examples(dir=dir, type=type)


    def _create_examples(self, dir, type=None):
        if type in ["train", "val", "test"]:
            if type == "train":
                return self.examples[:int(len(self.examples)*self.split_ratios[0])]
            if type == "val":
                return self.examples[int(len(self.examples) * self.split_ratios[0]):int(len(self.examples) * (self.split_ratios[0]+self.split_ratios[1]))]
            if type == "test":
                return self.examples[int(len(self.examples) * (self.split_ratios[0] + self.split_ratios[1])):]
        else:
            self.examples = []
            self.cover_file = os.path.join(dir, "cover.txt")
            self.stego_file = os.path.join(dir, "stego.txt")
            self.cover_kg_file = os.path.join(dir, "cover.kg.json")
            self.stego_kg_file = os.path.join(dir, "stego.kg.json")

            self.kg_vocab = os.path.join(dir, "kg_vocab.txt")
            self.concept2id = load_kg_vocab(self.kg_vocab, self.tokenizer)

            sentences = []
            labels = []
            concepts = []
            concepts_labels = []
            distances = []
            head_ids = []
            tail_ids = []
            relations = []
            triple_labels = []

            with open(self.cover_file, "r") as f:
                sentences += f.read().split("\n")
                labels += [self.label_list[0]] * len(sentences)

            with open(self.stego_file, "r") as f:
                sentences += f.read().split("\n")
                labels += [self.label_list[1]] * len(sentences)

            for kg_file in [self.cover_kg_file, self.stego_kg_file]:
                with open(kg_file, "r") as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        assert (len(line['concepts']) == len(line['labels'])), (
                        len(line['concepts']), len(line['labels']))
                        concepts.append(line['concepts'])
                        concepts_labels.append(line['labels'])
                        distances.append(line['distances'])
                        head_ids.append(line['head_ids'])
                        tail_ids.append(line['tail_ids'])
                        relations.append(line['relations'])
                        triple_labels.append(line['triple_labels'])
            for i in range(len(sentences)):
                self.examples.append(InputExample(sentence=sentences[i], label_id=labels[i], concepts=concepts[i],
                                                  concepts_labels=concepts_labels[i], distances=distances[i],
                                                  head_ids=head_ids[i],
                                                  tail_ids=tail_ids[i], relations=relations[i],
                                                  triple_labels=triple_labels[i]))
            return np.random.shuffle(self.examples)

    #
    def convert_examples_to_features(self, examples):
        '''
        only for bert tokenizer
        '''
        features = []
        for example in examples:
            inputs = self.tokenizer.encode_plus(
                example.sentence,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            if example.label is not None:
                label_id = self.label2id[example.label]
            else:
                label_id = -1


            _concept_ids, _concept_labels, _concept_distances = self.encode_concept(
                self.concept2id, example.concepts, example.concepts_labels, example.distances, self.max_concept_length, return_tensors="")
            relations = [x[0] for x in example.relations]
            _head_ids, _tail_ids, _relation_ids, _triple_labels = self.encode_triples(
                example.head_ids, example.tail_ids, relations, example.triple_labels, self.max_triple_len,return_tensors="")

            features.append(
            SeqInputFeatures(input_ids=input_ids,
                             input_mask=attention_mask,
                             segment_ids=token_type_ids,
                             label_ids=label_id,
                             concepts_ids = _concept_ids,
                             concepts_labels= _concept_labels,
                             distances=_concept_distances,
                             head_ids=_head_ids,
                             tail_ids=_tail_ids,
                             relations_ids=_relation_ids,
                             triple_labels=_triple_labels))



        oracle_concept_ids, oracle_concept_mask = None, None
        # if self.train:
        #     oracle_concept_ids, oracle_concept_mask, _ = self.encode_oracle_concept(
        #         self.concept2id, _concept, _cpt_label, _dist, max_oracle_concept_length)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_concepts_ids = torch.tensor([f.concepts_ids for f in features], dtype=torch.long)
        all_concepts_labels = torch.tensor([f.concepts_labels for f in features], dtype=torch.long)
        all_distances = torch.tensor([f.distances for f in features], dtype=torch.long)
        all_head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        all_tail_ids = torch.tensor([f.tail_ids for f in features], dtype=torch.long)
        all_relations_ids = torch.tensor([f.relations_ids for f in features], dtype=torch.long)
        all_triple_labels = torch.tensor([f.triple_labels for f in features], dtype=torch.long)


        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_concepts_ids,
                                all_concepts_labels,all_distances,all_head_ids,all_tail_ids,all_relations_ids,all_triple_labels)
        return dataset

    def encode_concept(self, tokenizer, concepts, labels, distances, max_len, return_tensors="pt"):

        concept_ids = []
        for c in concepts:
            concept_ids.append(tokenizer[c])
        if len(concept_ids) >= max_len - 1:
            concept_ids = [0] + concept_ids[:max_len - 2] + [2]
            labels = [0] + labels[:max_len - 2] + [0]
            distances = [0] + distances[:max_len - 2] + [0]
        if len(concept_ids) < max_len - 1:
            concept_ids = [0] + concept_ids + [2]
            labels = [0] + labels + [0]
            distances = [0] + distances + [0]
        while len(concept_ids) < max_len:
            concept_ids.append(1)  # PAD_ID = 1
            labels.append(-1)
            distances.append(0)
        if return_tensors == 'pt':
            return torch.tensor(concept_ids), torch.tensor(labels), torch.tensor(distances)
        else:
            return concept_ids, labels, distances

    def encode_oracle_concept(self, tokenizer, concepts, labels, distances, max_len, return_tensors="pt"):

        # drop zero label
        _c, _l, _d = [], [], []
        for c, l, d in zip(concepts, labels, distances):
            if l == 1:
                _c.append(c)
                _l.append(1)
                _d.append(d)

        if len(concepts) > 20:
            sampled_concepts = sample(concepts, 20)
            _c += sampled_concepts
            _l += [1] * len(sampled_concepts)
            _d += [2] * len(sampled_concepts)

        concepts = _c
        labels = _l
        distances = _d

        concept_ids = []
        for c in concepts:
            concept_ids.append(tokenizer[c])
        if len(concept_ids) >= max_len - 1:
            concept_ids = [0] + concept_ids[:max_len - 2] + [2]
            labels = [1] + labels[:max_len - 2] + [0]
            distances = [0] + distances[:max_len - 2] + [0]
        if len(concept_ids) < max_len - 1:
            concept_ids = [0] + concept_ids + [2]
            labels = [1] + labels + [0]
            distances = [0] + distances + [0]
        while len(concept_ids) < max_len:
            concept_ids.append(1)  # PAD_ID = 1
            labels.append(0)
            distances.append(0)

        if return_tensors == 'pt':
            return torch.tensor(concept_ids), torch.tensor(labels), torch.tensor(distances)
        else:
            return concept_ids, labels, distances

    def encode_triples(self, head_ids, tail_ids, relation_ids, triple_labels, max_len, return_tensors='pt'):
        if len(head_ids) > max_len:
            head_ids = head_ids[:max_len]
            tail_ids = tail_ids[:max_len]
            relation_ids = relation_ids[:max_len]
            triple_labels = triple_labels[:max_len]
        while len(head_ids) < max_len:
            head_ids.append(0)
            tail_ids.append(0)
            relation_ids.append(0)
            triple_labels.append(-1)
        if return_tensors == 'pt':
            return torch.tensor(head_ids), torch.tensor(tail_ids), \
                   torch.tensor(relation_ids), torch.tensor(triple_labels)
        else:
            return head_ids, tail_ids, relation_ids, triple_labels

    def get_labels(self):
        return self.label_list


if __name__ == '__main__':
    '''
    function testing
    '''

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    processor = KGSteganalysisProcessor(tokenizer)
    processor.get_examples("./dataset/ac")
    data = processor.convert_examples_to_features(processor.get_train_examples("./dataset/ac"))
    print()
    # get_examples = processor.get_train_examples
    # _, examples = get_examples('../data')
    # processor.get_test_examples("../data")
    # processor.get_dev_examples("../data")