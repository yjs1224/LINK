import torch
from torch.utils.data import TensorDataset
import csv
import os
import json


class InputExample(object):
    def __init__(self, sentence=None, label=None):
        self.sentence = sentence
        self.label = label


class SeqInputFeatures(object):
    """A single set of features of data for the ABSA task"""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class SteganalysisProcessor(object):
    def __init__(self, tokenizer, use_vocab=False, vocab_size=3000, data_dir=None, kn=0):
        self.tokenizer = tokenizer
        self.max_seq_len = 60
        self.label_list = [0, 1]
        self.num_labels = 2
        self.label2id = {}
        self.id2label = {}
        self.use_vocab = use_vocab
        self.vocab_size = vocab_size
        self.data_dir = data_dir
        if self.use_vocab:
            self.init_vocabulary(self.data_dir)
        else:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        for idx, label in enumerate(self.label_list):
            self.label2id[label] = idx
            self.id2label[idx] = label


    def get_examples(self, file_name):
        return self._create_examples(
            file_name=file_name
        )


    def get_train_examples(self, dir):
        return self.get_examples(os.path.join(dir, "train.csv"))


    def get_dev_examples(self, dir):
        return self.get_examples(os.path.join(dir, "val.csv"))


    def get_test_examples(self, dir):
        return self.get_examples(os.path.join(dir, "test.csv"))

    def init_vocabulary(self, dir):
        self.cover_file = os.path.join(dir, "cover.txt")
        self.stego_file = os.path.join(dir, "stego.txt")
        sentences = []
        with open(self.cover_file, "r", encoding='gb18030', errors='ignore') as f:
            sentences += f.read().split("\n")
        with open(self.stego_file, "r", encoding='gb18030', errors='ignore') as f:
            sentences += f.read().split("\n")
        words = []
        for sentence in sentences:
            sentence = sentence.lower().strip()
            words += sentence.split()
        from collections import Counter
        items = sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)
        self.word2id = {"[PAD]": 0, "[UNK]": 1}
        self.id2word = {0: "[PAD]", 1: "[UNK]"}
        id = 2
        for word, _ in items[:self.vocab_size - 2]:
            self.word2id[word] = id
            self.id2word[id] = word
            id += 1

    def _create_examples(self, file_name):
        examples = []
        file = file_name
        lines = csv.reader(open(file, 'r', encoding='utf-8'))
        for i, line in enumerate(lines):
            if i > 0:
                sentence = line[0].lower().strip()
                label_t = line[1].strip()
                if label_t == "0":
                    label = 0
                if label_t == "1":
                    label = 1
                examples.append(InputExample(sentence=sentence, label=label))
        return examples


    def convert_examples_to_features(self, examples):
        features = []
        for example in examples:
            if self.use_vocab:
                input_ids = [self.word2id.get(word, 1) for word in example.sentence.split()]
                attention_mask = [1] * len(input_ids)
                input_ids = input_ids[:self.max_seq_len] + [0] * (self.max_seq_len- len(input_ids))
                attention_mask = attention_mask[:self.max_seq_len] + [0] * (self.max_seq_len- len(attention_mask))
                token_type_ids = [0] * self.max_seq_len
            else:
                inputs = self.tokenizer.encode_plus(
                    example.sentence,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    padding='max_length',
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    truncation=True
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
            if example.label is not None:
                label_id = self.label2id[example.label]
            else:
                label_id = -1

            features.append(
                SeqInputFeatures(input_ids=input_ids,
                                 input_mask=attention_mask,
                                 segment_ids=token_type_ids,
                                 label_ids=label_id,))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
        # return features

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "steganalysis.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")

    def get_labels(self):
        return self.label_list
