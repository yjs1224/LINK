import torch
from torch import nn
from transformers import BertPreTrainedModel,BertModel


class TC_base(nn.Module):
    def __init__(self, in_features, class_num, dropout_rate,):
        super(TC_base, self).__init__()
        self.in_features = in_features
        self.dropout_prob = dropout_rate
        self.num_labels = class_num  #
        self.dropout = nn.Dropout(self.dropout_prob)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.normalization = nn.BatchNorm1d(self.in_features)
        self.classifier = nn.Linear(self.in_features, self.num_labels)

    def forward(self,features):
        clf_input = self.pool(features.permute(0,2,1)).squeeze()
        clf_input = self.normalization(clf_input)
        logits = self.classifier(clf_input)
        return logits

    def extra_repr(self) -> str:
        return 'features {}->{},'.format(
            self.in_features, self.class_num
        )


class TC(nn.Module):
    def __init__(self, vocab_size, embed_size, class_num, dropout_rate, criteration="CrossEntropyLoss"):
        super(TC,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size= embed_size
        self.dropout_prob = dropout_rate
        self.num_labels = class_num

        self.embedding = nn.Embedding(vocab_size, self.embed_size,)

        self.classifier = TC_base(self.embed_size, self.num_labels,self.dropout_prob)

        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()


    def forward(self,input_ids, labels, attention_mask=None,token_type_ids=None):
        clf_input = self.embedding(input_ids.long())
        logits = self.classifier(clf_input)
        loss = self.criteration(logits, labels)
        return loss, logits


