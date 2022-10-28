import torch
from torch import nn
from transformers import BertPreTrainedModel,BertModel,DistilBertPreTrainedModel,DistilBertModel


class TC_base(nn.Module):
	def __init__(self, in_features, filter_num, filter_size, class_num, dropout_rate,):
		super(TC_base, self).__init__()

		self.cnn_list = nn.ModuleList()
		for size in filter_size:
			self.cnn_list.append(nn.Conv1d(in_features, filter_num, size))
		self.relu = nn.ReLU()
		self.max_pool = nn.AdaptiveMaxPool1d(1)
		self.dropout = nn.Dropout(dropout_rate)
		self.normalization = nn.BatchNorm1d(filter_num * len(filter_size))
		self.output_layer = nn.Linear(filter_num * len(filter_size), class_num)

		self.in_features = in_features
		self.class_num = class_num

	def forward(self, features):
		#print(self.in_features)
		_ = features.permute(0, 2, 1)
		result = []
		for self.cnn in self.cnn_list:
			__ = self.cnn(_)
			__ = self.relu(__)
			__ = self.max_pool(__)
			result.append(__.squeeze(dim=2))

		_ = torch.cat(result, dim=1)
		_ = self.dropout(_)
		_ = self.normalization(_)
		_ = self.output_layer(_)
		return _

	def extra_repr(self) -> str:
		return 'features {}->{},'.format(
			self.in_features, self.class_num
		)


class TC(nn.Module):
	def __init__(self, vocab_size, embed_size, filter_num, filter_size, class_num, dropout_rate, criteration="CrossEntropyLoss",):
		super(TC, self).__init__()
		self.vocab_size = vocab_size
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.classifier = TC_base(embed_size,filter_num,filter_size,class_num,dropout_rate)
		if criteration == "CrossEntropyLoss":
			self.criteration = nn.CrossEntropyLoss()
		else:
			# default loss
			self.criteration = nn.CrossEntropyLoss()


	def forward(self, input_ids, labels,attention_mask=None,token_type_ids=None):
		clf_input = self.embedding(input_ids.long())
		logits = self.classifier(clf_input)
		loss = self.criteration(logits, labels)
		return loss, logits


