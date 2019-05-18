#encoding:utf-8
import torch.nn as nn
from ..ernie.modeling import PreTrainedBertModel, BertModel

class ErnieFine(PreTrainedBertModel):
    def __init__(self,bertConfig,num_classes):
        super(ErnieFine ,self).__init__(bertConfig)
        self.bert = BertModel(bertConfig) # bert模型
        # 默认情况下，bert encoder模型所有的参数都是参与训练的，
        # 可以通过以下设置为将其设为不训练，只将classifier这一层进行反响传播，
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.dropout = nn.Dropout(bertConfig.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=bertConfig.hidden_size, out_features=num_classes)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None, output_all_encoded_layers=False):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=output_all_encoded_layers)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

