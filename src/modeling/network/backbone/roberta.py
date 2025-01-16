import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from src.utils.registry import REGISTRY


@REGISTRY.register('roberta_lstm')
class RobertaLSTMModel(nn.Module):
    def __init__(self, num_classes=5, pretrained_model="roberta-base", max_seq_length=512):
        super(RobertaLSTMModel, self).__init__()

        # 配置RoBERTa模型
        config = RobertaConfig.from_pretrained(pretrained_model)
        config.num_labels = num_classes
        # config.max_position_embeddings = max_seq_length

        # # lstm
        # self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)
        # self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=256, num_layers=2, batch_first=True,
        #                     bidirectional=True)
        # self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Linear(256 * 2, num_classes)

        # Roberta_Vanilla代码
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)
        self.dropout = torch.nn.Dropout(0.1)  # 设置随机失活层
        self.fc1 = torch.nn.Linear(768, 768)  # 设置全连接层1，输入大小为base_model_output_size，输出大小为768
        self.fc2 = torch.nn.Linear(768, num_classes)  # 设置全连接层2，输入大小为768，输出大小为n_clases



    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)


        # # lstm
        # # 获取RoBERTa的输出并传递给LSTM
        # sequence_output = outputs[0]
        # self.lstm.flatten_parameters()
        # lstm_output, _ = self.lstm(sequence_output)
        # lstm_output = lstm_output[:, -1, :]
        #
        # # 分类
        # lstm_output = self.dropout(lstm_output)
        # logits = self.classifier(lstm_output)


        x = outputs[0]  # 获取outputs列表的第一个元素，赋值给变量x
        x = x[:, 0, :]  # 对x取指定的维度，只保留第0维和第2维，将结果重新赋值给x
        x = self.dropout(x)  # 对x进行dropout操作，将结果赋值给x
        x = torch.tanh(self.fc1(x))  # 使用self.fc1对x进行线性变换，然后应用tanh激活函数，将结果赋值给x
        x = self.dropout(x)  # 对x进行dropout操作，将结果赋值给x
        logits = self.fc2(x)  # 使用self.fc2对x进行线性变换，将结果赋值给logits


        return logits

    def get_layer_groups(self):
        # 将RoBERTa模型的参数视为特征提取器的一部分
        feature_extractor_layers = [elem[1] for elem in self.roberta.named_parameters()]

        # # 将LSTM和分类器的参数视为分类器的一部分
        # classifier_layers = [elem[1] for elem in self.lstm.named_parameters()] + \
        #                     [elem[1] for elem in self.classifier.named_parameters()]

        # v
        classifier_layers = [elem[1] for elem in self.dropout.named_parameters()] + \
                            [elem[1] for elem in self.fc1.named_parameters()] + \
                            [elem[1] for elem in self.fc2.named_parameters()]

        param_groups = {
            'feature_extractor': feature_extractor_layers,
            'classifier': classifier_layers
        }

        return param_groups


    def get_layer_groups1(self):
        roberta_layers = [elem[1] for elem in self.roberta.named_parameters()]
        lstm_layers = [elem[1] for elem in self.lstm.named_parameters()]


        classifier_layers = [elem[1] for elem in self.classifier.named_parameters()]
        param_groups = {
            'roberta': roberta_layers,
            'lstm': lstm_layers,
            'classifier': classifier_layers
        }
        return param_groups



