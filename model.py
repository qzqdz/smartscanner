import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import BertModel, AutoConfig,AutoModel
from capsule_layer import CapsuleLinear,CapsuleConv2d

from src.modeling.network.rankcse.models import ListNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]

    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()



# 自蒸馏组件1 差异度 change
class Divergence(nn.Module):
    """
    Jensen-Shannon divergence, used to measure ranking consistency between similarity lists obtained from examples with two different dropout masks
    """
    def __init__(self, beta_):
        super(Divergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log().clamp(min=self.eps)
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


# 相似度组件2 change
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class CompositionalEmbedding(nn.Module):
    r"""A simple compositional codeword and codebook that store embeddings.

     Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): size of each embedding vector
        num_codebook (int): size of the codebook of embeddings
        num_codeword (int, optional): size of the codeword of embeddings
        weighted (bool, optional): weighted version of unweighted version
        return_code (bool, optional): return code or not

     Shape:
         - Input: (LongTensor): (N, W), W = number of indices to extract per mini-batch
         - Output: (Tensor): (N, W, embedding_dim)

     Attributes:
         - code (Tensor): the learnable weights of the module of shape
              (num_embeddings, num_codebook, num_codeword)
         - codebook (Tensor): the learnable weights of the module of shape
              (num_codebook, num_codeword, embedding_dim)

     Examples:
         >>> m = CompositionalEmbedding(200, 64, 16, 32, weighted=False)
         >>> a = torch.randperm(128).view(16, -1)
         >>> output = m(a)
         >>> print(output.size())
         torch.Size([16, 8, 64])
     """

    def __init__(self, num_embeddings, embedding_dim, num_codebook, num_codeword=None, num_repeat=10, weighted=True,
                 return_code=False):
        super(CompositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_codebook = num_codebook
        self.num_repeat = num_repeat
        self.weighted = weighted
        self.return_code = return_code

        if num_codeword is None:
            num_codeword = math.ceil(math.pow(num_embeddings, 1 / num_codebook))
        self.num_codeword = num_codeword
        self.code = Parameter(torch.Tensor(num_embeddings, num_codebook, num_codeword))
        self.codebook = Parameter(torch.Tensor(num_codebook, num_codeword, embedding_dim))


        nn.init.normal_(self.code)
        nn.init.normal_(self.codebook)

    def forward(self, input):
        batch_size = input.size(0)
        index = input.view(-1)
        code = self.code.index_select(dim=0, index=index)
        # print(code.shape)
        if self.weighted:
            # reweight, do softmax, make sure the sum of weight about each book to 1
            code = F.softmax(code, dim=-1)
            out = (code[:, :, None, :] @ self.codebook[None, :, :, :]).squeeze(dim=-2).sum(dim=1)
        else:
            # because Gumbel SoftMax works in a stochastic manner, needs to run several times to
            # get more accurate embedding
            code = (torch.sum(torch.stack([F.gumbel_softmax(code) for _ in range(self.num_repeat)]), dim=0)).argmax(
                dim=-1)
            out = []
            for index in range(self.num_codebook):
                out.append(self.codebook[index, :, :].index_select(dim=0, index=code[:, index]))
            out = torch.sum(torch.stack(out), dim=0)
            code = F.one_hot(code, num_classes=self.num_codeword)

        out = out.view(batch_size, -1, self.embedding_dim)
        code = code.view(batch_size, -1, self.num_codebook, self.num_codeword)
        if self.return_code:
            return out, code
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_embeddings) + ', ' + str(self.embedding_dim) + ')'






class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x + inputs
        return self.relu(out)




class SimcseModel(nn.Module):
    def __init__(self, pretrained_model, pooling='cls', only_embeddings=False, teacher_isavailable=False):
        super(SimcseModel, self).__init__()
        self.only_embeddings = only_embeddings
        self.teacher_isavailable = teacher_isavailable
        config = AutoConfig.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.num_sen = 3
        # v1
        # embedding_size = 64
        # num_codebook = 8
        # num_codeword = None
        # self.hidden_size = 128

        # v2
        embedding_size = 3
        num_codebook = 8
        num_codeword = None
        self.hidden_size = 384

        # 使用CompositionalEmbedding(编码类型为'cc')
        self.embedding = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                weighted=True)

        self.embedding2 = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                 weighted=True)




        # 使用nn.Sequential构建1d_resnet结构
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=3, stride=3),
            # nn.Conv1d(in_channels=embedding_size * 2, out_channels=128, kernel_size=3, stride=3),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            ResBlock1D(in_channels=128, out_channels=128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            ResBlock1D(in_channels=128, out_channels=128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            ResBlock1D(in_channels=256, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            ResBlock1D(in_channels=256, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            ResBlock1D(in_channels=256, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            ResBlock1D(in_channels=512, out_channels=512),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Dropout(0.4),
            nn.Flatten()
        )


        if self.teacher_isavailable:
            # self.unfreeze_module(self.embedding)  # 第一阶段，解冻self.embedding
            self.freeze_module(self.embedding2)  # 同时冻结self.embedding2


            # 知识兼容
            self.freeze_module(self.layers)
            # 在模型的某个测试点调用此函数
            self.check_frozen_status(self.layers)
            # # 示例：检查layers模块中参数的状态
            # for name, param in self.layers.named_parameters():
            #     print(f"{name} requires_grad: {param.requires_grad}")
            #
            # # 检查BatchNorm层的运行模式
            # for layer in self.layers.modules():
            #     if isinstance(layer, nn.BatchNorm1d):
            #         print(f"BatchNorm1d in eval mode: {not layer.training}")

        else:
            # self.unfreeze_module(self.embedding2)  # 第二阶段，解冻self.embedding2
            # self.unfreeze_module(self.embedding)
            self.freeze_module(self.embedding)  # 同时冻结self.embedding


        # 对比学习
        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)


    def freeze_module_only_itself(self, module):
        """冻结模块的参数，使其在训练中不更新"""
        for param in module.parameters():
            param.requires_grad = False

    # def freeze_module(self, module):
    #     """冻结模块的参数，并确保所有批量归一化层都被设置为评估模式"""
    #     # for param in module.parameters():
    #     #     param.requires_grad = False
    #
    #     for child in module.children():
    #         # 冻结所有参数
    #         for param in child.parameters():
    #             param.requires_grad = False
    #         # 检查批量归一化层并设置为评估模式
    #         if isinstance(child, nn.BatchNorm1d):
    #             child.eval()
    #         # 递归地应用到所有子模块
    #         self.freeze_module(child)

    def freeze_module(self, module):
        """冻结模块中所有BatchNorm1d层的参数，并确保它们被设置为评估模式"""
        for child in module.children():
            if isinstance(child, nn.BatchNorm1d):
                # 冻结BatchNorm1d层的所有参数
                for param in child.parameters():
                    param.requires_grad = False
                # 将BatchNorm1d层设置为评估模式
                child.eval()
            # 递归地应用到所有子模块
            self.freeze_module(child)

    def unfreeze_module(self, module):
        """解冻模块的参数，使其在训练中可以更新"""
        for param in module.parameters():
            param.requires_grad = True

    def check_frozen_status(self, module):
        for child in module.children():
            if isinstance(child, nn.BatchNorm1d):
                print(f"BatchNorm1d layer {child} is frozen with eval mode: {not child.training}")
            elif hasattr(child, 'parameters'):
                for name, param in child.named_parameters():
                    print(f"{name} requires_grad: {param.requires_grad}")
            self.check_frozen_status(child)  # 递归检查

    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):



        if self.training and self.teacher_isavailable:

            # self.unfreeze_module(self.embedding)  # 第一阶段，解冻self.embedding
            self.freeze_module_only_itself(self.embedding2)  # 同时冻结self.embedding2


            # 知识兼容
            self.freeze_module(self.layers)


            # embeddings = self.embedding(input_ids)
            # embeddings2 = self.embedding2(input_ids)
            # embeddings = embeddings + embeddings2

            embeddings = self.embedding2(input_ids)

            # embeddings = torch.cat([embeddings, embeddings], dim=-1)
            # print(embeddings.shape)
        elif self.training and not self.teacher_isavailable:

            # self.unfreeze_module(self.embedding2)  # 第二阶段，解冻self.embedding2
            # self.unfreeze_module(self.embedding)
            # self.freeze_module_only_itself(self.embedding)  # 同时冻结self.embedding
            self.freeze_module(self.layers)
            self.freeze_module_only_itself(self.embedding2)  # 同时冻结self.embedding2

            # stage 1+2
            # embeddings = self.embedding(input_ids)
            embeddings = self.embedding2(input_ids)  # 使用self.embedding2生成嵌入
            # embeddings = embeddings+embeddings2

        else:
            # stage 1+2
            # embeddings = self.embedding(input_ids)
            # embeddings2 = self.embedding2(input_ids)  # 使用self.embedding2生成嵌入
            # embeddings = embeddings+embeddings2
            # stage 1
            # embeddings = self.embedding(input_ids)

            # stage 2
            embeddings = self.embedding2(input_ids)

            # joint concat
            # embeddings = torch.cat([embeddings, embeddings2], dim=1)
            # embeddings = torch.cat([embeddings2, 0.5*embeddings+embeddings2], dim=1)

            # jc v2
            # embeddings = torch.cat([embeddings+embeddings2, embeddings2], dim=1)

            # jc v3
            # embeddings = torch.cat([embeddings+embeddings2, embeddings2, embeddings], dim=1)

            # stack
            # embeddings = torch.cat([embeddings, embeddings2], dim=-1)



        embeddings = embeddings.permute(0, 2, 1)  # 调整维度以适应Conv1d
        sent_rep = self.layers(embeddings)
        # 这里返回的feature_rep可用于后续的任务，如分类、相似度计算等


        # 根据训练模式返回不同的输出
        if self.training and not self.only_embeddings:

            # if self.pooling == 'cls':
            #     sent_rep_2 = encoder_outputs[:, 0, :]
            # elif self.pooling == 'last-avg':
            #     fwd_output = encoder_outputs[:, :, :self.hidden_size]
            #     bwd_output = encoder_outputs[:, :, self.hidden_size:]
            #     avg_fwd = torch.mean(fwd_output, dim=1)
            #     avg_bwd = torch.mean(bwd_output, dim=1)
            #     sent_rep_2 = torch.cat([avg_fwd, avg_bwd], dim=1)
            #
            # # sent_rep_2 = self.dropout(sent_rep_2)
            #
            # z1_z2_cos = self.sim(sent_rep, sent_rep_2)
            # # z2_z1_cos = self.sim(sent_rep, sent_rep_2)
            # z2_z1_cos = self.sim(sent_rep_2, sent_rep)

            # print(z2_z1_cos,z1_z2_cos)
            # 计算自蒸馏损失
            # sd_loss = self.div(z1_z2_cos.softmax(dim=-1).clamp(min=1e-7), z2_z1_cos.softmax(dim=-1).clamp(min=1e-7))
            # sd_loss = self.div(z1_z2_cos, z2_z1_cos)
            # 或者使用负余弦相似度作为自蒸馏损失
            # sd_loss = 1 - (z1_z2_cos.softmax(dim=-1).clamp(min=1e-7) * z2_z1_cos.softmax(dim=-1).clamp(min=1e-7)).sum(dim=-1).mean()

            # 计算损失
            loss = simcse_sup_loss(sent_rep)

            # -----------------
            # Separate representation
            # v1


            batch_size = input_ids.size(0)
            sent_rep = sent_rep.view((batch_size//self.num_sen, self.num_sen, sent_rep.size(-1)))
            z1, z2 = sent_rep[:, 0], sent_rep[:, 1]
            # Hard negative
            if self.num_sen == 3:
                z3 = sent_rep[:, 2]

            # v2-----------------
            # batch_size = input_ids.size(0)
            # sent_rep = sent_rep.view((batch_size//self.num_sen, self.num_sen, sent_rep.size(-1)))
            # shuffle_indices = torch.randperm(sent_rep.size(0))
            # shuffled_sent_rep = sent_rep.clone()
            # shuffled_sent_rep[:, 0], shuffled_sent_rep[:, 1] = sent_rep[shuffle_indices][:, 1], sent_rep[shuffle_indices][:,0]
            #
            # z1, z2 = shuffled_sent_rep[:, 0], shuffled_sent_rep[:, 1]

            # -----------------
            z1_z2_cos = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            z2_z1_cos = self.sim(z2.unsqueeze(1), z1.unsqueeze(0))
            # print(z1_z2_cos.shape)
            # 知识蒸馏
            student_top1_sim_pred = z1_z2_cos.clone()

            if self.num_sen >= 3:
                z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
                z2_z3_cos = self.sim(z2.unsqueeze(1), z3.unsqueeze(0))
                z1_z2_cos = torch.cat([z1_z2_cos, z1_z3_cos], 1)
                z2_z1_cos = torch.cat([z2_z1_cos, z2_z3_cos], 1)

            # --------
            # 随机选择要交换的索引
            # num_rows, num_cols = z1_z2_cos.shape
            # row_idx = torch.randint(0, num_rows, (1,)).item()
            # col_idx = torch.randint(0, num_cols, (1,)).item()
            #
            # # 在相同i,j位置上进行数值交换
            # temp = z1_z2_cos[row_idx, col_idx].clone()
            # z1_z2_cos[row_idx, col_idx] = z2_z1_cos[row_idx, col_idx]
            # z2_z1_cos[row_idx, col_idx] = temp


            # --------


            sd_loss = self.div(z1_z2_cos.softmax(dim=-1).clamp(min=1e-7), z2_z1_cos.softmax(dim=-1).clamp(min=1e-7))


            # ----- teacher-------
            if teacher_top1_sim_pred is not None:
                kd_loss = self.distillation_loss_fct(teacher_top1_sim_pred, student_top1_sim_pred)



                if teacher_embedding is not None:
                    align_loss_value = align_loss(z1, teacher_embedding, alpha=2)
                else:
                    align_loss_value = 0

                # 计算KL散度损失
                # if teacher_embedding is not None:
                #     student_logits = F.log_softmax(z1, dim=-1)
                #     teacher_logits = F.softmax(teacher_embedding, dim=-1)
                #     kl_div_loss = F.kl_div(student_logits, teacher_logits, reduction='batchmean')
                # else:
                #     kl_div_loss = 0

                # print('hello')

                # loss = loss + 0.1 * sd_loss + 1 * kd_loss
                # loss = loss + 0.1 * sd_loss + kd_loss + align_loss_value
                # loss = loss + kd_loss + align_loss_value
                # loss = kd_loss

                return loss, sent_rep


            # al = align_loss(z1,z2)
            # ul = uniform_loss(z1)

            # loss = loss + 0.5*sd_loss
            # print(f"Primary loss: {loss.item()}, Self-distillation loss: {sd_loss.item()}")

            loss = loss + 0.1*sd_loss
            # loss = sd_loss




            return loss, sent_rep
        else:
            return sent_rep





class SimcseSemanticSimilarity(nn.Module):
    def __init__(self, pretrained_model: str, pooling: str, hidden_dim: int, dropout=0.2):
        super(SimcseSemanticSimilarity, self).__init__()
        self.encoder = SimcseModel(pretrained_model, pooling, only_embeddings=True)
        # self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.prediction = nn.Linear(hidden_dim, 2)

    def encode(self, src_token_ids, src_mask,src_token_type_ids=None):
        sent_rep_1 = self.encoder(src_token_ids, src_mask, src_token_type_ids)
        return sent_rep_1

    def sim_cal(self, src_embedding, tgt_embeddings):
        """
        使用torch.cat计算一个句子与多个句子的语义相似度
        """
        n = tgt_embeddings.size(0)
        # 将src_embedding增加一个维度并扩展以匹配tgt_embeddings的数量
        src_embedding = src_embedding.repeat(n, 1)

        # 重新审视tgt_embeddings的形状，确保它们在拼接维度上是兼容的
        tgt_embeddings = tgt_embeddings

        combined_output = torch.cat([
            src_embedding,
            tgt_embeddings,
            torch.abs(src_embedding - tgt_embeddings),
            src_embedding * tgt_embeddings
        ], dim=1)


        # 预测相似度分数
        combined_output = self.dropout(self.fc(combined_output))

        similarity_scores = F.softmax(self.prediction(combined_output), dim=1)[:, 1]

        # print(src_embedding.shape)
        # similarity_scores = F.cosine_similarity(src_embedding, tgt_embeddings, dim=1)

        return similarity_scores
    def forward(self, src_token_ids, src_mask, tgt_token_ids, tgt_mask, tgt_token_type_ids=None,src_token_type_ids=None):
        # 获取句子表示向量
        # print(src_token_ids.shape)

        # sent_rep_1 = self.encoder(src_token_ids, src_mask, src_token_type_ids).last_hidden_state[:, 0]
        # sent_rep_2 = self.encoder(tgt_token_ids, tgt_mask, tgt_token_type_ids).last_hidden_state[:, 0]
        sent_rep_1 = self.encoder(src_token_ids, src_mask, src_token_type_ids)
        sent_rep_2 = self.encoder(tgt_token_ids, tgt_mask, tgt_token_type_ids)

        # 组合句子表示向量
        combined_output = torch.cat([
            sent_rep_1, sent_rep_2,
            torch.abs(sent_rep_1 - sent_rep_2),
            sent_rep_1 * sent_rep_2
        ], dim=1)

        # 预测相似度分数
        combined_output = self.dropout(self.fc(combined_output))
        similarity_scores = F.softmax(self.prediction(combined_output), dim=1)




        return similarity_scores



    def load_b_model_params(self, b_model_path, freeze_encoder=False, freeze_fc=False, freeze_prediction=False):
        b_model_state_dict = torch.load(b_model_path)
        self.encoder.load_state_dict(b_model_state_dict['model_state_dict']['encoder'])
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.fc.load_state_dict(b_model_state_dict['model_state_dict']['fc'])
        if freeze_fc:
            for param in self.fc.parameters():
                param.requires_grad = False

        self.prediction.load_state_dict(b_model_state_dict['model_state_dict']['prediction'])
        if freeze_prediction:
            for param in self.prediction.parameters():
                param.requires_grad = False



