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




# transformers 8 单纯加层其实没有什么用？
class SimcseModel8(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model, pooling, only_embeddings=False):
        super(SimcseModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.num_sen = 3

        # v1
        # embedding_size = 64
        # num_codebook = 8
        # num_codeword = None
        # self.hidden_size = 128

        # v2
        # embedding_size = 288
        embedding_size = 224
        num_codebook = 8
        num_codeword = None
        self.hidden_size = 768



        # 使用CompositionalEmbedding(编码类型为'cc')
        self.embedding = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                weighted=True)

        # 使用Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4, dim_feedforward=embedding_size*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        self.linear = nn.Linear(embedding_size, self.hidden_size * 2)
        # self.dropout = nn.Dropout(0.2)
        # self.encoder_ln = nn.LayerNorm(self.hidden_size * 2)

        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)



        # Caps
        # # Parameters for the linear part of the capsule network
        # self.in_length = 8  # Input capsule length
        # self.out_length = 16  # Output capsule length
        # self.out_capsules = 32  # Number of output capsules
        #
        # # Transform BERT output to match Capsule Network input requirements
        # self.transformer = nn.Linear(self.hidden_size * 2, self.in_length * self.out_capsules) # Adjusted to match Capsule input
        #
        #
        # # Linear capsule layer
        # self.capsule_linear = CapsuleLinear(
        #     out_capsules=self.out_capsules,
        #     in_length=self.in_length,
        #     out_length=self.out_length,
        #     in_capsules=None,
        #     share_weight=True,
        #     # Assuming 'routing_type' and 'num_iterations' are parameters of CapsuleLinear
        #     routing_type='k_means',
        #     num_iterations=3
        # )
        #
        # self.dropout = nn.Dropout(0.15)
        # self.bn_sent_rep = nn.BatchNorm1d(768)
        # self.bn_sent_rep1 = nn.BatchNorm1d(32)

        self.only_embeddings = only_embeddings




    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None,teacher_embedding=None):
        # 使用CompositionalEmbedding编码
        embeddings = self.embedding(input_ids)


        # 准备Transformer编码器的attention_mask
        att_mask = ~attention_mask.bool()

        # 使用Transformer编码器
        encoder_outputs = self.encoder(embeddings, src_key_padding_mask=att_mask)

        # Pooling
        if self.pooling == 'cls':
            sent_rep = self.linear(encoder_outputs[:, 0])  # 取CLS token作为句子表示
        elif self.pooling == 'last-avg':
            sent_rep = torch.mean(encoder_outputs, dim=1)  # 对序列最后一层隐状态进行平均池化
            sent_rep = self.linear(sent_rep)
        # sent_rep = self.dropout(sent_rep)

        # sent_rep = self.bn_sent_rep(sent_rep)
        # transformed_rep = self.transformer(sent_rep).view(-1, self.out_capsules, self.in_length)
        # transformed_rep = transformed_rep.float()


        # Pass the transformed representation to the CapsuleLinear layer
        # sent_rep1 = self.capsule_linear(transformed_rep)[0].norm(dim=-1)
        #
        # sent_rep = self.bn_sent_rep(sent_rep)
        # sent_rep1 = self.bn_sent_rep1(sent_rep1)

        #sent_rep = torch.concat([sent_rep, sent_rep1], dim=-1)

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
            if teacher_top1_sim_pred is not None:
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
                # print(teacher_top1_sim_pred)
                # print(student_top1_sim_pred)


                if teacher_embedding is not None:
                    align_loss_value = align_loss(z1, teacher_embedding, alpha=2)
                else:
                    align_loss_value = 0

                # print('hello')
                # loss = loss + 0.1 * sd_loss + 0.1 * kd_loss
                loss = loss + kd_loss + align_loss_value
                # loss = kd_loss
                return loss, sent_rep


            loss = loss + 0.1 * sd_loss
            # print(f"Primary loss: {loss.item()}, Self-distillation loss: {sd_loss.item()}")
            # loss = loss + 0.1*sd_loss
            # loss = sd_loss



            return loss, sent_rep
        else:
            return sent_rep



# simcse bert 7 + dnn
#         if routing_type == 'k_means':
#             self.classifier = CapsuleLinear(out_capsules=num_class, in_length=self.in_length,
#                                             out_length=self.out_length, in_capsules=None, share_weight=True,
#                                             routing_type='k_means', num_iterations=num_iterations, bias=False)
#         elif routing_type == 'dynamic':
#             self.classifier = CapsuleLinear(out_capsules=num_class, in_length=self.in_length,
#                                             out_length=self.out_length, in_capsules=None, share_weight=True,
#                                             routing_type='dynamic', num_iterations=num_iterations, bias=False)
class SimcseModel7(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model: str, pooling: str, routing_type='k_means', only_embeddings=False):
        super(SimcseModel, self).__init__()
        # config = AutoConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.only_embeddings = only_embeddings
        self.num_sen = 3
        self.has_neg = False




        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)



        # Caps
        # Parameters for the linear part of the capsule network
        self.in_length = 8  # Input capsule length
        self.out_length = 16  # Output capsule length
        self.out_capsules = 32  # Number of output capsules

        # Transform BERT output to match Capsule Network input requirements
        self.transformer = nn.Linear(768, self.in_length * self.out_capsules) # Adjusted to match Capsule input


        # Linear capsule layer
        self.capsule_linear = CapsuleLinear(
            out_capsules=self.out_capsules,
            in_length=self.in_length,
            out_length=self.out_length,
            in_capsules=None,
            share_weight=True,
            # Assuming 'routing_type' and 'num_iterations' are parameters of CapsuleLinear
            routing_type='k_means',
            num_iterations=3
        )

        self.dropout = nn.Dropout(0.3)
        self.bn_sent_rep = nn.BatchNorm1d(768)
        self.bn_sent_rep1 = nn.BatchNorm1d(32)

    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            # return out.last_hidden_state[:, 0]  # [batch, 768]
            sent_rep = out.last_hidden_state[:, 0]

        if self.pooling == 'pooler':
            # return out.pooler_output            # [batch, 768]
            sent_rep = out.pooler_output

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            # return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
            sent_rep = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            # return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
            sent_rep = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

        # Transform BERT output dimension to match Capsule input dimension
        sent_rep = self.dropout(sent_rep)
        # transformed_rep = self.transformer(sent_rep).view(-1, self.out_capsules, self.in_length)

        # Pass the transformed representation to the CapsuleLinear layer
        # sent_rep1 = self.capsule_linear(transformed_rep)[0].norm(dim=-1)

        # sent_rep = self.bn_sent_rep(sent_rep)
        # sent_rep1 = self.bn_sent_rep1(sent_rep1)

        # sent_rep = torch.concat([sent_rep, sent_rep1], dim=-1)

        # print(len(sent_rep))

        if self.training and not self.only_embeddings:

            # 计算损失
            loss = simcse_sup_loss(sent_rep)

            # -----------------
            # Separate representation
            # v1


            batch_size = input_ids.size(0)
            sent_rep = sent_rep.view((batch_size//self.num_sen, self.num_sen, sent_rep.size(-1)))
            z1, z2 = sent_rep[:, 0], sent_rep[:, 1]
            # Hard negative
            if self.has_neg:
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
            if teacher_top1_sim_pred is not None:
                student_top1_sim_pred = z1_z2_cos.clone()

            if self.has_neg:
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
                # print(teacher_top1_sim_pred)
                # print(student_top1_sim_pred)


                # print('hello')
                loss = loss + sd_loss + kd_loss
                # loss = kd_loss
                return loss, sent_rep


            loss = loss + 0.1 * sd_loss
            # print(f"Primary loss: {loss.item()}, Self-distillation loss: {sd_loss.item()}")
            # loss = loss + 0.1*sd_loss
            # loss = sd_loss

            return loss, sent_rep
        else:
            return sent_rep



# simcse bert 5
class SimcseModel5(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model: str, pooling: str, only_embeddings=False, teacher_isavailable=False):
        super(SimcseModel, self).__init__()
        # config = AutoConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.pooling = pooling

        self.num_sen = 3
        self.has_neg = False



        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)
    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            # return out.last_hidden_state[:, 0]  # [batch, 768]
            sent_rep = out.last_hidden_state[:, 0]

        if self.pooling == 'pooler':
            # return out.pooler_output            # [batch, 768]
            sent_rep = out.pooler_output

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            # return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
            sent_rep = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            # return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
            sent_rep = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

        if self.training:

            # 计算损失
            loss = simcse_sup_loss(sent_rep)

            # -----------------
            # Separate representation
            # v1


            batch_size = input_ids.size(0)
            sent_rep = sent_rep.view((batch_size//self.num_sen, self.num_sen, sent_rep.size(-1)))
            z1, z2 = sent_rep[:, 0], sent_rep[:, 1]
            # Hard negative
            if self.has_neg:
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
            if teacher_top1_sim_pred is not None:
                student_top1_sim_pred = z1_z2_cos.clone()

            if self.has_neg:
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
                # print(teacher_top1_sim_pred)
                # print(student_top1_sim_pred)


                # print('hello')
                loss = loss + sd_loss + kd_loss
                # loss = kd_loss
                return loss, sent_rep


            loss = loss + 0.1 * sd_loss
            # print(f"Primary loss: {loss.item()}, Self-distillation loss: {sd_loss.item()}")
            # loss = loss + 0.1*sd_loss
            # loss = sd_loss

            return loss, sent_rep
        else:
            return sent_rep



# transformers 6
class SimcseModel6(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model, pooling, only_embeddings=False):
        super(SimcseModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.num_sen = 3
        self.only_embeddings = only_embeddings
        # v1
        # embedding_size = 64
        # num_codebook = 8
        # num_codeword = None
        # self.hidden_size = 128

        # v2
        # embedding_size = 288
        embedding_size = 224
        num_codebook = 8
        num_codeword = None
        self.hidden_size = 384



        # 使用CompositionalEmbedding(编码类型为'cc')
        self.embedding = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                weighted=True)

        # 使用Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4, dim_feedforward=embedding_size*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.linear = nn.Linear(embedding_size, self.hidden_size * 2)
        # self.dropout = nn.Dropout(0.2)
        # self.encoder_ln = nn.LayerNorm(self.hidden_size * 2)

        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)

    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None,teacher_embedding=None):
        # 使用CompositionalEmbedding编码
        embeddings = self.embedding(input_ids)


        # 准备Transformer编码器的attention_mask
        att_mask = ~attention_mask.bool()

        # 使用Transformer编码器
        encoder_outputs = self.encoder(embeddings, src_key_padding_mask=att_mask)

        # Pooling
        if self.pooling == 'cls':
            sent_rep = self.linear(encoder_outputs[:, 0])  # 取CLS token作为句子表示
        elif self.pooling == 'last-avg':
            sent_rep = torch.mean(encoder_outputs, dim=1)  # 对序列最后一层隐状态进行平均池化
            sent_rep = self.linear(sent_rep)
        # sent_rep = self.dropout(sent_rep)



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
            if teacher_top1_sim_pred is not None:
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
                # print(teacher_top1_sim_pred)
                # print(student_top1_sim_pred)


                if teacher_embedding is not None:
                    align_loss_value = align_loss(z1, teacher_embedding, alpha=2)
                else:
                    align_loss_value = 0

                # print('hello')
                # loss = loss + 0.1 * sd_loss + 0.1 * kd_loss
                loss = loss + kd_loss + align_loss_value
                # loss = kd_loss
                return loss, sent_rep


            loss = loss + 0.1 * sd_loss
            # print(f"Primary loss: {loss.item()}, Self-distillation loss: {sd_loss.item()}")
            # loss = loss + 0.1*sd_loss
            # loss = sd_loss



            return loss, sent_rep
        else:
            return sent_rep





# 双向lstm + 自蒸馏 + 教师 目前sota架构
# SimcseModel2
class SimcseModel2(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, only_embeddings=False):
        super(SimcseModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.num_sen = 3

        # v1
        # embedding_size = 64
        # num_codebook = 8
        # num_codeword = None
        # self.hidden_size = 128

        # v2
        embedding_size = 224
        num_codebook = 8
        num_codeword = None
        self.hidden_size = 384

        # 使用CompositionalEmbedding(编码类型为'cc')
        self.embedding = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                weighted=True)

        # 使用双向LSTM编码器
        self.encoder = nn.LSTM(embedding_size, self.hidden_size, num_layers=2, dropout=0.5, batch_first=True,
                               bidirectional=True)
        # self.encoder = nn.LSTM(embedding_size, self.hidden_size, num_layers=2, dropout=0.1, batch_first=True,
        #                        bidirectional=True)

        # self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        # self.dropout = nn.Dropout(0.2)
        # self.encoder_ln = nn.LayerNorm(self.hidden_size * 2)

        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)

    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):
        # 使用CompositionalEmbedding编码
        embeddings = self.embedding(input_ids)
        # 使用双向LSTM编码器
        encoder_outputs, _ = self.encoder(embeddings)



        # Pooling
        if self.pooling == 'cls':
            # 取每个序列的第一个输出作为句子表示
            sent_rep = encoder_outputs[:, 0, :]  # [batch, hidden_size * 2]
            sent_rep = self.linear(sent_rep)
            # sent_rep = self.encoder_ln(sent_rep)
            # sent_rep = self.dropout(sent_rep)

        elif self.pooling == 'last-avg':
            # 对序列的最后一层隐状态进行平均池化
            fwd_output = encoder_outputs[:, :, :self.hidden_size]
            bwd_output = encoder_outputs[:, :, self.hidden_size:]
            avg_fwd = torch.mean(fwd_output, dim=1)
            avg_bwd = torch.mean(bwd_output, dim=1)
            sent_rep = torch.cat([avg_fwd, avg_bwd], dim=1)  # [batch, hidden_size * 2]
            # sent_rep = self.linear(sent_rep) # 额外线性层
            # sent_rep = self.encoder_ln(sent_rep) # 归一化
            # sent_rep = self.dropout(sent_rep)

        # sent_rep = self.dropout(sent_rep)



        # 根据训练模式返回不同的输出
        if self.training:

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
                # loss = loss + 0.1 * sd_loss + 0.1 * kd_loss
                # loss = loss + 0.1 * sd_loss + kd_loss + align_loss_value
                loss = loss + kd_loss + align_loss_value
                # loss = kd_loss

                return loss, sent_rep


            # al = align_loss(z1,z2)
            # ul = uniform_loss(z1)

            loss = loss + 0.1*sd_loss
            # print(f"Primary loss: {loss.item()}, Self-distillation loss: {sd_loss.item()}")
            # loss = loss + 0.1*sd_loss
            # loss = sd_loss




            return loss, sent_rep
        else:
            return sent_rep


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


# 1d_ResNet 9
class SimcseModel9(nn.Module):
    def __init__(self, pretrained_model, pooling='cls', only_embeddings=False, teacher_isavailable=False):
        super(SimcseModel, self).__init__()
        self.only_embeddings = only_embeddings
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



        # 使用nn.Sequential构建1d_resnet结构
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=3, stride=3),
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


        # 对比学习
        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)


    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):
        embeddings = self.embedding(input_ids)
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
                # loss = loss + 0.1 * sd_loss + 0.1 * kd_loss
                # loss = loss + 0.1 * sd_loss + kd_loss + align_loss_value
                loss = loss + kd_loss + align_loss_value
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





# super resnet 10
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



# # 导入数学相关模块
# import math
# # 导入partial，用于创建部分应用函数
# from functools import partial
# # 导入json模块，用于处理JSON格式数据
# import json
# # 导入os模块，用于访问操作系统提供的功能
# import os
#
# # 导入命名元组，用于定义一种简单的不可变数据结构
# from collections import namedtuple
#
# # 导入PyTorch相关模块
# import torch
# import torch.nn as nn
#
# # 从mamba_ssm.models.config_mamba导入MambaConfig，用于配置管理
# from mamba_ssm.models.config_mamba import MambaConfig
# # 从mamba_ssm.modules.mamba_simple导入Mamba和Block，用于构建模型
# from mamba_ssm.modules.mamba_simple import Mamba, Block
# # 从mamba_ssm.utils.generation导入GenerationMixin，提供生成相关功能
# from mamba_ssm.utils.generation import GenerationMixin
# # 从mamba_ssm.utils.hf导入load_config_hf和load_state_dict_hf，用于加载Hugging Face的配置和状态字典
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
#
# # 尝试导入mamba_ssm.ops.triton.layernorm中的RMSNorm, layer_norm_fn, rms_norm_fn，
# # 如果导入失败，则将它们设置为None
# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
#


def create_block(
    d_model,  # 模型的维度
    ssm_cfg=None,  # 状态空间模型的配置，默认为None
    norm_epsilon=1e-5,  # 层归一化时的epsilon值，防止除以零
    rms_norm=False,  # 是否使用RMSNorm而非LayerNorm
    residual_in_fp32=False,  # 是否在残差连接中使用FP32
    fused_add_norm=False,  # 是否融合加法和归一化操作
    layer_idx=None,  # 层的索引，默认为None
    device=None,  # 指定运行设备，默认为None
    dtype=None,  # 数据类型，默认为None
):
    """
    创建一个层块（Block）。

    参数：
    - d_model: 模型的维度。
    - ssm_cfg: 状态空间模型的配置，默认为None，如果需要自定义SSM配置，则传入相应的字典。
    - norm_epsilon: 层归一化中用于避免除以零的epsilon值，默认为1e-5。
    - rms_norm: 是否使用RMSNorm而不是LayerNorm，默认为False。
    - residual_in_fp32: 是否在残差连接中使用FP32精度，默认为False。
    - fused_add_norm: 是否将加法和归一化操作融合，默认为False。
    - layer_idx: 层的索引，默认为None，用于标识层在模型中的位置。
    - device: 指定运行设备，默认为None，自动选择。
    - dtype: 数据类型，默认为None，自动选择。

    返回值：
    - block: 创建的层块实例。
    """
    if ssm_cfg is None:
        ssm_cfg = {}  # 默认的SSM配置为空字典

    # 根据配置创建Mixer类的实例化函数
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    # 根据是否使用RMSNorm选择归一化类
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    # 创建Block实例
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )

    block.layer_idx = layer_idx  # 设置层索引
    return block



# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # 仅用于嵌入层的初始化范围。
    rescale_prenorm_residual=True,  # 是否根据层的数量重新调整残差的规模。
    n_residuals_per_layer=1,  # 如果有MLP，则每个层的残差数量变为2。
):
    """
    初始化模块的权重。

    参数:
    - module: 需要初始化权重的模块。
    - n_layer: 模型的层数，用于残差的重新初始化。
    - initializer_range: 用于嵌入层权重初始化的标准差。
    - rescale_prenorm_residual: 是否根据层数调整预归一化残差的权重。
    - n_residuals_per_layer: 每层的残差数量，用于计算残差权重的重初始化规模。
    """
    if isinstance(module, nn.Linear):
        # 对线性层的偏置进行初始化
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # 对嵌入层的权重使用正态分布进行初始化
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # 根据OpenAI GPT-2论文中的方案重新初始化选定的权重
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # 特殊的规模初始化，针对每个变换块中有两个层归一化的场景
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)  # 调整权重规模以考虑残差的累积


# 1d_ResNet+mamba 11
class SimcseModel11(nn.Module):
    def __init__(self, pretrained_model, pooling='cls', only_embeddings=False):
        super(SimcseModel, self).__init__()
        self.only_embeddings = only_embeddings
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

        # Mamba 模块配置
        d_model = 512
        n_layer = 27
        ssm_cfg = {'d_state': 16, 'expand': 2, 'd_conv': 4}
        norm_epsilon = 1e-5
        rms_norm = True
        fused_add_norm = True
        factory_kwargs = {}

        self.layers = nn.ModuleList([
            create_block(
                d_model=d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=True,
                layer_idx=i,
                **factory_kwargs
            ) for i in range(n_layer)
        ])

        # 使用nn.Sequential构建1d_resnet结构
        self.layers.extend([
            nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=3, stride=3),
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
        ])

        # 归一化层初始化
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # 权重初始化
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                initializer_range=0.02  # 假设的初始化范围
            )
        )

        # 对比学习
        self.div = Divergence(beta_=0.5)
        self.sim = Similarity(temp=0.05)


        # teacher
        gamma_ = 0.10
        tau2 = 0.05
        self.distillation_loss_fct = ListNet(tau2, gamma_)


    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):
        embeddings = self.embedding(input_ids)
        embeddings = embeddings.permute(0, 2, 1)  # 调整维度以适应Conv1d
        sent_rep = self.layers(embeddings)

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
                # loss = loss + 0.1 * sd_loss + 0.1 * kd_loss
                # loss = loss + 0.1 * sd_loss + kd_loss + align_loss_value
                loss = loss + kd_loss + align_loss_value
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


# 单向lstm
# SimcseModel1
class SimcseModel1(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model: str, pooling: str, only_embeddings=False):
        super(SimcseModel, self).__init__()
        self.pooling = pooling
        config = AutoConfig.from_pretrained(pretrained_model)

        embedding_size = 128
        num_codebook = 8
        num_codeword = None
        hidden_size = 328



        # 使用CompositionalEmbedding(编码类型为'cc')
        self.embedding = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                weighted=True)

        # 使用LSTM编码器
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True,
                               bidirectional=True)

    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):
        # 使用CompositionalEmbedding编码
        embeddings = self.embedding(input_ids)

        # 使用LSTM编码器
        encoder_outputs, _ = self.encoder(embeddings)

        # Pooling
        if self.pooling == 'cls':
            return encoder_outputs[:, 0]  # [batch, 768]
        elif self.pooling == 'last-avg':
            last = encoder_outputs.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            raise NotImplementedError("first-last-avg pooling not available without BERT embeddings")


# simcse model0
class SimcseModel0(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str, only_embeddings=False):
        super(SimcseModel, self).__init__()
        # config = AutoConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):
        
        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)




        if self.pooling == 'cls':
            # return out.last_hidden_state[:, 0]  # [batch, 768]
            return out.last_hidden_state[:, 0]
        if self.pooling == 'pooler':
            # return out.pooler_output            # [batch, 768]
            return out.pooler_output
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            # return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
            sent_rep = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            # return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
            sent_rep = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)



        if self.training:
            loss = simcse_sup_loss(sent_rep)
            return loss, sent_rep
        else:
            return sent_rep


# 双向lstm
# SimcseModel4
class SimcseModel4(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, only_embeddings=False):
        super(SimcseModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.pooling = pooling

        # v1
        # embedding_size = 64
        # num_codebook = 8
        # num_codeword = None
        # self.hidden_size = 128

        # v2
        embedding_size = 128
        num_codebook = 8
        num_codeword = None
        self.hidden_size = 328


        # 使用CompositionalEmbedding(编码类型为'cc')
        self.embedding = CompositionalEmbedding(config.vocab_size, embedding_size, num_codebook, num_codeword,
                                                weighted=True)

        # 使用双向LSTM编码器
        self.encoder = nn.LSTM(embedding_size, self.hidden_size, num_layers=2, dropout=0.1, batch_first=True,
                               bidirectional=True)

    def forward(self, input_ids, attention_mask, token_type_ids, teacher_top1_sim_pred=None, teacher_embedding=None):
        # 使用CompositionalEmbedding编码
        embeddings = self.embedding(input_ids)

        # 使用双向LSTM编码器
        encoder_outputs, _ = self.encoder(embeddings)

        # Pooling
        if self.pooling == 'cls':
            # 取每个序列的第一个输出作为句子表示
            return encoder_outputs[:, 0, :]  # [batch, hidden_size * 2]

        elif self.pooling == 'last-avg':
            # 对序列的最后一层隐状态进行平均池化
            fwd_output = encoder_outputs[:, :, :self.hidden_size]
            bwd_output = encoder_outputs[:, :, self.hidden_size:]
            avg_fwd = torch.mean(fwd_output, dim=1)
            avg_bwd = torch.mean(bwd_output, dim=1)
            sent_rep = torch.cat([avg_fwd, avg_bwd], dim=1)
            return sent_rep  # [batch, hidden_size * 2]


    # def forward(self, tensor_inputs):
    #     src_outputs, tgt_outputs = self.embedding(tensor_inputs)
    #     combined_output = torch.cat(
    #         [src_outputs, tgt_outputs, torch.abs(src_outputs - tgt_outputs), src_outputs * tgt_outputs], dim=1)
    #     outputs = self.prediction(self.dropout(self.fc(combined_output)))
    #     return outputs


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



