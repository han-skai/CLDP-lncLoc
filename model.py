import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings('ignore')
import torch.optim
import torch.utils.data
import dgl.nn.pytorch as dglnn
from utils import *
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from info_nce_loss import info_nce_loss


class HGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, bel_out_feats, inc_out_feats):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            'belongs_to': dglnn.GraphConv(in_feats, hid_feats),  # (286, 128)
            'including': dglnn.GraphConv(in_feats, hid_feats),  # (286, 128)
            'connected_to': dglnn.GraphConv(in_feats, hid_feats)  # (286, 128)
        }, aggregate='mean')
        self.bn1_sample = nn.BatchNorm1d(hid_feats)
        self.bn1_label = nn.BatchNorm1d(hid_feats)

        self.conv2 = dglnn.HeteroGraphConv({
            'belongs_to': dglnn.GraphConv(hid_feats, bel_out_feats),  # (128, 256)
            'including': dglnn.GraphConv(hid_feats, inc_out_feats),   # (128, 256)
            'connected_to': dglnn.GraphConv(hid_feats, inc_out_feats)  # (128, 256)
        }, aggregate='mean')
        self.bn2_sample = nn.BatchNorm1d(inc_out_feats)
        self.bn2_label = nn.BatchNorm1d(bel_out_feats)


    def forward(self, graph, inputs):

        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}


        h['sequence'] = self.bn1_sample(h['sequence'])
        h['label'] = self.bn1_label(h['label'])

        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}

        h['sequence'] = self.bn2_sample(h['sequence'])
        h['label'] = self.bn2_label(h['label'])

        return h


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(286, 300, 1, bidirectional=True)
        self.linear = nn.Linear(300 * 2, 286)

    def forward(self, X):
        '''
        :param X: [batch_size, seq_len]
        :return:
        '''
        X = X.view(len(X), 1, -1)
        output, (final_hidden_state, final_cell_state) = self.lstm(
            X)  # output shape: [batch_size, seq_len=1,n_hidden * 2]
        output = output.transpose(0, 1)  # output : [seq_len=1, batch_size, n_hidden * num_directions(=2)]
        output = output.squeeze(0)  # [batch_size, n_hidden * num_directions(=2)]
        output = self.linear(output)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Linear(286, 256, bias=True)
        self.encoder.weight.data = self.encoder.weight.data.float()

    def forward(self, x):
        x = self.encoder(x)
        return x

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(572, 1, bias=False)
        # self.fc2 = nn.Linear(256, 1, bias=False)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.transfer.weight.data = self.transfer.weight.data.float()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        x = self.sigmoid(x)
        return x



class MLP_M_V(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_M_V, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.fc(x)

class MLP_T(nn.Module):
    def __init__(self, input_dim, output_dim):  #hidden_dim,
        super(MLP_T, self).__init__()
        # self.fc_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        # self.fc_2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        # h1 = F.tanh(self.fc_1(x))
        # out = self.fc_2(h1)
        out = F.tanh(self.fc(x))
        return out


class New_Net(nn.Module):
    def __init__(self, args):
        super(New_Net, self).__init__()
        self.args = args
        self.HGCN = HGCN(args.in_feats, args.hid_feats, args.bel_out_feats, args.head_inc_out_feats)
        if args.BiLSTM==True:
            self.extractor = BiLSTM()
        self.encoder = Encoder()
        # self.predictor = Predictor()
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = args.norm
        self.scale = args.scale
        # self.mlp_mean = MLP_M_V(256, 256)                 #  286,286
        # self.mlp_var = MLP_M_V(256, 256)                  #  286,286
        # self.mlp_threshold = MLP_T(args.feats_dim, args.feats_dim // 2, 1)
        self.mlp_threshold = MLP_T(256, 1)                           #  286, 1
        self.criterion = nn.BCELoss()
        # self.TransformNN = TransformNN()
        if args.DNABERT==True:
            self.pretrain_path = args.pretrain_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_path, trust_remote_code=True)
            self.Bert_model = AutoModel.from_pretrained(self.pretrain_path, trust_remote_code=True)
            self.bert_init()

    def attention(self, features, prototypes):
        attention_weights = torch.softmax(torch.mm(features, prototypes.T), dim=1)
        output = torch.mm(attention_weights, prototypes)
        return output

    def bert_init(self):
        all_layers = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6',
                      'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler']
        if self.args.freeze_bert:
            unfreeze_layers = all_layers[self.args.freeze_layer_num + 1:]
            for name, param in self.Bert_model.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break


    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, graph, tag):

        target = graph.nodes['sequence'].data['label']
        sequence_feature = graph.nodes['sequence'].data['feature']
        label_features = graph.nodes['label'].data['feature']


        node_features = {'sequence': sequence_feature,
                         'label': label_features}


        H = self.HGCN(graph, node_features)

        if self.args.merge_train==True:
            proto = H['label']
            target = target
        else:
            if tag == 'h':
                proto = H['label'][:self.args.head_class_num]  # 0,1,2
                target = target[:, :self.args.head_class_num]
            else:  # tag == 't'
                proto = H['label'][self.args.head_class_num:]  # 3,4
                target = target[:, self.args.head_class_num:]

        if self.args.BiLSTM == True:
            x = self.extractor(sequence_feature)
            if self.norm:
                x = self.l2_norm(x)
            if self.scale:
                x = self.s * x
            if self.args.Encoder == True:
                repre = self.encoder(x)
            elif self.args.Encoder == False:
                repre = x
        elif self.args.Encoder == True:
            repre = self.encoder(sequence_feature)
            if self.norm:
                repre = self.l2_norm(repre)
            if self.scale:
                repre = self.s * repre
        else:
            repre = sequence_feature


        positive_samples, negative_samples = extract_samples(target, repre)

        kl_divs = []
        for pos_samples, neg_samples in zip(positive_samples, negative_samples):

            pos_mean, pos_var = compute_mean_var_by_eq(pos_samples)
            neg_mean, neg_var = compute_mean_var_by_eq(neg_samples)

            kl_div = kl_divergence(pos_mean, pos_var, neg_mean, neg_var)
            kl_divs.append(kl_div)

        kl_tensor = torch.stack(kl_divs)
        threshold_pred = torch.sigmoid(self.mlp_threshold(kl_tensor).transpose(0, 1))

        pre_proba = calculate_proba(proto, repre)

        pre_proba_d = pre_proba.detach()
        smooth_pre_target = torch.sigmoid((pre_proba_d - threshold_pred) * 100)


        loss_cla = self.criterion(pre_proba, target)
        loss_th = self.criterion(smooth_pre_target, target)
        loss_con = info_nce_loss(repre, proto, target, tau=0.3)

        if tag == 'h':
            loss = 0.5 * loss_cla + 0.3 * loss_con + 0.2 * loss_th
            # loss = 0.7 * loss_cla  + 0.3 * loss_th
        else:
            loss = 0.5 * loss_cla + 0.3 * loss_con + 0.2 * loss_th
            # loss = 0.7 * loss_cla  + 0.3 * loss_th
        return loss, threshold_pred, proto, repre, target





