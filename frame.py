import torch

from model import HGCN, MLP_M_V, MLP_T, New_Net
import torch.nn as nn
from tqdm import tqdm
from utils import extract_samples, compute_mean_var, kl_divergence,distances
import torch.nn.functional as F
from metric import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from get_graph import *
from torch.optim.lr_scheduler import LambdaLR
from utils import *


def head_train(Global_g, seq_val,lbl_val, feat_val, args, tag):

    model_h = New_Net(args)
    optimizer = torch.optim.AdamW(model_h.parameters(), lr=args.h_lr, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.h_step_size, gamma=0.1)

    # tran_optimizer = torch.optim.Adam(model_h.transfer.parameters(), lr=0.001, betas=(0.9, 0.99))

    best_ap = 0
    best_acc = 0
    best_ap_add_acc = 0
    best_model_state = None
    best_proto = torch.Tensor()
    best_threshold_pred = torch.Tensor()

    for epoch in tqdm(range(args.head_epochs), desc="Head Training Progress"):

        model_h.train()
        scheduler.step()

        loss, threshold_pred, proto, repre, target = model_h(Global_g, tag='h')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if args.merge_train==True:

            tqdm.write(f' epoch {epoch}-th  train loss: {loss.item()}')
        else:
            tqdm.write(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ epoch {epoch}-th  head class train loss: {loss.item()}')

        if tag=='tra-val-tes':
            ap, acc, val_loss = validation(model_h, threshold_pred.detach(), proto.detach(), seq_val, lbl_val, feat_val, args, epoch, tag='h')

            print(f'######################################################################## head epoch {epoch}-th val loss: {val_loss.item()}')

            if ap + acc > best_ap_add_acc:
                best_ap = ap
                best_acc = acc
                best_ap_add_acc = ap + acc
                best_model_state = model_h.state_dict()
                best_proto = proto
                best_threshold_pred = threshold_pred
                print(f'New best model saved with AP: {ap} and ACC: {acc}')


            print(f'Final best model with AP: {best_ap} and ACC: {best_acc}')

        if tag=='fold':
            best_model_state = model_h.state_dict()
            best_proto = proto
            best_threshold_pred = threshold_pred

    return best_model_state, best_proto, best_threshold_pred, repre, target


def tail_train(Global_g, seq_val, lbl_val, feat_val, args, tag):

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ tail class training ++++++++++++++++++++++++++++++++++++++++")

    model_t = New_Net(args)
    optimizer = torch.optim.AdamW(model_t.parameters(), lr=args.t_lr, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.t_step_size, gamma=0.1)

    best_ap = 0
    best_acc = 0
    best_ap_add_acc = 0
    best_model_state = None
    best_proto = torch.Tensor()
    best_threshold_pred = torch.Tensor()

    for epoch in tqdm(range(args.tail_epochs), desc="Tail Training Progress"):
        model_t.train()
        scheduler.step()

        loss, threshold_pred, proto = model_t(Global_g, tag='t')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm.write(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ tail class train loss: {loss.item()}')

        if tag=='tra-val-tes':
            ap, acc, val_loss = validation(model_t, threshold_pred.detach(), proto.detach(), seq_val, lbl_val, feat_val, args, epoch, tag='t')

            print(
                f'####################################################################### tail epoch {epoch}-th val loss: {val_loss.item()}')
            if ap + acc > best_ap_add_acc:
                best_ap = ap
                best_acc = acc
                best_ap_add_acc = ap + acc
                best_model_state = model_t.state_dict()
                best_proto = proto
                best_threshold_pred = threshold_pred
                # torch.save(best_model_state, f'best_model.pth')
                print(f'New best model saved with AP: {ap} and ACC: {acc}')

            print(f'Final best model with AP: {best_ap} and ACC: {best_acc}')

        if tag == 'fold':
            best_model_state = model_t.state_dict()
            best_proto = proto
            best_threshold_pred = threshold_pred

    return best_model_state, best_proto, best_threshold_pred


def validation(model, threshold, proto, seq_val, lbl_val, feat_val, args, epoch, tag):

    model.eval()
    criterion = nn.BCELoss()  # reduction='sum'
    if args.merge_train==True:
        class_num = args.all_class_num
    else:
        if tag=='h':
            class_num = args.head_class_num
        else:
            class_num = args.all_class_num - args.head_class_num

    F1 = np.zeros(class_num)

    acc, ap = 0, 0

    with torch.no_grad():

        if args.DNABERT==True:
            embedding = []
            for d in tqdm(seq_val, desc="seq_val get DNABERT embeddings"):
                # print('the len of current data:{}'.format(len(d)))
                inputs = model.tokenizer(d, return_tensors='pt')["input_ids"]
                hidden_states = model.Bert_model(inputs)[0]  # [1, sequence_length, 768]

                embedding_mean = torch.mean(hidden_states[0], dim=0)
                embedding.append(embedding_mean)

            input = torch.stack(embedding)
        else:
            input = torch.tensor(feat_val.values).float()


        if args.merge_train==True:
            target = torch.tensor(lbl_val.iloc[:, :].values)

        else:
            if tag=='h':
                target = torch.tensor(lbl_val.iloc[:, :args.head_class_num].values)
            else:
                target = torch.tensor(lbl_val.iloc[:, args.head_class_num:].values)

        if args.BiLSTM==True:
            input = model.extractor(input)
            if args.norm:
                repre = model.l2_norm(input)
            if args.scale:
                repre = model.s * repre
            if args.Encoder == True:
                repre = model.encoder(input)
            elif args.Encoder == False:
                repre = repre
        elif args.Encoder==True:
            repre = model.encoder(input)
            if args.norm:
                repre = model.l2_norm(repre)
            if args.scale:
                repre = model.s * repre
        else:
            repre = input



        pre_proba = calculate_proba(proto, repre)


        val_loss = criterion(pre_proba, target.float())


        ap += average_precision(pre_proba, target)


        pre_target = (pre_proba > threshold).float()

        for l in range(class_num):
            F1[l] += f1_score(target[:, l], pre_target[:, l], average='binary')

        acc += accuracy(pre_target, target)

        macro_auroc = roc_auc_score(target, pre_proba, average='macro')


        macro_auprc = average_precision_score(target, pre_proba, average='macro')


        macro_mcc = np.mean([matthews_corrcoef(target[:, i], pre_target[:, i]) for i in range(target.shape[1])])


        print('validation the result of F1: \n', F1)
        print("validation AP: %.4f, ACC: %.4f" % (ap, acc))
        print("Macro-AUROC: %.4f, Macro-AUPRC: %.4f, Macro-MCC: %.4f" % (macro_auroc, macro_auprc, macro_mcc))

    return ap, acc, val_loss


def test(H_best_model_state,H_best_proto, H_best_threshold, T_best_model_state, T_best_proto, T_best_threshold, seq_test, lbl_test, feat_test, args, tag):


    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ test ++++++++++++++++++++++++++++++++++++++++")
    model_h = New_Net(args)
    model_t = New_Net(args)
    model_h.load_state_dict(H_best_model_state)
    model_t.load_state_dict(T_best_model_state)


    model_h.eval()
    # model_t.eval()

    F1 = np.zeros(args.all_class_num)
    acc, ap,rl, = 0, 0, 0
    with torch.no_grad():

        if args.DNABERT==True:
            embedding = []
            for d in tqdm(seq_test, desc="seq_val get DNABERT embeddings"):
                # print('the len of current data:{}'.format(len(d)))
                inputs = model_h.tokenizer(d, return_tensors='pt')["input_ids"]
                hidden_states = model_h.Bert_model(inputs)[0]  # [1, sequence_length, 768]

                embedding_mean = torch.mean(hidden_states[0], dim=0)
                embedding.append(embedding_mean)

            input = torch.stack(embedding)
        else:
            input = torch.tensor(feat_test.values).float()

        target = torch.tensor(lbl_test.values)

        if args.BiLSTM==True:
            input_h = model_h.extractor(input)  # BiLSTM
            input_t = model_t.extractor(input)  # BiLSTM
            if args.norm:
                repre_h = model_h.l2_norm(input_h)
                repre_t = model_t.l2_norm(input_t)
            if args.scale:
                repre_h = model_h.s * repre_h
                repre_t = model_t.s * repre_t
            if args.Encoder == True:
                repre_h = model_h.encoder(repre_h)  # Encoder
                repre_t = model_t.encoder(repre_t)  # Encoder

            elif args.Encoder == False:
                repre_h = repre_h
                repre_t = repre_t

        elif args.Encoder == True:
            repre_h = model_h.encoder(input)  # Encoder
            repre_t = model_t.encoder(input)

            if args.norm:
                repre_h = model_h.l2_norm(repre_h)
                repre_t = model_t.l2_norm(repre_t)
            if args.scale:
                repre_h = model_h.s * repre_h
                repre_t = model_t.s * repre_t

        else:
            repre_h = input
            repre_t = input


        pre_proba_h = calculate_proba(H_best_proto, repre_h)
        pre_proba_t = calculate_proba(T_best_proto, repre_t)

        pre_proba =  torch.cat((pre_proba_h, pre_proba_t), dim=1)

        all_best_threshold = torch.cat((H_best_threshold, T_best_threshold), dim=1)

        ap += average_precision(pre_proba, target)



        pre_target = (pre_proba > all_best_threshold).float()

        for l in range(args.all_class_num):
            F1[l] += f1_score(target[:, l], pre_target[:, l], average='binary')

        acc += accuracy(pre_target, target)

        # macro_auroc = 0
        # Calculate Macro-AUROC
        macro_auroc = roc_auc_score(target, pre_proba, average='macro')

        # Calculate Macro-AUPRC
        macro_auprc = average_precision_score(target, pre_proba, average='macro')

        # Calculate Macro-MCC
        macro_mcc = np.mean([matthews_corrcoef(target[:, i], pre_target[:, i]) for i in range(target.shape[1])])

        if tag=='tra-val-tes':

            print('target:', target[-50:, :])
            print('pre_proba:', pre_proba[-50:, :])
            print('all_best_threshold:', all_best_threshold)

            print('test the result of F1: \n', F1)
            print("test AP: %.4f, ACC: %.4f" % (ap, acc))
            print("Macro-AUROC: %.4f, Macro-AUPRC: %.4f, Macro-MCC: %.4f" % (macro_auroc, macro_auprc, macro_mcc))

        if tag=='fold':
            return ap, acc, macro_auroc, macro_auprc, macro_mcc, F1


def test_merge(M_best_model_state, M_best_proto, M_best_threshold, seq_test, lbl_test, feat_test, args, tag):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ test++++++++++++++++++++++++++++++++++++++++")
    model_m = New_Net(args)
    model_m.load_state_dict(M_best_model_state)

    model_m.eval()
    # model_t.eval()

    F1 = np.zeros(args.all_class_num)
    ap, acc = 0, 0,
    with torch.no_grad():

        if args.DNABERT==True:
            embedding = []
            for d in tqdm(seq_test, desc="seq_val get DNABERT embeddings"):

                inputs = model_m.tokenizer(d, return_tensors='pt')["input_ids"]
                hidden_states = model_m.Bert_model(inputs)[0]  # [1, sequence_length, 768]

                embedding_mean = torch.mean(hidden_states[0], dim=0)
                embedding.append(embedding_mean)

            input = torch.stack(embedding)
        else:
            input = torch.tensor(feat_test.values).float()

        target = torch.tensor(lbl_test.values)

        if args.BiLSTM==True:
            input_h = model_m.extractor(input)  # BiLSTM
            if args.norm:
                repre_m = model_m.l2_norm(input_h)
            if args.scale:
                repre_m = model_m.s * repre_m
            if args.Encoder == True:
                repre_m = model_m.encoder(repre_m)  # Encoder

            elif args.Encoder == False:
                repre_m = repre_m


        elif args.Encoder == True:
            repre_m = model_m.encoder(input)  # Encoder

            if args.norm:
                repre_m = model_m.l2_norm(repre_m)
            if args.scale:
                repre_m = model_m.s * repre_m

        else:
            repre_m = input

        pre_proba = calculate_proba(M_best_proto, repre_m)


        all_best_threshold = M_best_threshold

        ap += average_precision(pre_proba, target)



        pre_target = (pre_proba > all_best_threshold).float()

        for l in range(args.all_class_num):
            F1[l] += f1_score(target[:, l], pre_target[:, l], average='binary')

        acc += accuracy(pre_target, target)

        # macro_auroc = 0
        # Calculate Macro-AUROC
        macro_auroc = roc_auc_score(target, pre_proba, average='macro')

        # Calculate Macro-AUPRC
        macro_auprc = average_precision_score(target, pre_proba, average='macro')

        # Calculate Macro-MCC
        macro_mcc = np.mean([matthews_corrcoef(target[:, i], pre_target[:, i]) for i in range(target.shape[1])])

        if tag=='tra-val-tes':

            print('target:', target[-50:, :])
            print('pre_proba:', pre_proba[-50:, :])
            print('all_best_threshold:', all_best_threshold)

            print('test the result of F1: \n', F1)
            print("test AP: %.4f,  ACC: %.4f" % (ap, acc))
            print("Macro-AUROC: %.4f, Macro-AUPRC: %.4f, Macro-MCC: %.4f" % (macro_auroc, macro_auprc, macro_mcc))

        if tag=='fold':
            return ap,  acc, macro_auroc, macro_auprc, macro_mcc, F1
