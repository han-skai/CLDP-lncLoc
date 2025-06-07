
import argparse
import time
from utils import *
from frame import *
from get_graph import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--all_class_num', default=5, type=int, metavar='N',
                    help='the number of head  and tail all class')
parser.add_argument('--head_class_num', default=5, type=int, metavar='N',
                    help='the number of head class')

parser.add_argument('-fp', '--fea_path', default=r'../data/K562/lnc_K562_3220_to_1697_KRTCPCFL286d_header.csv',type=str, metavar='PATH',help='path of feature path')
parser.add_argument('-lp', '--lab_path', default=r'../data/K562/lnc_K562_3220_to_k562_1697_seq_label.csv',type=str, metavar='PATH', help='path of label path')
parser.add_argument('-cp', '--con_path', default=r'../data/K562/b0.1_t0.75_100993e_1697s_connection_matrix.csv', type=str, metavar='PATH',help='path of connection_matrix path')
parser.add_argument('-pp', '--pretrain_path', default=r'/home/ubuntu/han_skai/model-master/DNABERT-2-117M',type=str, metavar='PATH',help='path of DNA_BERT_2 model')
parser.add_argument('-slp', '--label_seq_path', default=r'../data/K562/lnc_K562_3220_to_k562_1697_seq_label.csv',type=str, metavar='PATH',help='path of connection_matrix path')
parser.add_argument('--freeze_bert', default=True, help='freeze_bert or not')
parser.add_argument('--freeze_layer_num', default=8, type=int, help='freeze_layer_num of DNABERT model')
# ++++++++++++ proto train ++++++++++++++++++
parser.add_argument('--feats_dim', default=286, type=int, help='Input features dimensions')
parser.add_argument('--label_init_dim', default=286, type=int, help='Input labels dimensions')
parser.add_argument('--in_feats', default=286, type=int, help='Input features dimension')
parser.add_argument('--hid_feats', default=512, type=int, help='Hidden features dimension')
parser.add_argument('--bel_out_feats', default=256, type=int, help='Belief output features dimensions')
parser.add_argument('--HGCN_proto_epochs', default=300, type=int, help='Number of epochs for proto training')

parser.add_argument('--Meta_proto_do', default=True, help='training threshold...')
parser.add_argument('--Meta_proto_epochs', default=60, type=int, help='Metapath2vec proto dimensions')
parser.add_argument('--Meta_proto_dim', default=128, type=int, help='Metapath2vec proto dimension')

# ++++++++++++threshold++++++++++++++++++++++++
parser.add_argument('--merge_train', default=True, help='threshold of head and tail class merge training')
parser.add_argument('--DNABERT', default=False, help='Enable DNABERT to get embeddings')
parser.add_argument('--BiLSTM', default=False, help='Enable BiLSTM')
parser.add_argument('--Encoder', default=True, help='Enable Encoder')
parser.add_argument('--norm', default=True, help='Enable normalization')
parser.add_argument('--scale', default=True, help='Enable scaling')

parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",)
parser.add_argument('--alpha', type=float, default=0.8, help='Importance of positive and negative samples')
parser.add_argument('--gamma', type=float, default=0, help='Focus parameter for focal loss')
parser.add_argument('--dis', type=str, choices=['euclidean', 'cosine', 'manhattan'], default='cosine',
                    help='Distance calculation method')
parser.add_argument('--h_fai', type=float, default=0.6, help='Importance of classification loss of head class')
parser.add_argument('--t_fai', type=float, default=0.7, help='Importance of classification loss of tail class')
parser.add_argument('--F_l', default=False, help='Whether or not to use focal_loss')
parser.add_argument('--val_th', default=False, help='learning threshold in validation')

# ++++++++++++++++++head_tail_train++++++++++++++++++++++++++++++++++++++++
parser.add_argument('--fold', default=False, help='Perform k-fold cross-validation')
parser.add_argument('--K', default=5, type=int, help='number of k-fold to run')

parser.add_argument('--single_split', default=True, action='store_true',
                    help='Perform single train-test-validation split instead of k-fold')
parser.add_argument('--split-ratio',type=float, nargs=3, default=[0.8, 0.1, 0.1],help="Specify the split ratio for train, validation, and test datasets. Default is [0.8, 0.1, 0.1].")

parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
parser.add_argument('--head_epochs', type=int, default=200, help="head_tasks")
parser.add_argument('--h_lr', '--head_learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')   
parser.add_argument('--h_step_size', default=2800, type=int, metavar='N', help='run step_size steps to adjust lr')

parser.add_argument('--tail_epochs', type=int, default=1000, help="tail_tasks")
parser.add_argument('--t_lr', '--tail_learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate') 
parser.add_argument('--t_step_size', default=2800, type=int, metavar='N', help='run step_size steps to adjust lr')

parser.add_argument('--head_inc_out_feats', default=256, type=int, help='HGCN sequences embedding dimensions')
parser.add_argument('--scData_enable', default=True, help='whether ues single cell data nor not')

args = parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()

    if args.single_split == True:
        seq_train, lbl_train, feat_train, seq_val, lbl_val, feat_val, seq_test, lbl_test, feat_test, con_matrix_train, train_index, val_index, test_index = read_tra_val_tes(args)

        Global_g = get_global_graph(args, lbl_train, feat_train, con_matrix_train, args.label_init_dim)

        H_best_model_state, H_best_proto, H_best_threshold, repre, target = head_train(Global_g, seq_val, lbl_val, feat_val, args, tag='tra-val-tes')


        if args.merge_train == False:
            T_best_model_state, T_best_proto, T_best_threshold = tail_train(Global_g, seq_val, lbl_val, feat_val, args, tag='tra-val-tes')
            test(H_best_model_state, H_best_proto, H_best_threshold, T_best_model_state, T_best_proto, T_best_threshold,
                 seq_test, lbl_test, feat_test, args, tag='tra-val-tes')

        else:
            test_merge(H_best_model_state, H_best_proto, H_best_threshold, seq_test, lbl_test, feat_test, args, tag='tra-val-tes')


    if args.fold == True:
        folds = read_Kflod(args)

        metrics_list = []
        i = 1
        for seq_train, lbl_train, feat_train, seq_val, lbl_val, feat_val, con_matrix_train, train_index, val_index in folds:

            print(f"####################################  fold {i}  ########################################")
            i = i + 1

            Global_g = get_global_graph(args, lbl_train, feat_train, con_matrix_train, args.label_init_dim)

            H_best_model_state, H_best_proto, H_best_threshold = head_train(Global_g, seq_val, lbl_val, feat_val, args, tag='fold')

            T_best_model_state, T_best_proto, T_best_threshold = tail_train(Global_g, seq_val, lbl_val, feat_val, args, tag='fold')

            ap, hl, rl, oe, cov, acc, macro_auroc, macro_auprc, macro_mcc, F1 = \
                test(H_best_model_state, H_best_proto, H_best_threshold,
                     T_best_model_state, T_best_proto, T_best_threshold,
                     seq_val, lbl_val, feat_val, args, tag='fold')


            metrics_list.append({
                'ap': ap,
                'hl': hl,
                'rl': rl,
                'oe': oe,
                'cov': cov,
                'acc': acc,
                'macro_auroc': macro_auroc,
                'macro_auprc': macro_auprc,
                'macro_mcc': macro_mcc,
                'F1': F1
            })


        average_metrics = {key: sum(d[key] for d in metrics_list) / len(metrics_list) for key in metrics_list[0].keys()}

        print("Average Metrics over K-Folds:")
        for metric, value in average_metrics.items():
            print(f"{metric}: {value:.4f}")



    end_time = time.time()


    elapsed_time = end_time - start_time

    print(f"run time：{elapsed_time} s")
    print(f"run time：{elapsed_time/60} min")
    print(f"run time：{elapsed_time/3600} h")
