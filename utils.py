import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import torch
import numpy as np
import dgl
import torch.nn.functional as F

def read_data(DNABERT, label_seq_path, features_file, labels_file, k, single_split):
    if DNABERT==True:
        feat_df = pd.read_csv(features_file, header=None)
        df = pd.read_csv(label_seq_path, header=0)
        sequences_df = df.iloc[:, 10]
        labels_df = df.iloc[:, :5]
        combined_df = pd.concat([sequences_df, labels_df], axis=1)
        if single_split == True:
            indices = np.arange(combined_df.shape[0])
            train_index, test_index = train_test_split( indices, test_size=1 / k, random_state=42)
            train_df = combined_df.iloc[train_index]
            test_df = combined_df.iloc[test_index]

            train_feat_df = feat_df.iloc[train_index]
            test_feat_df = feat_df.iloc[test_index]

            # Separate sequences and labels for train and test sets
            train_sequences = train_df.iloc[:, 0]
            train_labels = train_df.iloc[:, 1:]
            test_sequences = test_df.iloc[:, 0]
            test_labels = test_df.iloc[:, 1:]
            folds = [(train_feat_df, test_feat_df, train_sequences, test_sequences, train_labels, test_labels, train_index, test_index)]
        else:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            folds = []
            for fold, (train_index, test_index) in enumerate(kf.split(combined_df)):
                train_df = combined_df.iloc[train_index]
                test_df = combined_df.iloc[test_index]

                train_feat_df = feat_df.iloc[train_index]
                test_feat_df = feat_df.iloc[test_index]

                # Separate sequences and labels for train and test sets
                train_sequences = train_df.iloc[:, 0]
                train_labels = train_df.iloc[:, 1:]
                test_sequences = test_df.iloc[:, 0]
                test_labels = test_df.iloc[:, 1:]
                folds.append((train_feat_df, test_feat_df, train_sequences, test_sequences, train_labels, test_labels, train_index, test_index))
        return folds

    else:

        features_df = pd.read_csv(features_file, header=None)
        labels_df = pd.read_csv(labels_file, header=0)
        labels_df  = labels_df.iloc[:, :5]

        # Convert DataFrames to numpy arrays
        X = features_df.values
        y = labels_df.values

        if single_split:
            indices = np.arange(X.shape[0])
            X_train, X_val, y_train, y_val, train_index, val_index = train_test_split(
                X, y, indices, test_size=1 / k, random_state=42)

            train_sequences = []
            test_sequences = []
            folds = [(X_train, X_val, train_sequences, test_sequences,  y_train, y_val, train_index, val_index)]
        else:
            kf = KFold(n_splits=k, shuffle=False)
            train_sequences = []
            test_sequences = []
            folds = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                folds.append((X_train, X_val, train_sequences, test_sequences, y_train, y_val, train_index, val_index))
        return folds

def read_tra_val_tes(args):

    df = pd.read_csv(args.label_seq_path, header=0)
    labels_df = df.iloc[:, :args.all_class_num]

    features_df = pd.read_csv(args.fea_path, header=0)

    sequences_df = pd.DataFrame(np.random.rand(df.shape[0], 1), columns=['A'])
    # sequences_df = df.iloc[:, 10]

    if args.scData_enable==True:

        con_matrix = pd.read_csv(args.con_path, header=None)
    else:
        con_matrix = pd.DataFrame(np.random.rand(df.shape[0], df.shape[0]))

    train_ratio, val_ratio, test_ratio = args.split_ratio

    seq_train, seq_temp, lbl_train, lbl_temp = train_test_split(
        sequences_df, labels_df, test_size=(val_ratio + test_ratio), random_state=42)

    test_size_ratio = test_ratio / (val_ratio + test_ratio)
    seq_val, seq_test, lbl_val, lbl_test = train_test_split(
        seq_temp, lbl_temp, test_size=test_size_ratio, random_state=42)


    train_index = seq_train.index.tolist()
    val_index = seq_val.index.tolist()
    test_index = seq_test.index.tolist()

    feat_train = features_df.iloc[train_index, :]
    feat_val = features_df.iloc[val_index, :]
    feat_test = features_df.iloc[test_index, :]

    con_matrix_train = con_matrix.loc[train_index, train_index]

    return seq_train, lbl_train, feat_train, seq_val, lbl_val, feat_val, seq_test, lbl_test, feat_test, con_matrix_train, train_index, val_index, test_index


def read_Kflod(args):
    df = pd.read_csv(args.label_seq_path, header=0)
    labels_df = df.iloc[:, :args.all_class_num]

    con_matrix = pd.read_csv(args.con_path, header=None)

    features_df = pd.read_csv(args.fea_path, header=0)


    sequences_df = pd.DataFrame(np.random.rand(df.shape[0], 1), columns=['A'])


    kf = KFold(n_splits=args.K, shuffle=True, random_state=42)


    folds = []

    for train_index, val_index in kf.split(sequences_df):
        seq_train = sequences_df.iloc[train_index]
        lbl_train = labels_df.iloc[train_index]
        seq_val = sequences_df.iloc[val_index]
        lbl_val = labels_df.iloc[val_index]


        con_matrix_train = con_matrix.loc[train_index, train_index]

        feat_train = features_df.iloc[train_index, :]
        feat_val = features_df.iloc[val_index, :]

        folds.append(
            (seq_train, lbl_train, feat_train, seq_val, lbl_val, feat_val, con_matrix_train, train_index, val_index))

    return folds




epsilon = 1e-8


def extract_samples(target, input):
    positive_samples = []
    negative_samples = []
    for i in range(target.shape[1]):
        pos_indices = (target[:, i] == 1).nonzero(as_tuple=True)[0]
        neg_indices = (target[:, i] == 0).nonzero(as_tuple=True)[0]

        positive_samples.append(input[pos_indices])
        negative_samples.append(input[neg_indices])

    return positive_samples, negative_samples

def check_for_nan_inf(data, name="data"):
    if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)):
        print(f"{name} contains NaN or Inf values")
        return True
    return False


def compute_mean_var(samples, mlp_mean, mlp_var):
    if samples.size(0) == 0:
        mean = torch.zeros(768)
        var = torch.ones(768) * epsilon
    else:
        if check_for_nan_inf(samples, "samples"):
            samples = torch.nan_to_num(samples)

        mean = mlp_mean(samples).mean(dim=0)

        if samples.size(0) > 1:
            var = mlp_var(samples).var(dim=0)
        else:
            var = torch.full_like(mean, epsilon)

    if check_for_nan_inf(mean, "mean") or check_for_nan_inf(var, "var"):
        # mean = torch.nan_to_num(mean)
        # var = torch.nan_to_num(var)
        print("var has nan value")
    return mean, var


def compute_mean_var_by_eq(samples):
    if samples.size(0) == 0:
        mean = torch.zeros(768)
        var = torch.ones(768) * epsilon
    else:
        if check_for_nan_inf(samples, "samples"):
            samples = torch.nan_to_num(samples)

        mean = samples.mean(dim=0)

        if samples.size(0) > 1:
            var = samples.var(dim=0)
        else:
            var = torch.full_like(mean, epsilon)

    if check_for_nan_inf(mean, "mean") or check_for_nan_inf(var, "var"):
        print("var has nan value")
    return mean, var


def kl_divergence(mean1, var1, mean2, var2):
    if check_for_nan_inf(mean1, "mean1") or check_for_nan_inf(var1, "var1") or check_for_nan_inf(mean2, "mean2") or check_for_nan_inf(var2, "var2"):
        mean1 = torch.nan_to_num(mean1)
        var1 = torch.nan_to_num(var1)
        mean2 = torch.nan_to_num(mean2)
        var2 = torch.nan_to_num(var2)
    kl_div = 0.5 * (torch.log(var2) - torch.log(var1) + (var1 + (mean1 - mean2) ** 2) / var2 - 1)
    if check_for_nan_inf(kl_div, "kl_div"):
        # kl_div = torch.nan_to_num(kl_div)
        print("kl_div has nan value")
    return kl_div


def distances(proto, repre, dis="euclidean"):

    if dis == "euclidean":

        distances = torch.cdist(repre, proto)
    elif dis == "cosine":

        proto_norm = F.normalize(proto, p=2, dim=1)
        repre_norm = F.normalize(repre, p=2, dim=1)
        distances = 1 - torch.mm(repre_norm, proto_norm.T)
    elif dis == "manhattan":

        distances = torch.cdist(repre, proto, p=1)
    else:
        raise ValueError("Unsupported distance type. Choose from 'euclidean', 'cosine', 'manhattan'.")


    return distances

def get_proba(proto, sequence_feature, predictor):

    pre_proba_tensors = []
    for i in range(sequence_feature.size(0)):
        sequence_feature_i = sequence_feature[i:i+1, :]

        pre_proba_one = []

        for j in range(proto.size(0)):
            proto_j = proto[j:j+1, :]

            input = torch.cat((sequence_feature_i, proto_j), dim=1)
            pre_proba = predictor(input)

            pre_proba_one.append(pre_proba)

        pre_proba_one_tensor = torch.tensor([matrix[0][0] for matrix in pre_proba_one])
        pre_proba_tensors.append(pre_proba_one_tensor)

    pre_proba_tensors = torch.stack(pre_proba_tensors)

    return pre_proba_tensors

def calculate_proba(prototypes, samples_feature):

    samples_norm = torch.norm(samples_feature, p=2, dim=1, keepdim=True)
    prototypes_norm = torch.norm(prototypes, p=2, dim=1, keepdim=True)

    similarity = torch.matmul(samples_feature, prototypes.T) / (samples_norm * prototypes_norm.T)

    proba = 1 / (1 + torch.exp(-similarity))

    return proba

