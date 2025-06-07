import torch
import numpy as np
import dgl
import random

class LocalSubgraph:
    def __init__(self, h_g, h, seed=None):

        self.h_g = h_g
        self.h = h
        if seed is not None:
            random.seed(seed)

    def extract_subgraph(self, N, K, tag):

        if tag == 'f':
            label_nodes = self.h_g.nodes('label')[:N].tolist()
        if tag == 'a':
            label_nodes = self.h_g.nodes('label')[N:].tolist()


        nodes_to_include = {'sequence': set(), 'label': set(label_nodes)}

        for label_node in label_nodes:

            frontier = set([('label', label_node)])
            for _ in range(self.h):
                current_frontier = set()
                for node_type, node_id in frontier:
                    if node_type == 'label':
                        successors = self.h_g.successors(node_id, etype='including').tolist()
                        predecessors = self.h_g.predecessors(node_id, etype='belongs_to').tolist()
                        current_frontier.update([('sequence', nid) for nid in successors + predecessors])
                    elif node_type == 'sequence':
                        successors = self.h_g.successors(node_id, etype='connected_to').tolist()
                        predecessors = self.h_g.predecessors(node_id, etype='connected_to').tolist()
                        current_frontier.update([('sequence', nid) for nid in successors + predecessors])
                        L_successors = self.h_g.successors(node_id, etype='belongs_to').tolist()
                        L_predecessors = self.h_g.predecessors(node_id, etype='including').tolist()
                        current_frontier.update([('label', nid) for nid in L_successors + L_predecessors])

                frontier.update(current_frontier)

            frontier = [
                (node_type, node_id) for node_type, node_id in current_frontier
                if (node_type == 'sequence' and node_id not in nodes_to_include['sequence']) or
                   (node_type == 'label' and node_id not in nodes_to_include['label'])
            ]


            selected_nodes = random.sample(frontier, min(K, len(frontier)))
            for node_type, node_id in selected_nodes:
                nodes_to_include[node_type].add(node_id)


        nodes_to_include['sequence'] = list(nodes_to_include['sequence'])
        nodes_to_include['label'] = list(nodes_to_include['label'])


        sub_h_g = dgl.node_subgraph(self.h_g, nodes_to_include)

        return sub_h_g


def get_global_graph(args, labels, feat_train, connect_matrix, label_init_dim):

    # Utilize the sequence-to-label associations to build the graph
    seq_ids, label_ids = np.where(labels.values == 1)

    h_g = dgl.heterograph({
        ('sequence', 'belongs_to', 'label'): (seq_ids, label_ids),
        ('label', 'including', 'sequence'): (label_ids, seq_ids),

        ('sequence', 'connected_to', 'sequence'): ([], [])
    })

    if args.scData_enable==True:
        # Utilize the connectivity matrix to build the sequence-to-sequence relationship graph
        row_seq_ids, col_seq_ids = np.where(connect_matrix.values == 1)
        h_g.add_edges(row_seq_ids, col_seq_ids, etype=('sequence', 'connected_to', 'sequence'))


    # h_g.nodes['sequence'].data['raw_seq'] = list(samples_seqences)
    h_g.nodes['sequence'].data['label'] = torch.tensor(labels.values).float()
    h_g.nodes['sequence'].data['feature'] = torch.tensor(feat_train.values).float()
    h_g.nodes['label'].data['feature'] = torch.randn(labels.shape[1], label_init_dim).float()


    # h_g.nodes['sequence'].data['orig_id'] = torch.tensor(labels.index.values)
    # h_g.nodes['label'].data['orig_id'] = torch.arange(h_g.num_nodes('label'))

    return h_g

def get_tasks(lbl_train, feat_train, con_matrix_train, args, tag):
    Y_trn = lbl_train
    if tag=='H':
        base = Y_trn.iloc[:, :args.N]
        sample_tasks = args.head_tasks
        samples = args.H_K
    else:
        base = Y_trn.iloc[:, args.N:]
        sample_tasks = args.tail_tasks
        samples = args.T_K

    base1 = base.T

    tasks = []
    for _ in range(sample_tasks):


        base_id = []
        for _, row in base1.iterrows():

            matching_indices = row[row == 1].index.tolist()

            if len(matching_indices) >= samples:
                selected_samples = random.sample(matching_indices, samples)
            else:
                selected_samples = random.choices(matching_indices, k=samples)

            base_id.extend(selected_samples)

        selected_ids = list(np.unique(base_id))


        task_lbl = lbl_train.loc[selected_ids]
        if tag == 'H':
            task_lbl = task_lbl.iloc[:, :args.N]
        else:
            task_lbl = task_lbl.iloc[:, args.N:]
        task_feat = feat_train.loc[selected_ids]
        task_con_mat = con_matrix_train.loc[selected_ids, selected_ids]

        tasks.append((task_feat, task_lbl, task_con_mat))

    return tasks


def task_graph(task):
    task_feat, task_lbl, task_con_mat = task[0], task[1], task[2]
    seq_ids, label_ids = np.where(task_lbl.values == 1)

    row_seq_ids, col_seq_ids = np.where(task_con_mat.values == 1)

    h_g = dgl.heterograph({
        ('sequence', 'belongs_to', 'label'): (seq_ids, label_ids),
        ('label', 'including', 'sequence'): (label_ids, seq_ids),
        ('sequence', 'connected_to', 'sequence'): (row_seq_ids, col_seq_ids),
    })

    h_g.nodes['sequence'].data['label'] = torch.tensor(task_lbl.values).float()
    h_g.nodes['sequence'].data['feature'] = torch.tensor(task_feat.values).float()
    h_g.nodes['label'].data['feature'] = torch.randn(task_lbl.shape[1], 768).float()

    return h_g

