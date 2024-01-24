import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
import scipy.io as sio

class Data(object):
    def __init__(self, dataset_name, adj, adjZ, adj_nonorm, edge_index, features, labels, train_mask, val_mask, test_mask):
        self.dataset_name = dataset_name
        self.adj = adj
        self.adjZ = adjZ
        self.adj_nonorm = adj_nonorm
        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1

    def to(self, device):
        self.adj = self.adj.to(device)
        self.adjZ = self.adjZ.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)


def load_data(dataset_str, ntrain=None, seed=None, repeat=0):
    if dataset_str in ['cora', 'citeseer']:
        data = load_planetoid_data(dataset_str)
    elif dataset_str in ['amaphoto', 'amacomp']:
        data = load_amazon_data(dataset_str, ntrain, seed)
    else:
        raise ValueError("Dataset {0} does not exist".format(dataset_str))
    return data

def load_amazon_data(dataset_str, ntrain, seed):
    with np.load('data/amazon/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)

    feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                shape=loader['attr_shape']).todense()
    features = torch.tensor(feature_mat)

    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    edges = [(u, v) for u, v in zip(adj_mat.row.tolist(), adj_mat.col.tolist())]
    G = nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))
    G.add_edges_from(edges)

    edges = torch.tensor([[u, v] for u, v in G.edges()]).t()
    edge_list = torch.cat([edges, torch.stack([edges[1], edges[0]])], dim=1)
    edge_list = add_self_loops(edge_list, loader['adj_shape'][0])

    adj = normalize_adj(edge_list)
    adjZ = normalize_adj2Z(edge_list)
    adj_nonorm = edge2adj((edge_list))

    labels = loader['labels']
    labels = torch.tensor(labels).long()

    # load fixed train nodes
    if ntrain is None:
        with np.load('data/amazon/' + dataset_str + '_mask.npz', allow_pickle=True) as masks:
            train_idx, val_idx, test_idx = masks['train_idx'], masks['val_idx'], masks['test_idx']
            train_mask = index_to_mask(train_idx, labels.size(0))
        val_mask = index_to_mask(val_idx, labels.size(0))
        test_mask = index_to_mask(test_idx, labels.size(0))
    else:
        train_mask, val_mask, test_mask = split_data(labels, ntrain, 500, seed)

    data = Data(dataset_str, adj, adjZ, adj_nonorm, edge_list, features, labels, train_mask, val_mask, test_mask)

    return data

def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1), dtype=torch.int)
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)
    adjZ = normalize_adj2Z(edge_list)
    adj_nonorm = edge2adj(edge_list)

    # labeled_num = int(features.shape[0]*0.5)
    # train_idx = torch.arange(labeled_num, dtype=torch.long)
    # val_idx = torch.arange(labeled_num, labeled_num + 500, dtype=torch.long)
    # test_idx = torch.arange(labeled_num+500, features.shape[0], dtype=torch.long)

    train_mask = index_to_mask(train_idx, labels.shape[0])
    val_mask = index_to_mask(val_idx, labels.shape[0])
    test_mask = index_to_mask(test_idx, labels.shape[0])

    data = Data(dataset_str, adj, adjZ, adj_nonorm, edge_list, features, labels, train_mask, val_mask, test_mask)

    return data

def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def split_data(labels, n_train_per_class, n_val, seed):
    np.random.seed(seed)
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask

def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list

def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg

def normalize_adj(edge_list):
    # D-0.5 A D-0.5
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj

def normalize_adj2Z(edge_list):
    # D0.5 A D-0.5
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt_right = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt_right[deg_inv_sqrt_right == float('inf')] = 0.0
    deg_inv_sqrt_left = torch.pow(deg.to(torch.float), 0.5)
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt_left[row] * weight * deg_inv_sqrt_right[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj

def edge2adj(edge_list):
    v = torch.ones(edge_list.size(1))
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx,1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    mx = mx*r_inv.unsqueeze(1)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj_coo(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
