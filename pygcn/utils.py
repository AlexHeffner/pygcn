import numpy as np
import pickle as pkl

import scipy.sparse as sp
import torch
import sys

from torch_geometric.datasets import NELL
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

# from torch_geometric.datasets import NELL
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import Data


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))
    
    new_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # print("New dataset: ", new_dataset)
    
    # loader = DataLoader(new_dataset)
    # print("Loader: ", loader)

    data_object = new_dataset[0]

    ###############################
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

###########################################
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    ####################################################

    new_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # print("New dataset: ", new_dataset)
    
    # loader = DataLoader(new_dataset)
    # print("Loader: ", loader)

    data_object = new_dataset[0]
    # print("Data Object----------------")
    # print(data_object)
    # print("")

    adj2 = sp.coo_matrix((np.ones(data_object.edge_index.shape[0]), (data_object.edge_index[:, 0], data_object.edge_index[:, 1])),
                    shape=(data_object.y.shape[0], data_object.y.shape[0]),
                    dtype=np.float32)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    adj2 = normalize(adj2 + sp.eye(adj2.shape[0]))
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)

    # print("----lables----")
    # torch.set_printoptions(threshold=10_000)
    # print(data_object.y)
    # print(labels)

    # print("len of each: ",data_object.y.shape, " ", labels.shape)
    # if(torch.equal(data_object.y,labels)):
    #     print("SAME")
    # else:
    #     print("Not Same")
    # print('')


    # print("adj---------------")
    # torch.set_printoptions(threshold=10_000)
    # print(adj2)
    # torch.set_printoptions(threshold=100_000)
    # print(adj)
    # print("len of each: ",adj2.shape, " ", adj.shape)
    # # if(torch.equal(adj2,adj)):
    # #     print("SAME")
    # # else:
    # #     print("Not Same")
    # # print('')

    # print("----Featurs----")
    torch.set_printoptions(threshold=1000000000000000000)
    # print(data_object.x)
    # print(features)
    # print("len of each: ",data_object.x.shape, " ", features.shape)
    # if(torch.equal(data_object.x,features)):
    #     print("SAME")
    # else:
    #     print("Not Same")
    # print('')

    # exit(0)

    # adj = new_dataset[0].edge_index
    # features = new_dataset[0].x
    # labels = new_dataset[0].y

    # x = features
    # y = lable ??? possibly
    #####################################################################  
    # return adj2, data_object.x , data_object.y, idx_train, idx_val, idx_test
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
