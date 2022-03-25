import math
from pickle import FALSE

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import time
import pybind_11_example as pbe



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #break down input in chunks INFORMATION FOR MATRIX MULTIPLICATION
        # print("\n\nX-input: " + str(len(input)) + " x " + str(len(input[0])))
        # print("W-weight: " + str(len(self.weight)) + " x " + str(len(self.weight[0])))
        # print("A-adj: " + str(len(adj)) + " x " + str(len(adj[0])))

        # input_shortent = input[0:100, 0:100]
        # weight_shortent = self.weight[0:100, 0:16]
        # input_shortent = input
        # weight_shortent = self.weight

        # pip3 install -e . -vvv   
        # start_time = time.time()
        adjnew = adj.to_dense()
        cppanswer = pbe.matrix_multi_cpp(input.tolist(), self.weight.tolist() , adjnew.tolist())
        output = torch.Tensor(cppanswer)
        # print("cpp time --- %s seconds ---" % (time.time() - start_time))
        
        torch.set_printoptions(threshold=1000000000000000000000000000000000000000000000000000)
        print(output)
        exit(0)
       
        # AB aka phase 1
        # start_time = time.time()
        # support = torch.mm(adj, input)
        # output = torch.mm(support, self.weight)
        # print("cpp time --- %s seconds ---" % (time.time() - start_time))
        # torch.set_printoptions(threshold=1000000000000000000000000000000000000000000000000000)
        # print(output)
        # exit (0)

        # #BC aka phase 2
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        
        # torch.set_printoptions(threshold=1000000000000000000000000000000000000000000000000000)
        # print(output)
        # exit(0)


        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    # print("B-support = " + "X-input(" + str(len(input)) + " x " + str(len(input[0])) + ") * W-weight(" + str(len(self.weight)) + " x " + str(len(self.weight[0])) + ")")
    # print("B-support: " + str(len(support)) + " x " + str(len(support[0])))

    # print("output =  A-adj(" + str(len(adj)) + " x " + str(len(adj[0])) + ") * B-support(" + str(len(support)) + " x " + str(len(support[0])))
    # print("output: " + str(len(output)) + " x " + str(len(output[0])))
            # start_time = time.time()
        # res = [[0 for x in range(len(weight_shortent[0]))] for y in range(len(input_shortent))] 
        # # res = torch.zeros([len(input_shortent), len(weight_shortent[0])], dtype=torch.float32)
        # # print("\n\nressize:", len(res), len(res[0]))
        # # print("lenof i:, ",len (input))
        # for i in range(len(input_shortent)):
        #     for j in range(len(weight_shortent[0])):
        #         for k in range(len(weight_shortent)):
        #             # resulted matrix
        #             res[i][j] += input_shortent[i][k] * weight_shortent[k][j]
                    # print("calculating", i, j, k)
        # print("DONE")
        # print("python time --- %s seconds ---" % (time.time() - start_time))
        # res_tensor = torch.Tensor(res)