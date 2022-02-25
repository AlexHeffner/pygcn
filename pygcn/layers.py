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
        input_shortent = input
        weight_shortent = self.weight
        # print(adj[0][0])
        # exit(0)


        # support = torch.mm(input_shortent, weight_shortent)
        
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        # print("Input", input)
        # print("selfweight", self.weight)

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


        # print("----TENSORS----")
        # torch.set_printoptions(threshold=10_000)
        # pip3 install -e . -vvv   
        # a = [[1.1],[1.1],[1.1]]
        # b = [[1.1],[1.1],[1.1]]
        c = [[1.1],[1.1],[1.1]]
        # start_time = time.time()
        adjnew = adj.to_dense()
        cppanswer = pbe.matrix_multi_cpp(input_shortent.tolist(), weight_shortent.tolist() , adjnew.tolist())
        # c_adj = adj.tolist()
        # cppsupport = pbe.matrix_multi_cpp(input.tolist(), self.weight.tolist() , c)
        # cppout = pbe.matrix_multi_cpp(cppsupport, adj.tolist(), c)
        # print("cpp time --- %s seconds ---" % (time.time() - start_time))
        torch.set_printoptions(threshold=1000000000000000000000000000000000000000000000000000)
        print(torch.Tensor(cppanswer)) 
        # print(res) 
        # print(support) 
        # print(output)

        # print("len of each: ",res_tensor.shape, " ", support.shape)
        # print(torch.equal(res_tensor,support))
        # if(torch.equal(res_tensor,support)):
        #     print("SAME")
        # else:
        #     print("Not Same")

        # print('')
        exit(0)
        
        # print("B-support = " + "X-input(" + str(len(input)) + " x " + str(len(input[0])) + ") * W-weight(" + str(len(self.weight)) + " x " + str(len(self.weight[0])) + ")")
        # print("B-support: " + str(len(support)) + " x " + str(len(support[0])))

        # output = torch.spmm(adj, support)

        # print("output =  A-adj(" + str(len(adj)) + " x " + str(len(adj[0])) + ") * B-support(" + str(len(support)) + " x " + str(len(support[0])))
        # print("output: " + str(len(output)) + " x " + str(len(output[0])))
        

##########################################
        # AB aka phase 1
        # support = torch.mm(adj, input)
        # output = torch.mm(support, self.weight)

        # #BC aka phase 2
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # exit(0)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def mmAlexSlow(self, input):
        # explicit for loops
        res = [[0 for x in range(len(self.weight[0]))] for y in range(len(input))] 
        for i in range(len(input)):
            for j in range(len(self.weight[0])):
                for k in range(len(self.weight)):
                    # resulted matrix
                    res[i][j] += input[i][k] * self.weight[k][j]
        print("Here")
        return res
    # def areSameMM(A,B):
    #     for i in range(len(A)):
    #         for j in range(len(A[0])):
    #             if (A[i][j] != B[i][j]):
    #                 return 0
    #     return 1
        
