import argparse
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from Popi import *

class JSSP(ElementwiseProblem):
    def __init__(self, ins_data, **kwargs):
        n, m, PT, MT, ni, mm = self.jssp_data_trans(ins_data)  # TODO
        args = self.get_args(n, m, PT, MT, ni, mm)
        self.args = args
        self.JS = Job_shop(args)
        self.PT = PT
        self.MT = MT
        self.num_ope = sum(ni)
        self.m = m
        self.n = n
        self.nums_ope = ni
        self.mm = mm

        self.os_list = []
        for i in range(len(self.nums_ope)):
            self.os_list.extend([i for _ in range(self.nums_ope[i])])

        # self.ms_list = []
        # self.J_site = []  # 方便后面的位置查找
        # for i in range(len(self.MT)):
        #     for j in range(len(self.MT[i])):
        #         self.ms_list.append(len(self.MT[i][j]))
        #         self.J_site.append((i, j))  # 保存工件序号和工序相对序号

        self.l1 = self.num_ope

        super(JSSP, self).__init__(
            n_var=sum(ni),
            n_obj=1,
            vtype=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # out['F'] = self.get_route_length(x)
        self.JS.reset()
        C_max = self.decode(x)
        out['F'] = C_max

    def decode(self, CHS):
        self.JS.reset()
        for i in CHS:
            self.JS.decode(i, 0)
        return self.JS.C_max

    def jssp_data_trans(self, ins_data):
        ins_proc_data = ins_data[0]
        ope_num_list = ins_data[1]
        end_bias_list = np.cumsum(ope_num_list)
        bias_list = end_bias_list - ope_num_list

        n = len(ope_num_list)
        m = len(ins_proc_data[0])
        ni = ope_num_list

        PT = [[[] for _ in range(ni[i])] for i in range(n)]
        MT = [[[] for _ in range(ni[i])] for i in range(n)]
        for i in range(n):
            for j in range(ni[i]):
                for k in range(m):
                    if ins_proc_data[bias_list[i]+j][k] > 0:
                        MT[i][j].append(k + 1)
                        PT[i][j].append(ins_proc_data[bias_list[i]+j][k])
                        break

        mm = 1
        return n, m, PT, MT, ni, mm

    def get_args(self, n, m, PT, MT, ni, means_m=1):
        parser = argparse.ArgumentParser()
        # params for FJSPF:
        parser.add_argument('--n', default=n, type=int, help='job number')
        parser.add_argument('--m', default=m, type=int, help='Machine number')
        parser.add_argument('--O_num', default=ni, type=list, help='Operation number of each job')
        parser.add_argument('--Processing_Machine', default=MT, type=list, help='processing machine of operations')
        parser.add_argument('--Processing_Time', default=PT, type=list, help='processing machine of operations')
        parser.add_argument('--means_m', default=means_m, type=float, help='avaliable machine')

        args = parser.parse_args()
        return args



