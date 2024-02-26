import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
import random

class JSSP_Mutation(Mutation):

    def __init__(self, problem, pm=0.05):

        super().__init__()
        self.prob = pm
        self.FJSP_args = problem.args
        self.half_len_chromo = len(problem.os_list)
        # self.J_site = problem.J_site

    def swap_mutation(self ,p):
        pos1 = random.randint(0, len(p) - 1)
        pos2 = random.randint(0, len(p) - 1)
        if pos1 == pos2:
            return p
        if pos1 > pos2:   # <柔性作业车间调度智能算法及应用>书中的互换变异
            pos1, pos2 = pos2, pos1
        offspring = p[:pos1] + [p[pos2]] + \
                    p[pos1 + 1:pos2] + [p[pos1]] + \
                    p[pos2 + 1:]
        return offspring

    def _do(self, problem, X, **kwargs):
        """
        :param problem:
        :param X: input sequence
        :param kwargs:
        :return: Y: sequence after mutation
        """
        Y = X.copy()
        for i, y in enumerate(X):
            if np.random.random() < self.prob:
                p11 = self.swap_mutation(list(y))
                Y[i] = np.array(p11)
        return Y

class JSSP_Crossover(Crossover):

    def __init__(self, problem, pc, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.pc = pc
        self.JSSP_args = problem.args
        self.half_len_chromo = len(problem.os_list)

    def POX(self ,p1, p2):
        jobsRange = range(0, self.JSSP_args.n)
        sizeJobset1 = random.randint(1, self.JSSP_args.n)
        jobset1 = random.sample(jobsRange, sizeJobset1)
        o1 = []
        p1kept = []
        for i in range(len(p1)):
            e = p1[i]
            if e in jobset1:
                o1.append(e)
            else:
                o1.append(-1)
                p1kept.append(e)
        o2 = []
        p2kept = []
        for i in range(len(p2)):
            e = p2[i]
            if e in jobset1:
                o2.append(e)
            else:
                o2.append(-1)
                p2kept.append(e)
        for i in range(len(o1)):
            if o1[i] == -1:
                o1[i] = p2kept.pop(0)
        for i in range(len(o2)):
            if o2[i] == -1:
                o2[i] = p1kept.pop(0)
        return o1, o2

    def _do(self, problem, X, **kwargs):
        """
            :param problem:
            :param X: input sequence
            :param kwargs:
            :return: Y: sequence after mutation
        """
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            p1, p2 = X[:, i, :]
            if random.random() < self.pc:
                p1, p2 = self.POX(p1, p2)

            Y[0, i, :] = np.array(p1)
            Y[1, i, :] = np.array(p2)

        return Y