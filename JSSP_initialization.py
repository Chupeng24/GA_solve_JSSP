
import numpy as np
import random
import copy
def JSSP_initial(problem, pop_size):
    Pop = []
    for i in range(int(pop_size)):
        random.shuffle(problem.os_list)
        Pop.append(copy.copy(problem.os_list))

    return np.array(Pop)