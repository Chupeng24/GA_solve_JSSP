import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from gen_order_data import gen_order_data
from job_shop_scheduling_problem import JSSP
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from JSSP_initialization import JSSP_initial
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.figure_factory as ff

pyplt = py.offline.plot
import time
import os
from JSSP_operator import *

if __name__ == "__main__":
    t1 = time.time()
    # problem
    job_list = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5
    proc_data, ope_num_list, proc_tab_array = gen_order_data(job_list)
    ins_data = (proc_data, ope_num_list)
    problem = JSSP(ins_data)
    Table = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: []}
    Task_list = []

    # initialize population
    Pop_size = 100
    X = JSSP_initial(problem, Pop_size)


    # algorithm
    algorithm = GA(
        pop_size=100,
        sampling=X,
        crossover=JSSP_Crossover(pc=0.8, problem=problem),
        mutation=JSSP_Mutation(pm=0.05, problem=problem),
        eliminate_duplicates=True)

    termination = get_termination("n_gen", 10)

    # optimize process
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    t2 = time.time()
    spend_time = t2 - t1

    F = res.F
    optim_solution = res.X
    print(f"optimized makespan: {F[0]}, spend time: {spend_time}")

    # plot res
    hist = res.history
    n_iter = []  # corresponding number of function evaluations\
    Fitness = []  # the objective space values in each generation
    for idx, algo in enumerate(hist):
        n_iter.append(idx)
        opt = algo.opt
        Fitness.append(opt.get("F")[0])
    plt.figure(figsize=(7, 5))
    plt.plot(n_iter, Fitness, color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_iter, Fitness, facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function iteration")
    plt.ylabel("makespan")
    plt.show()

    # plot gantt
    problem = JSSP(ins_data)
    optim_makespan = problem.decode(optim_solution)

    mch_proc_info = dict()
    for i in range(problem.JS.m):
        mch_proc_info[f"M {i + 1}"] = []

    gantt_data = []

    job_type_num_dict = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
    }
    for job_idx, job in enumerate(problem.JS.Jobs):
        job_type = job_list[job_idx]
        job_type_num_dict[job_type] += 1
        op_idx = 1
        for m_idx, start, end in zip(job._on, job.start, job.end):
            task = f"task {job_type}{job_type_num_dict[job_type]}"
            task_ = task + f"-{op_idx}"
            mch_proc_info[f"M {m_idx + 1}"].append([task_,
                                                    start,
                                                    end])
            Table[m_idx + 1].append(start)
            Table[m_idx + 1].append(end)
            if task not in Task_list:
                Task_list.append(task)
            op_idx += 1
            gantt_data.append(dict(Task=task, Start=start, Finish=end, Resource=m_idx + 1))
    print(gantt_data)

    # df = pd.DataFrame(gantt_data)
    # fig = ff.create_gantt(df, index_col='')
    # fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Task")
    # fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
    # fig.show()

    # if not os.path.exists("images"):
    #     os.mkdir("images")
    # fig.write_image("images/fig1.png")

    def Decode(self, CHS):
        for i in range(self.State):
            self.Stage_Decode(CHS, i)
            Job_end = [self.Jobs[i].last_ot for i in range(self.J_num)]
            CHS = sorted(range(len(Job_end)), key=lambda k: Job_end[k], reverse=False)

    # 画甘特图
    def Gantt(data):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
             'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
             'navajowhite', 'navy', 'sandybrown']
        M_num = 0
        for i in range(len(data)):
            Start_time = data[i]['Start']
            End_time = data[i]['Finish']
            Job = data[i]['Task']
            plt.barh(data[i]['Resource'], width=End_time - Start_time, height=0.8, left=Start_time,
                     color=M[Task_list.index(Job)], edgecolor='black')
            plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=data[i]['Resource'] - 0.2,
                     s=Task_list.index(Job) + 1, size=15, fontproperties='Times New Roman')
            M_num = max(M_num, data[i]['Resource'])
        plt.yticks(np.arange(1, M_num + 1), size=20, fontproperties='Times New Roman')

        plt.ylabel("Machine", size=20, fontproperties='SimSun')
        plt.xlabel("Time", size=20, fontproperties='SimSun')
        plt.tick_params(labelsize=20)
        plt.tick_params(direction='in')
        plt.show()
    Gantt(gantt_data)

    output = ""
    for key, machine_info in mch_proc_info.items():
        # Sort by starting time.
        # if len(machine_info) == 0:
        #     continue
        machine_info = sorted(machine_info, key=lambda x: x[1])
        sol_line_tasks = "Machine " + str(key[2:]) + ": "
        sol_line = "           "

        for assigned_task in machine_info:
            name = assigned_task[0]
            # Add spaces to output to align columns.
            sol_line_tasks += f"{name:15}"

            start = assigned_task[1]
            duration = assigned_task[2] - assigned_task[1]
            sol_tmp = f"[{start},{start + duration}]"
            # Add spaces to output to align columns.
            sol_line += f"{sol_tmp:15}"

        sol_line += "\n"
        sol_line_tasks += "\n"
        output += sol_line_tasks
        output += sol_line

    print(output)

    # Table = {key: sorted(value) for key, value in Table.items()}
    # # 转换每个键的列表为大小为500的列表，区间内为1，区间外为0
    # converted_dict = {}
    # for key, intervals in Table.items():
    #     # 初始化大小为500的列表，初始值为0
    #     task_list = [0] * 500
    #     # 将每个区间内的元素设置为1
    #     for i in range(0, len(intervals), 2):
    #         start, end = intervals[i], intervals[i + 1]
    #         for j in range(start, end):
    #             task_list[j] = 1
    #     converted_dict[key] = task_list
    # Machine_status = pd.DataFrame(converted_dict)
    # Machine_status.columns = [f"Machine{col}" for col in Machine_status.columns]
    # print(Machine_status)
