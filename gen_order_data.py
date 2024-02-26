
import numpy as np
import pandas as pd

proc_time_range = [1, 100]

job_type = ['A', 'B', 'C', 'D']

# Possible processing routes for different type job
proc_routeA = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
proc_routeB = [2, 3, 5, 6, 7, 8, 9, 10, 11, 13]
proc_routeC = [4, 5, 6, 7, 8, 9, 10, 11, 13]
proc_routeD = [12, 13]

machine = [x+1 for x in range(13)]

def gen_job_ope(proc_route, prob, proc_time_range):
    job_ope_tab = []

    for i in machine:
        if i in proc_route:
            if np.random.random() < prob:
                ope_tab = [0] * len(machine)
                ope_tab[i-1] = np.random.randint(low=proc_time_range[0], high=proc_time_range[1])
                job_ope_tab.append(ope_tab)
    return job_ope_tab

def gen_proc_table(job_type_list):
    proc_tab = []
    for x in job_type_list:
        job_ope_tab = []
        if x == "A":
            job_ope_tab = gen_job_ope(proc_routeA, 0.5, proc_time_range)
        elif x == "B":
            job_ope_tab = gen_job_ope(proc_routeB, 0.5, proc_time_range)
        elif x == "C":
            job_ope_tab = gen_job_ope(proc_routeC, 0.5, proc_time_range)
        elif x == "D":
            job_ope_tab = gen_job_ope(proc_routeD, 1, proc_time_range)

        proc_tab.append(job_ope_tab)

    return proc_tab

def gen_order_data(job_list):
    np.random.seed(200)
    proc_tab = gen_proc_table(job_list)

    ope_num_list = np.array([len(x) for x in proc_tab])
    # bias_list = np.cumsum(ope_num_list)

    data = []
    for job_tab in proc_tab:
        for tab in job_tab:
            data.append(tab)
    proc_tab_array = np.array(data)

    return data, ope_num_list, proc_tab_array


if __name__ == '__main__':
    # change seed to gen different proc_tab
    np.random.seed(200)

    job_type_list = ["A", "B", "C", "D"]

    proc_tab =  gen_proc_table(job_type_list)
    print(proc_tab)

    # generate excel table
    ope_num_list = np.array([len(x) for x in proc_tab])
    bias_list = np.cumsum(ope_num_list)
    print(bias_list)

    # trans proc_tab list to proc_tab array
    data = []
    for job_tab in proc_tab:
        for tab in job_tab:
            data.append(tab)
    proc_tab_array = np.array(data)
    print(proc_tab_array)

    # columns in excel file
    columns = ['job', 'operation']
    M_list = [f'M {x}' for x in machine]
    columns = columns + M_list

    # data in excel file
    job_type_num_dict = {
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 1,
    }
    data = []
    job_idx = 1
    job_type = job_type_list[0]
    ope_idx = 1
    for row_idx, sin_row in enumerate(proc_tab_array):
        if row_idx + 1 > bias_list[job_idx-1]:
            job_type_num_dict[job_type] += 1
            job_idx += 1
            job_type = job_type_list[job_idx-1]
            ope_idx = 1

        sin_row = sin_row.tolist()
        sin_row = [f"job {job_type}-{job_type_num_dict[job_type]}", f"O {job_type}-{job_type_num_dict[job_type]} {ope_idx}"] + sin_row
        data.append(sin_row)
        ope_idx += 1

    print(data)

    df = pd.DataFrame(data = data, columns = columns)
    df.to_csv('process_tab.csv')





















