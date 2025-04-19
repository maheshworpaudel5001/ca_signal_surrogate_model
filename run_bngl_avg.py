#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:58:00 2022

@author: ixn004
"""

import numpy as np
import bionetgen
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt


def run_bionetgen_avg(n, p, K1, K2):

    # 0. Bionetgen parameter set
    model = bionetgen.bngmodel("zeta_0.bngl")
    model.parameters.Kab = K1  # assigning new kon
    model.parameters.KU = K2  # assigning new koff

    # 1. Create n output bionetgen folders
    for i in range(n):
        dir_name = f"output_{i}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(
            f"{dir_name}/zeta.bngl", "w"
        ) as f:  # write a new file assigning new kon, koff
            f.write(str(model))  # writes the changed model to new_model file
        result = bionetgen.run(
            f"{dir_name}/zeta.bngl", out=dir_name, suppress=True
        )  # run bionetgen

    t = []
    pzap = []
    for i in range(n):
        data = pd.read_csv(
            f"output_{i}/zeta.gdat", sep="\t", skiprows=1, header=None
        )  # read the output file
        data = np.array(data)  # convert to numpy array
        t.append(data[:, 0])
        pzap.append(data[:, p])  # pZAP70 is in the 2nd colum

    t_avg = np.mean(t, axis=0).T  # average over the time
    pzap_avg = np.mean(pzap, axis=0).T  # average over the pZAP70

    if not os.path.exists("average_pzap_output"):
        os.makedirs("average_pzap_output")
    # 2. Write the average data to a new fil
    X = np.column_stack((t_avg, pzap_avg))  # stack the time and pZAP70
    np.savetxt(
        "average_pzap_output/avg_time_pZAP.dat",
        X,
        delimiter="\t",
        header="time\tpZAP70",
    )  # write to file

    for i in range(n):
        shutil.rmtree(f"output_{i}")

    return t_avg, pzap_avg
