import random
import numpy as np
import pandas as pd

from interpolation import spline
from interpolation_exp_data import exp_idata
from ca_ODE import calcium, solve
import matplotlib.pyplot as plt


def solve_ca(time, mean_pZAP, C1, C2, g):

    # . In silico Model run starts here
    N = 300  # Ca signal at N time points
    tstart = 0.0  # 26.0 # Fit to be start from which timepoint : interpolation of pZAP70 signal starts at 25 sec
    Vc = 25  # pZAP molecules in the simulation box of size 25 um^3
    z = 602  # constant factor to convert from molecules/um3 to uM

    # 2.Interpolate time, PZAP  from 600 points to 2000 points  [26,300]
    # tnew, PZAP = spline(time, mean_pZAP, N, tstart)
    # tnew = np.asarray(tnew)
    # PZAP = np.asarray(
    #     PZAP
    # )  # total number of PZAP molecule in the simulation box of size Vc
    # plt.plot(time,mean_pZAP,'co',tnew,PZAP,'k-')
    # plt.show()
    mean_pZAP = mean_pZAP / (Vc * z)  # pZAP in uM unit
    # print(tnew[0:5])Ca is simulate from 30 sec

    # 3. Ca Signal  from the ODE model feeding the PZAP signal
    CA0 = 33212  # do not change
    y0 = [CA0, 1]  # changed
    ca, h = solve(
        time, N, y0, mean_pZAP, C1, C2, g
    )  # [26,300] ca0 is at 26 sec, using ZAP signal after 26 sec
    ca = np.asarray(ca)
    h = np.asarray(h)

    # # Number of (kon,koff) exists, printing in the files
    # f = f"Model.txt"  # Generate file name (e.g., output1.txt)

    # # 4. print tnew, PZAP, ca in the files
    # with open(f, "w") as file:
    #     file.write(
    #         str(kon)
    #         + "\t"
    #         + str(koff)
    #         + "\t"
    #         + str(C1)
    #         + "\t"
    #         + str(C2)
    #         + "\t"
    #         + str(g)
    #         + "\n"
    #     )
    #     for k in range(len(tnew)):
    #         file.write(
    #             str(tnew[k])
    #             + "\t"
    #             + str(PZAP[k])
    #             + "\t"
    #             + str(h[k])
    #             + "\t"
    #             + str(ca[k])
    #             + "\n"
    #         )  ### Maitreya you can change the order the line [time, pZAP, h,Ca]

    # 5. Take the experimental data and interpolate at the theoretical points
    # idata, tnew = exp_idata(TT, Ca_data, tnew)
    # idata = np.asarray(idata)
    # tnew = np.asarray(tnew)
    # print(idata)

    # 6. Finding objective function only at t>60 sec and t<300 between model and interpolated Ca data
    # OBJ = idata - ca
    # # print(ca_diff)
    # SSR = np.sum(np.square(OBJ))
    # print(f" (SSR error)**0.25 =", SSR ** (1 / 4))

    # plotting experimental data vs. Model
    return time, ca
