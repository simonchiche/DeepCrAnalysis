import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from PlotConfig import MatplotlibConfig
import os


DataPath = "/Users/chiche/Desktop/EMdistrib/Data"

OutputPath = MatplotlibConfig(os.getcwd(), DataPath)

SimFolder = glob.glob(DataPath + "/*")

EmDistrib = dict()
for i in range(len(SimFolder)):

    print(SimFolder[i])
    EmData = glob.glob(SimFolder[i] + "/*")
    zenith = int(SimFolder[i].split("/")[-1].split("_")[3])
    print(EmData[0])
    EmDistrib[i] = np.loadtxt(EmData[0], unpack = True).T

    plt.plot(EmDistrib[i][:,0], EmDistrib[i][:,1], label ="$\\theta = %.d^{\circ}$" %zenith)

plt.xlabel("Slant depth [m]")
plt.ylabel("$N_{EM}$")
plt.legend()
plt.yscale("log")
plt.grid()
plt.title("$E_p = 10^{17.5} {eV}$")
plt.savefig("/Users/chiche/Desktop/EMdistrib/Plots/EmDistrib_vs_depth_log.pdf")
plt.show()


