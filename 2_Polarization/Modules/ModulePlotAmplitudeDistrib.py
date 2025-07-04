import numpy as np
import matplotlib.pyplot as plt

def PlotAmplitudeDistribution(Etot1, Etot2, Etot3, bin_edges, scale = "linear"):

    labels=('$10^{16.5}$ eV', '$10^{17}$ eV', '$10^{17.5}$ eV')

    plt.hist(Etot1, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
    plt.hist(Etot2, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
    plt.hist(Etot3 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
    plt.xlabel('$E_{tot}\, [\mu V s/m]$')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
    #plt.savefig("/Users/chiche/Desktop/InAirFilteredPulseDistrib.pdf", bbox_inches="tight")
    plt.show()