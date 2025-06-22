import numpy as np
import matplotlib.pyplot as plt

def PlotAmplitudeDistribution(Etot1, Etot2, Etot3, bin_edges, labels, Save, OutputPath, pretitle, scale = "linear"):


    plt.hist(Etot1, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
    plt.hist(Etot2, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
    plt.hist(Etot3 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
    plt.xlabel('$E_{tot}^{peak}\, [\mu V /m]$')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")]
    OutputPath = OutputPath + pretitle
    plt.title(f"{pretitle} emission")
    plt.tight_layout()
    if(Save):
        plt.savefig(OutputPath + "FilteredPulseDistrib.pdf", bbox_inches="tight") 
    plt.show()