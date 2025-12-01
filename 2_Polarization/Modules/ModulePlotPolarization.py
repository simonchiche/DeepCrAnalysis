import numpy as np
import matplotlib.pyplot as plt

def PlotHVratioAirDistribperZen(ZenithAll, HVratioAirAll, Save, OutputPath):
    
    ZenithBins = np.unique(ZenithAll)

    for i in range(len(ZenithBins)):

        sel = (ZenithAll == ZenithBins[i])
        HVratiozen_air = HVratioAirAll[sel].flatten()
        bin_edges = np.linspace(0, 10, 80) 

        plt.hist(HVratiozen_air, bin_edges, alpha=0.6, edgecolor='black')
        plt.xlabel('Hpol/Vpol')
        plt.ylabel('Nant')
        #plt.xlim(0,2000)
        plt.legend()
        #if(scale=="log"): plt.yscale("log")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
        #plt.title("In-air emission")
        plt.title(r"In-air, $\theta =%.d^{\circ}$" %ZenithBins[i])
        plt.savefig(OutputPath + "InAirFilteredHVratio_zen%.d.pdf" %ZenithBins[i], bbox_inches="tight") if Save else None
        plt.show()



def PlotHVratioIceDistribperZen(ZenithAll, HVratioIceAll, Save, OutputPath):
    
    ZenithBins = np.unique(ZenithAll)
    bin_edges = np.linspace(0, 10, 80) 
    for i in range(len(ZenithBins)):
        
        sel = (ZenithAll == ZenithBins[i])    

        HVratiozen_ice = HVratioIceAll[sel].flatten()

        plt.hist(HVratiozen_ice, bin_edges, alpha=0.6, edgecolor='black')
        plt.xlabel('Hpol/Vpol')
        plt.ylabel('Nant')
        #plt.xlim(0,2000)
        plt.legend()
        #if(scale=="log"): plt.yscale("log")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axvline(x=np.sqrt(2), color='red', linestyle='--', linewidth=2)
        #plt.title("In-air emission")
        plt.title(r"In-ice, $\theta =%.d^{\circ}$" %ZenithBins[i])
        plt.savefig(OutputPath + "InIceFilteredHVratio_zen%.d.pdf" %ZenithBins[i], bbox_inches="tight") if Save else None
        plt.show()

def GetHVratioAirvsE(EtotAirAll16_5, EtotAirAll17, EtotAirAll17_5, OutputPath):
    labels=('$10^{16.5}$ eV', '$10^{17}$ eV', '$10^{17.5}$ eV')
    bin_edges = np.linspace(0, 10, 80) 


    plt.hist(EtotAirAll16_5, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
    plt.hist(EtotAirAll17, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
    plt.hist(EtotAirAll17_5 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
    plt.xlabel('Hpol/Vpol')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    #if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
    plt.title("In-air emission")
    #plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
    #plt.savefig(OutputPath + "InAirFilteredHVratio.pdf", bbox_inches="tight") if Save else None
    plt.show()



def GetHVratioIcevsE(EtotIceAll16_5, EtotIceAll17, EtotIceAll17_5, OutputPath, Save=False):
    labels=('$10^{16.5}$ eV', '$10^{17}$ eV', '$10^{17.5}$ eV')
    bin_edges = np.linspace(0, 10, 80) 

    plt.hist(EtotIceAll16_5, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
    plt.hist(EtotIceAll17, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
    plt.hist(EtotIceAll17_5 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
    plt.xlabel('Hpol/Vpol')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    #if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
    plt.title("In-ice emission")
    #plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
    plt.savefig(OutputPath + "InIceFilteredHVratio.pdf", bbox_inches="tight") if Save else None
    plt.show()
