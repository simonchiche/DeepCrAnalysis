import numpy as np
import matplotlib.pyplot as plt
import glob
def plot_Xmax_pos(EnergyAll, XmaxPosAll, OutputPath):
    Ebins = np.unique(EnergyAll)

    for i in range(len(Ebins)):
        maskE = EnergyAll == Ebins[i]
    # Xmax Position for different zenith
        plt.scatter(XmaxPosAll[:,0][maskE], XmaxPosAll[:,2][maskE], label = f"E = {Ebins[i]} EeV")
        plt.xlabel("x [m]")
        plt.ylabel("Xmax height [m]")
        plt.legend()
        plt.grid()
        plt.ylim(2000, max(XmaxPosAll[:,2])+200)
        plt.axhspan(2000, 3216, color='skyblue', alpha=0.3) 
    #plt.savefig(OutputPath + "XmaxAirPositions.pdf", bbox_inches = 'tight')
    plt.show()


def PlotXmaxHeightvsZenith(EnergyAll, ZenithAll, XmaxPosAll, OutputPath):
    Ebins = np.unique(EnergyAll)

    for i in range(len(Ebins)):
        maskE = EnergyAll == Ebins[i]
        plt.scatter(ZenithAll[maskE], XmaxPosAll[:,2][maskE],  label = f"E = {Ebins[i]} EeV", marker='x', s=50)
        plt.xlabel("Zenith [Deg.]")
        plt.ylabel("Xmax height [m]")
        plt.legend()
        plt.ylim(2000, max(XmaxPosAll[:,2])+200)
        plt.axhspan(2000, 3216, color='skyblue', alpha=0.3) 
        plt.grid()
    plt.savefig(OutputPath + "XmaxAirHeightvsZenith.pdf", bbox_inches = 'tight')
    plt.show()


def PlotXmaxGrammage(EnergyAll, ZenithAll, XmaxAll, OutputPath):
    Ebins = np.unique(EnergyAll)

    for i in range(len(Ebins)):
        maskE = EnergyAll == Ebins[i]
        plt.scatter(ZenithAll[maskE], XmaxAll[maskE],  label = f"E = {Ebins[i]} EeV")
        plt.xlabel("Zenith [Deg.]")
        plt.ylabel(r"Xmax Depth [$\mathrm{g/cm^2}$]")
        plt.legend()
        plt.grid()
    #plt.savefig(OutputPath + "XmaxAirDepth.pdf", bbox_inches = 'tight')
    plt.show()

def GenerateXmaxIceData(DataPath):
    Data = glob.glob(DataPath)
    XmaxIceAll = []
    EiceAll = []
    ZenIceAll = []
    for i in range(len(Data)):
        XmaxIce = np.loadtxt(Data[i])
        XmaxIceAll.append(XmaxIce)
        EiceAll.append(float(Data[i].split("/")[-1].split("_")[1]))
        ZenIceAll.append(float(Data[i].split("/")[-1].split("_")[2][:-4]))

    XmaxIceData = np.array([XmaxIceAll, EiceAll, ZenIceAll]).T
    np.savetxt("./XmaxIce/XmaxIceData.txt", XmaxIceData)


def PlotXmaxDistribution(EnergyAll, XmaxAll, OutputPath):
    Ebins = np.unique(EnergyAll)
    Xmax_16_5 = XmaxAll[EnergyAll == Ebins[0]]
    Xmax_17 = XmaxAll[EnergyAll == Ebins[1]]
    Xmax_17_5 = XmaxAll[EnergyAll == Ebins[2]]

    bins = np.linspace(640, 770, 30)
    plt.hist(Xmax_16_5, bins = bins, alpha=0.5, label = f"E = {Ebins[0]} EeV", color='blue', edgecolor='black')
    plt.hist(Xmax_17, bins = bins, alpha=0.5, label = f"E = {Ebins[1]} EeV", color='orange', edgecolor='black')
    plt.hist(Xmax_17_5, bins = bins, alpha=0.5, label = f"E = {Ebins[2]} EeV", color='green', edgecolor='black')
    plt.legend()
    plt.xlabel(r"In-air Xmax Depth [$\mathrm{g/cm^2}$]")
    plt.ylabel("Counts")
    plt.savefig(OutputPath + "XmaxAirDepthHist.pdf", bbox_inches = 'tight')
    plt.show()

def PlotSlantXmaxIce(EiceAll, ZenIceAll, XmaxIceAll, OutputPath):
    Ebins = np.unique(EiceAll)
    mask = EiceAll == Ebins[0]

    for i in range(len(Ebins)):
        mask = EiceAll == Ebins[i]
        
        # Slant Xmax vs zenith
        plt.scatter(ZenIceAll[mask], XmaxIceAll[mask], marker='*', label = f"E = {Ebins[i]} EeV", s=50)
        plt.xlabel("Zenith [Deg.]")
        plt.ylabel("Slant depeth in-ice Xmax [m]")
        plt.legend()
        plt.grid()
        plt.ylim(2,7.5)
    plt.savefig(OutputPath + "XmaxIceSlantvsZenith.pdf", bbox_inches = 'tight')
    plt.show()

def PlotXmaxIceDepth(EiceAll, ZenIceAll, XmaxIceAll, OutputPath):
    Ebins = np.unique(EiceAll)
    for i in range(len(Ebins)):
        mask = EiceAll == Ebins[i]
        
        # Xmax depth vs zenith
        plt.plot(ZenIceAll[mask], XmaxIceAll[mask]*np.cos(ZenIceAll[mask]*np.pi/180.0), 'o', label = f"E = {Ebins[i]} EeV")
        plt.xlabel("Zenith [Deg.]")
        plt.ylabel("Xmax Depth [m]")
        plt.legend()
        plt.grid()
    #plt.savefig(OutputPath + "XmaxIceDepthvsZenith.pdf", bbox_inches = 'tight')
    plt.show()

def PlotIceXmaxDistribution(EnergyAll, XmaxAll, OutputPath):
    Ebins = np.unique(EnergyAll)
    Xmax_16_5 = XmaxAll[EnergyAll == Ebins[0]]
    Xmax_17   = XmaxAll[EnergyAll == Ebins[1]]
    Xmax_17_5 = XmaxAll[EnergyAll == Ebins[2]]

    bins = np.linspace(3, 6, 5)

    plt.figure(figsize=(7,5))

    plt.hist(Xmax_16_5, bins=bins, histtype='step',
             linewidth=2, label=f"E = {Ebins[0]} EeV", linestyle='dashed', color='black')
    plt.hist(Xmax_17,   bins=bins, histtype='step',
             linewidth=2, label=f"E = {Ebins[1]} EeV", color='red')
    plt.hist(Xmax_17_5, bins=bins, histtype='step',
             linewidth=2, label=f"E = {Ebins[2]} EeV", color='blue', linestyle='dotted')

    plt.legend()
    plt.xlabel(r"In-ice Xmax slant Depth [m]")
    plt.ylabel("Counts")
    
    plt.tight_layout()
    plt.savefig(OutputPath + "XmaxIceSlantHist.pdf", bbox_inches = 'tight')
    plt.show()