import numpy as np
import matplotlib.pyplot as plt


def PlotAllAirLdfs(posx_MaxLDF_All, Ex_MaxAirLDF_All, ZenithAll, EnergyAll, selE, selDepth, title, OutputPath, Save):
    maskE = np.where(EnergyAll == selE)
    posx_MaxLDF_sel = posx_MaxLDF_All[maskE]
    Ex_MaxLDF_sel = Ex_MaxAirLDF_All[maskE]

    print(selE, selDepth)
    for i in range(len(posx_MaxLDF_sel)):
        

        plt.plot(posx_MaxLDF_sel[i][:, 0], Ex_MaxLDF_sel[i], "-", label="$\\theta = %.d^{\circ}$" %(ZenithAll[maskE][i]), linewidth=2.5)

    plt.xlabel("Position [m]")
    plt.ylabel("$E_{tot}^{peak}$ [$\mu$V/m]")
    plt.title("In-air, E=%.2f EeV, Depth =$%d\,m$" %(selE, selDepth))
    plt.legend()
    plt.grid()
    
    plt.savefig(OutputPath + "LDF_" + title + "_E%.2f_Detph%d.pdf" %(selE, selDepth), bbox_inches="tight") if Save else None
    plt.show()


def PlotAllIceLdfs(posx_MaxLDF_All, Ex_MaxAirLDF_All, ZenithAll, EnergyAll, selE, selDepth, title, OutputPath, Save):
    maskE = np.where(EnergyAll == selE)
    posx_MaxLDF_sel = posx_MaxLDF_All[maskE]
    Ex_MaxLDF_sel = Ex_MaxAirLDF_All[maskE]

    print(selE, selDepth)
    for i in range(len(posx_MaxLDF_sel)):
        

        plt.plot(posx_MaxLDF_sel[i][:, 0], Ex_MaxLDF_sel[i], "-", label="$\\theta = %.d^{\circ}$" %(ZenithAll[maskE][i]), linewidth=2.5)

    plt.xlabel("Position [m]")
    plt.ylabel("$E_{tot}^{peak}$ [$\mu$V/m]")
    plt.title("In-air, E=%.2f EeV, Depth =$%d\,m$" %(selE, selDepth))
    plt.legend()
    plt.grid()
    plt.xlim(-400, 400)
    plt.savefig(OutputPath + "LDF_" + title + "_E%.2f_Detph%d.pdf" %(selE, selDepth), bbox_inches="tight") if Save else None
    plt.show()
