import numpy as np
import matplotlib.pyplot as plt

def PlotPeakEfield(Pos, Epeak, Nlay, energy, zenith,label):

    savedir = "/Users/chiche/Desktop/DoublePulseMaps"
    sel = (Pos[:,2] == 3116)
    plt.scatter(Pos[sel,0], Pos[sel,1], c=Epeak, cmap="jet")
    cbar = plt.colorbar()
    cbar.set_label("Log($E_{peak}$)")
    plt.xlabel("x [m]")
    plt.xlabel("y [m]")
    plt.title(r"E = %.3f EeV, $\theta=%.d^{\circ}$, |z| =100 m" %(energy, zenith))
    plt.savefig(savedir + "/" + label +"_E%.3f_th%.d.pdf" %(energy, zenith), bbox_inches = "tight")
    plt.show()

def PlotDumbleBumpsMaps(Pos, isDoubleBump, energy, zenith):

    savedir = "/Users/chiche/Desktop/DoublePulseMaps"

    

    sel = (Pos[:,2] == 3116)
    Ndouble = np.sum(isDoubleBump[sel])
    plt.scatter(Pos[sel,0], Pos[sel,1], c=isDoubleBump[sel], cmap="viridis", label ="Ndouble = %.d" %Ndouble)
    plt.xlabel("x [m]")
    plt.xlabel("y [m]")
    plt.legend()
    plt.title(r"E = %.3f EeV, $\theta=%.d^{\circ}$, |z| =100 m" %(energy, zenith))
    #plt.savefig(savedir + "/DoublePulsesMap_E%.3f_th%.d.pdf" %(energy, zenith), bbox_inches = "tight")
    plt.show()

    PosDoubleBumps_x = Pos[sel,0][isDoubleBump[sel] == 1]
    PosDoubleBumps_y = Pos[sel,1][isDoubleBump[sel] == 1]
    PosDoubleBumps_z = Pos[sel,2][isDoubleBump[sel] == 1]
    PosDoubleBumps = np.array([PosDoubleBumps_x, PosDoubleBumps_y, PosDoubleBumps_z]).T
    print(PosDoubleBumps.shape)
    return  PosDoubleBumps

def PlotDumbleBumpsMapsHighRes(Pos, isDoubleBump, energy, zenith, OutputPath):

    savedir = "/Users/chiche/Desktop/DoublePulseMaps"

    sel = (Pos[:,2] == 3116)

    colors = np.where(isDoubleBump[sel], 'gold', 'indigo')



    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Pos[sel,0],Pos[sel,1], c=colors, edgecolor='black', s=80, alpha=0.9, linewidth=0.5)

    # Axis and labels
    plt.xlabel("y [m]")
    plt.ylabel("x [m]")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().set_aspect('equal')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gold', edgecolor='black', label='Double pulse'),
        Patch(facecolor='indigo', edgecolor='black', label='No double pulse'),
        ]
    plt.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        frameon=True,
        framealpha=0.9
    )

    # Title
    plt.title(r"E = %.3f EeV, $\theta=%.d^{\circ}$, Depth =100 m" %(energy, zenith), fontsize =13)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(OutputPath + "DoublePulsesMap_E%.3f_th%.d.pdf" %(energy, zenith), bbox_inches = "tight")
    plt.show()

    return  



def PlotTrace(Time, E_t):
        plt.plot(Time*1e9, E_t[i])
        plt.xlim(200,2000)
        plt.show()

def PlotDoubleBumpTrace(TimeAir, TimeIce, Eair, Eice):
        plt.plot(TimeAir*1e9, Eair, label ="In-air emission")
        plt.plot(TimeIce*1e9, Eice, label ="In-ice emission")
        plt.xlim(200,2000)
        plt.xlabel("Time[ns]")
        plt.ylabel(r"$E_{\rm tot}\,  (\mu V/m)$")
        plt.show()

def PlotDoubleBumpVsZen(ZenAll, NdoubleAll):
    plt.scatter(ZenAll, NdoubleAll)
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel("Number of double pulse events")
    #plt.savefig(savedir + "/Ndouble_vs_theta.pdf")
    plt.grid()
    plt.show()

def PlotTimeDelay(DeltaT):
    plt.scatter(np.arange(len(DeltaT)), np.array(DeltaT)*1e9)
    plt.ylabel("Time delay [ns]")
    plt.xlabel("Antenna ID")
    #plt.savefig("Deltat_E%.3f_th%.d.pdf" %(energy, zenith), bbox_inches = "tight")
    plt.show()


def PlotNAirtrigger(ZenithAll, Nsingleair_x, Nsingleair_y, Nsingleair_z, threshold1, threshold2, selE):
    plt.scatter(ZenithAll, Nsingleair_x, label ="x")
    plt.scatter(ZenithAll, Nsingleair_y, label ="y")
    plt.scatter(ZenithAll, Nsingleair_z, label="z")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{trigger}^{air}$")
    plt.title("$E=%.2f$ EeV, $th1 = %.d \, \mu $V/m, $th2 = %.d \, \mu $V/m" %(selE, threshold1, threshold2),  fontsize =13)
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/Ntrigair_E0.316_vs_zen_high_thresold.pdf")
    plt.show()



def PlotNIcetrigger(ZenithAll, Nsingleice_x, Nsingleice_y, Nsingleice_z, threshold1, threshold2, selE):
    plt.scatter(ZenithAll, Nsingleice_x, label ="x")
    plt.scatter(ZenithAll, Nsingleice_y, label ="y")
    plt.scatter(ZenithAll, Nsingleice_z, label="z")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{trigger}^{ice}$")
    plt.title("$E=%.2f\,$ EeV, $th1 = %.d \, \mu $V/m, $th2 = %.d \, \mu$V/m" %(selE, threshold1, threshold2), fontsize =13)
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/Ntrigice_E0.316_vs_zen_high_thresold.pdf")
    plt.show()


def PlotNtriggAll(ZenithAll, NtriggerAll):
    plt.scatter(ZenithAll, NtriggerAll, label ="tot")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{double}$")
    #plt.title("$E=10^{17.5} eV$, $th1 = 600 \, \mu Vs/m$, $th2 = 400 \, \mu Vs/m$")
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/DoubleRate_E0.316_vs_zen.pdf")
    plt.show()

def PlotNdoubleTot(ZenithAll, Ndouble_tot):
    plt.scatter(ZenithAll, Ndouble_tot, label ="tot")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{double}$")
    #plt.title("$E=10^{17.5} eV$, $th1 = 600 \, \mu Vs/m$, $th2 = 400 \, \mu Vs/m$")
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/DoubleRate_E0.316_vs_zen.pdf")
    plt.show()

def PlotDoubleRateTot(ZenithAll, Ndouble_tot, NtriggerAll):
    DoubleRateTot = Ndouble_tot/NtriggerAll

    plt.scatter(ZenithAll, DoubleRateTot, label ="tot")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{double}$")
    #plt.title("$E=10^{17.5} eV$, $th1 = 600 \, \mu Vs/m$, $th2 = 400 \, \mu Vs/m$")
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/DoubleRate_E0.316_vs_zen.pdf")
    plt.show()


def PlotDoubleRateTotperChannel(ZenithAll, Ndouble_x, Ndouble_y, Ndouble_z, Ntrigger_All_x, Ntrigger_All_y, Ntrigger_All_z, th1, th2):

    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue, Orange, Green (colorblind-safe)
    linestyles = ["-", "--", "-."]

    arg = np.argsort(ZenithAll)
    plt.plot(np.array(ZenithAll)[arg], Ndouble_x[arg]/Ntrigger_All_x[arg], label ="x", color=colors[0], linestyle=linestyles[0], linewidth=2, marker='o', markersize=5)
    plt.plot(np.array(ZenithAll)[arg], Ndouble_y[arg]/Ntrigger_All_y[arg], label ="y", color=colors[1], linestyle=linestyles[1], linewidth=2, marker='o', markersize=5)
    plt.plot(np.array(ZenithAll)[arg], Ndouble_z[arg]/Ntrigger_All_z[arg], label="z", color=colors[2], linestyle=linestyles[2], linewidth=2, marker='o', markersize=5)
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel(r"$N_{\mathrm{double}}/N_{\mathrm{trigger}}$")
    plt.title("$E=10^{17.5}\,$eV, Thresolds $= %.d, %.d \, \mu V/m$" %(th1, th2), fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    #plt.savefig(OutputPath + "DoubleRateAllchannels.pdf", bbox_inches="tight")
    plt.show()