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
