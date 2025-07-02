import numpy as np
import matplotlib.pyplot as plt

def PlotAirIceEradRatiovsThetavsE(Eradair_allsims, Eradice_allsims, SelDepth, OutputPath):

    EnergyAll = np.unique(Eradair_allsims[:,5])    
    
    for i in range(len(EnergyAll)):
        sel = (Eradair_allsims[:,4] == SelDepth) & (Eradair_allsims[:,5] == EnergyAll[i])
        
        #EradAirIceRatio_x = Eradair_allsims[sel][:,0]/ Eradice_allsims[sel][:,0]
        #EradAirIceRatio_y = Eradair_allsims[sel][:,1]/ Eradice_allsims[sel][:,1]
        #EradAirIceRatio_z = Eradair_allsims[sel][:,2]/ Eradice_allsims[sel][:,2]
        EradAirIceRatio_tot = Eradair_allsims[sel][:,3]/ Eradice_allsims[sel][:,3]

        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_tot[arg], label ="E= $%.2f$ EeV" %EnergyAll[i])
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_x[arg], label ="$E^{rad, air}_x/E^{rad, ice}_x$")
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_y[arg], label ="$E^{rad, air}_y/E^{rad, ice}_y$")
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_z[arg], label ="$E^{rad, air}_z/E^{rad, ice}_z$")
    #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
    plt.axhline(y=1.0, color='k', linestyle='--', linewidth=2.0)
    plt.yscale("log")
    #plt.ylim(min(data)/5, max(data)*5)
    plt.ylabel("$E_{rad}^{air}/E_{rad}^{ice}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.legend()
    #plt.title("$|z| =%d$ m" %(SelDepth))
    plt.title("Depth = $100\,$m", fontsize=12)
    plt.grid()
    plt.savefig(OutputPath + "air_ice_ratio_vs_theta_vsE_z%d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()

    return