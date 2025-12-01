import numpy as np
import matplotlib.pyplot as plt

def PlotEradThetaScaling(Erad_allsims, Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    for i in range(len(Depths)):

        sel = (Erad_allsims[:,4] == Depths[i]) & (Erad_allsims[:,5] == SelE)
        
        arg = np.argsort(Erad_allsims[sel][:,6])
        plt.plot(Erad_allsims[sel][:,6][arg], Erad_allsims[sel][:,0][arg], label ="$E^{rad}_x$")
        plt.plot(Erad_allsims[sel][:,6][arg], Erad_allsims[sel][:,1][arg], label ="$E^{rad}_y$")
        plt.plot(Erad_allsims[sel][:,6][arg], Erad_allsims[sel][:,2][arg], label ="$E^{rad}_z$")
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{rad} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend()
        plt.title(title + " $E=%.2f\,$ EeV |z| =%d m" %(SelE, Depths[i]))
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return

def PlotEradDepthScaling(Erad_allsims, SelZen, title, OutputPath):

    EnergyAll = np.unique(Erad_allsims[:,5])
    for i in range(len(EnergyAll)):
        SelE = EnergyAll[i]
        sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
        plt.scatter(Erad_allsims[sel][:,4], Erad_allsims[sel][:,3],\
                     label = "E = %.2f EeV" %EnergyAll[i])
    plt.yscale("log")
    #plt.ylim(min(Erad_allsims[:,5])/5, max(Erad_allsims[:,5])*5)
    plt.ylabel("$E_{rad} \, $[MeV]")
    plt.title(title + " $\\theta = %.d^{\circ}$" %(SelZen))
    plt.xlabel("Depth [m]")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9), framealpha=0.5)
    plt.savefig(OutputPath + "_" + title + "_vs_Depth_th%.d.pdf" %SelZen, bbox_inches = "tight")
    plt.show()
    return


def PlotEradEnergyScaling(Erad_allsims, SelDepth, title, OutputPath):

    ZenithAll = np.unique(Erad_allsims[:,6])

    for i in range(len(ZenithAll)):
        sel = (Erad_allsims[:,6] == ZenithAll[i]) & (Erad_allsims[:,4] == SelDepth)

        arg = np.argsort(Erad_allsims[sel][:,5])
        plt.plot(Erad_allsims[sel][:,5][arg], Erad_allsims[sel][:,3][arg],\
                 label ="$\\theta =%.d^{\circ}$" %ZenithAll[i])
    plt.xlabel("Energy [EeV]")
    plt.ylabel("$E_{rad} \, [eV]")
    plt.legend()
    plt.title(title + " $|z|=%.d$" %SelDepth)
    plt.savefig(OutputPath + "_" + title + "_vs_E_|z|%.d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()


def PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath):

    ZenithAll =  np.unique(Eradair_allsims[:,6])
    EnergyAll =  np.unique(Eradair_allsims[:,5])
    
    for i in range(len(ZenithAll)):

        selE1 = (Eradair_allsims[:,6] == ZenithAll[i]) &  (Eradair_allsims[:,5] == EnergyAll[Eindex])
        EindexLow = (Eindex -1)
        selE2 = (Eradair_allsims[:,6] == ZenithAll[i]) &  (Eradair_allsims[:,5] == EnergyAll[EindexLow])
        EscalingRatio = Eradair_allsims[:,3][selE1]/ Eradair_allsims[:,3][selE2]

        plt.plot(Eradair_allsims[:,4][selE1], EscalingRatio, label = "$\\theta =%.d^{\circ}$" %ZenithAll[i])
    plt.legend()
    plt.xlabel("Depth [m]")
    plt.ylabel("$E%.2f/E%.2f$" %(EnergyAll[Eindex], EnergyAll[EindexLow]))
    plt.title(title)
    plt.savefig(OutputPath + "_" + title + "Escaling%.2f_vs_Depth.pdf" %EnergyAll[Eindex], bbox_inches = "tight")
    plt.show()

    return


def PlotAirIceEradRatiovsTheta(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, OutputPath):
    sel = (Eradair_allsims[:,6] == SelZen) & (Eradair_allsims[:,5] == SelE)
    
    
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        EradAirIceRatio_x = Eradair_allsims[sel][:,0]/ Eradice_allsims[sel][:,0]
        EradAirIceRatio_y = Eradair_allsims[sel][:,1]/ Eradice_allsims[sel][:,1]
        EradAirIceRatio_z = Eradair_allsims[sel][:,2]/ Eradice_allsims[sel][:,2]

        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_x[arg], label ="$E^{rad, air}_x/E^{rad, ice}_x$")
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_y[arg], label ="$E^{rad, air}_y/E^{rad, ice}_y$")
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_z[arg], label ="$E^{rad, air}_z/E^{rad, ice}_z$")
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{rad}^{air}/E_{rad}^{ice}\,$[50-1000 MHz]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend()
        plt.title("$E=%.2f\,$ EeV |z| =%d m" %(SelE, Depths[i]))
        plt.savefig(OutputPath + "air_ice_ratio_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return


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
    plt.yscale("log")
    #plt.ylim(min(data)/5, max(data)*5)
    plt.ylabel("$E_{rad}^{air}/E_{rad}^{ice}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.legend()
    plt.title("$|z| =%d$ m" %(SelDepth))
    plt.savefig(OutputPath + "air_ice_ratio_vs_theta_vsE_z%d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()

    return

def PlotHpolVpolEradRatiovsThetavsE(Shower, Erad_allsims, SelDepth, title, OutputPath, Save,  ylowlim=None, yhighlim=None):

    EnergyAll = np.unique(Erad_allsims[:,5])  
    Gdeep = Shower.glevel - SelDepth  
    for i in range(len(EnergyAll)):
        sel = (Erad_allsims[:,4] == SelDepth) & (Erad_allsims[:,5] == EnergyAll[i])
        
        EradHpole_tot = (Erad_allsims[sel][:,0] + Erad_allsims[sel][:,1])
        EradVpole_tot = Erad_allsims[sel][:,2]
        EradHpoleVpoleRatio_tot = EradHpole_tot/EradVpole_tot

        arg = np.argsort(Erad_allsims[sel][:,6])
        plt.plot(Erad_allsims[sel][:,6][arg], EradHpoleVpoleRatio_tot[arg], label ="$E= %.2f$ EeV" %EnergyAll[i])
    #plt.yscale("log")
    #plt.ylim(min(data)/5, max(data)*5)
    plt.ylabel("$E_{rad}^{Hpol}/E_{rad}^{Vpol}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.grid()
    plt.legend()
    if(ylowlim and yhighlim):
        plt.ylim(ylowlim, yhighlim)
    plt.title(title + ", Depth =%d m" %(Gdeep), fontsize =12)
    plt.savefig(OutputPath + "_" + title + "_Hpol_over_Vpol_vs_E_vs_zenith_z%d.pdf"\
                 %SelDepth, bbox_inches = "tight") if Save else None
    plt.show()

    return

def PlotMeanHpolVpolEradRatiovsThetavsE(Shower, Erad_allsims, SelDepth, title, OutputPath, Save):

    EnergyAll = np.unique(Erad_allsims[:,5])  
    Gdeep = Shower.glevel - SelDepth
    HVratioAll = dict()
    for i in range(len(EnergyAll)):
        sel = (Erad_allsims[:,4] == SelDepth) & (Erad_allsims[:,5] == EnergyAll[i])
        
        EradHpol_tot = (Erad_allsims[sel][:,0] + Erad_allsims[sel][:,1])
        EradVpol_tot = Erad_allsims[sel][:,2]
        EradHpoleVpoleRatio_tot = EradHpol_tot/EradVpol_tot

        arg = np.argsort(Erad_allsims[sel][:,6])
        HVratioAll[i] = EradHpoleVpoleRatio_tot[arg]
        #plt.plot(Erad_allsims[sel][:,6][arg], EradHpoleVpoleRatio_tot[arg], label ="$E= %.2f$ EeV" %EnergyAll[i])
    HVratioAllArr = np.vstack([HVratioAll[0], HVratioAll[1], HVratioAll[2]])
    HVratiomean, HVratiostd =  np.mean(HVratioAllArr, axis=0), np.std(HVratioAllArr, axis=0)
    #plt.yscale("log")
    #plt.ylim(min(data)/5, max(data)*5)
    plt.errorbar(Erad_allsims[sel][:,6][arg], HVratiomean, yerr = HVratiostd)
    plt.ylabel("$E_{rad}^{Hpol}/E_{rad}^{Vpol}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.grid()
    plt.legend()
    #plt.ylim(1, 4)
    plt.title(title + ", Depth =%d m" %(Gdeep), fontsize =12)
    plt.savefig(OutputPath + "_" + title + "_Hpol_over_Vpol_vs_E_vs_zenith_z%d.pdf"\
                 %SelDepth, bbox_inches = "tight") if Save else None
    plt.show()

    return

def PlotMeanHpolVpolEradRatiovsThetavsE(Shower, Erad_allsims, SelDepth, title, OutputPath, Save, ylowlim=None, yhighlim=None):

    EnergyAll = np.unique(Erad_allsims[:,5])  
    Gdeep = Shower.glevel - SelDepth
    HVratioAll = dict()
    for i in range(len(EnergyAll)):
        sel = (Erad_allsims[:,4] == SelDepth) & (Erad_allsims[:,5] == EnergyAll[i])
        
        EradHpol_tot = (Erad_allsims[sel][:,0] + Erad_allsims[sel][:,1])
        EradVpol_tot = Erad_allsims[sel][:,2]
        EradHpoleVpoleRatio_tot = EradHpol_tot/EradVpol_tot

        arg = np.argsort(Erad_allsims[sel][:,6])
        HVratioAll[i] = EradHpoleVpoleRatio_tot[arg]
        #plt.plot(Erad_allsims[sel][:,6][arg], EradHpoleVpoleRatio_tot[arg], label ="$E= %.2f$ EeV" %EnergyAll[i])
    HVratioAllArr = np.vstack([HVratioAll[0], HVratioAll[1]]) #np.vstack([HVratioAll[0], HVratioAll[1], HVratioAll[2]])
    HVratiomean, HVratiostd =  np.mean(HVratioAllArr, axis=0), np.std(HVratioAllArr, axis=0)
    #plt.yscale("log")
    print(HVratioAllArr)
    #plt.ylim(min(data)/5, max(data)*5)
    
    print(len(Erad_allsims[sel][:,6][arg]), len(HVratioAllArr), len(HVratiostd))
    if(title == "In-air"):
        plt.errorbar(Erad_allsims[sel][:,6][arg], HVratiomean, yerr = HVratiostd, label ="in-air", marker ="o", color = "#D62728")
    if(title == "In-ice"):
        plt.errorbar(Erad_allsims[sel][:,6][arg], HVratiomean, yerr = HVratiostd, label ="in-ice", marker ="o", color = "#4F81BD")
    plt.ylabel("$E_{\mathrm{rad}}^{\mathrm{Hpol}}/E_{\mathrm{rad}}^{\mathrm{Vpol}}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.grid()
    plt.legend()
    if(ylowlim and yhighlim):
        plt.ylim(ylowlim, yhighlim)
    #plt.ylim(1, 4)
    plt.title(title + ", Depth =%d m" %(Gdeep), fontsize =14)
    plt.savefig(OutputPath + "_" + title + "mean_Hpol_over_Vpol_vs_E_vs_zenith_z%d.pdf"\
                 %SelDepth, bbox_inches = "tight") if Save else None
    plt.show()

    return