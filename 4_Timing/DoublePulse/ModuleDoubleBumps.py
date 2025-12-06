import numpy as np
import matplotlib.pyplot as plt
import pickle
import subprocess
import os
#from Modules.Fluence.FunctionsGetFluence import LoadTraces
from ModulePlotDumbleBumps import PlotTrace, PlotDoubleBumpTrace
import sys
from scipy.stats import gaussian_kde
'''
def LoadSimulation(SimDataPath):

    if(not(os.path.exists(SimDataPath))):
    
        cmd = "mkdir -p " + SimDataPath
        p =subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
        stdout, stderr = p.communicate()
        Nant, Traces_C, Traces_G, Pos = LoadTraces(Path)
        
        np.save(SimDataPath + "/Nant", Nant)
        with open(SimDataPath + '/Traces_C.pkl', 'wb') as file:
            pickle.dump(Traces_C, file)
        with open(SimDataPath + '/Traces_G.pkl', 'wb') as file:
            pickle.dump(Traces_G, file)
        np.save(SimDataPath + "/Pos", Pos)
    
    else:
        
        Nant  = np.load(SimDataPath + "/Nant.npy")
        with open(SimDataPath + '/Traces_C.pkl', 'rb') as file:
            Traces_C = pickle.load(file)
        with open(SimDataPath + '/Traces_G.pkl', 'rb') as file:
            Traces_G = pickle.load(file)
        Pos  = np.load(SimDataPath + "/Pos.npy", allow_pickle=True)

    return Nant, Traces_C, Traces_G, Pos
'''


def ClassBumps(Eair, Eice, thresold1, thresold2, pulse_flags, i):
        
        keys = ["x", "y", "z", "tot"]
        #print(Eair[3][i], Eice[3][i])
        for j, key in enumerate(keys):

            #if( (abs(Eair[j][i]) >= thresold1) & (abs(Eice[j][i]) < thresold2)):
            if( (abs(Eair[j][i]) >= thresold1)):                    
                pulse_flags["isAirSinglePulse"][key].append(True)
                #if(j==3): print("Air trigger", i)
            else:
                pulse_flags["isAirSinglePulse"][key].append(False)
                #if(j==3): print("No Air trigger", i)

            if((abs(Eice[j][i]) >= thresold1)):
                pulse_flags["isIceSinglePulse"][key].append(True)
                #if(j==3): print("Ice trigger", i)
            else:
                pulse_flags["isIceSinglePulse"][key].append(False)
                #if(j==3): print("No ice trigger", i)
            
            sel1 = (abs(Eair[j][i]) >= thresold1) & (abs(Eice[j][i]) >= thresold2)
            sel2 = (abs(Eair[j][i]) >= thresold2) & (abs(Eice[j][i]) >= thresold1)
            if(sel1 or sel2):
                Deltat = Eair[4][i] - Eice[4][i] 
                print(Deltat*1e9)
                pulse_flags["Deltat"][key].append(Deltat)
                if(abs(Deltat*1e9) > 50):  # 3 ns separation between the two pulses
                    pulse_flags["isDoublePulse"][key].append(True)
                else:
                    pulse_flags["isDoublePulse"][key].append(False)

            else:
                pulse_flags["isDoublePulse"][key].append(False)
                pulse_flags["Deltat"][key].append(np.nan)

        return pulse_flags


def GetDoubleBumps(Shower, Eall_c, Eall_g, thresold1, thresold2, Plot = False):

    Pos = Shower.pos
    Nantlay, Nplane, Depths = Shower.GetDepths()
    subpos = Pos[Pos[:, 2] == min(Depths)]

    pulse_flags = {
        "isAirSinglePulse": {"x": [], "y": [], "z": [], "tot": []},
        "isIceSinglePulse": {"x": [], "y": [], "z": [], "tot": []},
        "isDoublePulse": {"x": [], "y": [], "z": [], "tot": []},
        "Deltat": {"x": [], "y": [], "z": [], "tot": []}
    }
    keys = ["x", "y", "z", "tot"]
    for i in range(len(Pos)):
        #if(Pos[i, 2] == Depths[4]):
        ClassBumps(Eall_c, Eall_g, thresold1, thresold2, pulse_flags, i)

    return pulse_flags

def GetNtriggered(Epeak_air, Epeak_ice, thresold):
    
    Emax = np.max([Epeak_air, Epeak_ice], axis=0)
    Ntrigger_tot = len(Emax[Emax > thresold])
    thresold_channel =thresold / np.sqrt(3)  # Adjusted threshold for each channel

    Emax_x = np.max([Epeak_air[0], Epeak_ice[0]], axis=0)
    Emax_y = np.max([Epeak_air[1], Epeak_ice[1]], axis=0)
    Emax_z = np.max([Epeak_air[2], Epeak_ice[2]], axis=0)
    Ntrigger_x = len(Emax_x[Emax_x > thresold_channel])
    Ntrigger_y = len(Emax_y[Emax_y > thresold_channel])
    Ntrigger_z = len(Emax_z[Emax_z > thresold_channel])

    return Ntrigger_x, Ntrigger_y, Ntrigger_z, Ntrigger_tot

def GetDoublePulsesMap(PosDoubleBumpsAll, OutputPath, zen=-1):
        # KDE
    from scipy.stats import gaussian_kde
    pos_dp_flat = np.concatenate(PosDoubleBumpsAll)
    x_dp = pos_dp_flat[:, 0]
    y_dp = pos_dp_flat[:, 1]

    xmin, xmax =-350,350
    ymin, ymax = -350, 350
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])


    values = np.vstack([x_dp, y_dp])
    kde = gaussian_kde(values, bw_method=0.3)  # bw_method à ajuster selon la densité
    density = kde(positions).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(6, 5))
    density_normalized = density / np.max(density)
    plt.imshow(density_normalized.T, origin='lower', cmap="hot",
            extent=[xmin, xmax, ymin, ymax], aspect='equal', vmin=0, vmax=1)
    plt.colorbar(label="Double pulses density")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(r"$E=10^{17.5}\,$eV, $\theta=%.d^{\circ}$, Depth = 100 m" %zen, fontsize=12)
    if(zen==-1):
        plt.title(r"E = $10^{17.5}$ eV, All zeniths, Depth = 100 m", fontsize=12)
    #plt.title(rf"Double pulse density map at $\theta = {zenith}^\circ$")
    #plt.scatter(x_all, y_all, c="white", s=5, alpha=0.3)  # All antennas in background
    plt.scatter(x_dp, y_dp, c="yellow", s=15, label="Double pulse", edgecolor="black")
    plt.legend()
    plt.savefig(OutputPath + "DoublePulseDensityMap_zen%.d.pdf" %zen, bbox_inches="tight")
    plt.show()


def GetPulseFlagsData(EnergyAll, ZenithAll, Pos, pulse_flags_all, SelDepth):


    Ndouble_x, Ndouble_y, Ndouble_z, Ndouble_tot = np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll))
    Nsingleair_x, Nsingleair_y, Nsingleair_z = np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll))
    Nsingleice_x, Nsingleice_y, Nsingleice_z = np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll))

    i = 0
    sel = (Pos[:,2] == SelDepth)  # Select the 100m deep layer
    for k in range(len(EnergyAll)):
        if(ZenithAll[k] == 10):
            continue
        
        isAirSinglePulse, isIceSinglePulse, isDoublePulse, Deltat = \
            (pulse_flags_all[i][key] for key in ["isAirSinglePulse", "isIceSinglePulse", "isDoublePulse", "Deltat"])
            
        Nsingleair_x[i] = np.sum(np.array(isAirSinglePulse["x"])[sel])
        Nsingleair_y[i] = np.sum(np.array(isAirSinglePulse["y"])[sel])
        Nsingleair_z[i] = np.sum(np.array(isAirSinglePulse["z"])[sel])

        Nsingleice_x[i] = np.sum(np.array(isIceSinglePulse["x"])[sel])
        Nsingleice_y[i] = np.sum(np.array(isIceSinglePulse["y"])[sel])
        Nsingleice_z[i] = np.sum(np.array(isIceSinglePulse["z"])[sel])

        Ndouble_x[i] = np.sum(np.array(isDoublePulse["x"])[sel])
        Ndouble_y[i] = np.sum(np.array(isDoublePulse["y"])[sel])
        Ndouble_z[i] = np.sum(np.array(isDoublePulse["z"])[sel])
        Ndouble_tot[i] = np.sum(np.array(isDoublePulse["tot"])[sel])

        i = i + 1
    
    return Nsingleair_x, Nsingleair_y, Nsingleair_z, Nsingleice_x, Nsingleice_y, Nsingleice_z,  Ndouble_x, Ndouble_y, Ndouble_z, Ndouble_tot