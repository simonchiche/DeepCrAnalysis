import numpy as np
import matplotlib.pyplot as plt
import pickle
import subprocess
import os
#from Modules.Fluence.FunctionsGetFluence import LoadTraces
from ModulePlotDumbleBumps import PlotTrace, PlotDoubleBumpTrace
import sys

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
                pulse_flags["isDoublePulse"][key].append(True)
                Deltat = Eair[4][i] - Eice[4][i] 
                pulse_flags["Deltat"][key].append(Deltat)
            else:
                pulse_flags["isDoublePulse"][key].append(False)
                pulse_flags["Deltat"][key].append(np.nan)

        return pulse_flags


def GetDoubleBumps(Shower, Eall_c, Eall_g, thresold1, thresold2, Plot = False):

    Pos = Shower.pos
    Nantlay, Nplane, Depths = Shower.GetDepths()
    subpos = Pos[Pos[:, 2] == Depths[4]]

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