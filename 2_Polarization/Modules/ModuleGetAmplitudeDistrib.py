import numpy as np

def GetAmplitudeDistribution(Etot, EnergyAll, PosAll, SelE, SelDepth):

    MaskE = (EnergyAll == SelE)
    #MaskDepth = (PosAll[MaskE].reshape(-1, 3)[:,2] == SelDepth)
    Emasked = Etot[MaskE]

    EmaskedDepth = []
    for i in range(len(Emasked)):

        MaskDepth = PosAll[i][:,2] == SelDepth
        EmaskedDepth.append(Emasked[i][MaskDepth])
    
    return np.array(EmaskedDepth).flatten()