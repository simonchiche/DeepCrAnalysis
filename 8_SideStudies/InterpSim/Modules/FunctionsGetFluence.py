#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:57:24 2024

@author: chiche
"""

import sys
import os
import numpy as np
import glob
sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
#import Modules.SimParam.GetLayoutScaling
#from GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.integrate import trapz, simps
from scipy.signal import hilbert


def Norm(Ax, Ay, Az):
    
    Norm = np.sqrt(Ax**2 + Ay**2 + Az**2)
    
    return Norm

def GetDepths(Pos):
    
    Nant = len(Pos)
    
    Depths = [Pos[0,2]]
    Depth = Pos[0,2]
    for i in range(Nant):
        if(Pos[i,2]!=Depth):
            Depth = Pos[i,2]
            Depths.append(Depth)
    
    Nlay = len(Depths)
    Nplane = int(Nant/Nlay)
    
    return Nlay, Nplane, Depths
        
    

def LoadTraces(Path):
    
    Cpath = Path + "Coreas"
    Gpath = Path + "Geant"
    Pospath = Path + "Parameters/SIM.list"
    
    
    Nant = len(glob.glob(Cpath +"/*.dat"))
    
    Cdata = dict()#glob.glob(Cpath +"/*.dat")
    Gdata = dict()#glob.glob(Gpath +"/antenna*")
    
    for i in range(Nant):
        
        Cdata[i] = Cpath + "/raw_ch%.d.dat" %i
        
        Gdata[i] = Gpath + "/Antenna%.d.dat" %i
    
        if(not(os.path.exists(Gdata[i]))): print(i)
    #sys.exit()
    
    #Nlay = 42
    Pos =np.genfromtxt(Pospath, delimiter=' ')[:,2:5]
    
    Traces_G = dict()
    Traces_C = dict()
    
    for i in range(Nant):
        print(i)
        if(i%10==0): print("%.d/%.d" %(i,Nant))
        try: 
            Traces_G[i] = np.genfromtxt(Gdata[i])
        except(FileNotFoundError):
            Gdata[i] = Gpath + "/raw_ch%.d.dat" %i
            Traces_G[i] = np.genfromtxt(Gdata[i])
        print(Cdata[i])
        Traces_C[i] = np.loadtxt(Cdata[i])
    
    #Traces_G = np.array(Traces_G)
   #Traces_C = np.array(Traces_C)
    
    
    return Nant, Traces_C, Traces_G, Pos


def LoadairTraces(Path):
    
    Cpath = Path + "Coreas"
    Pospath = Path + "Parameters/SIM.list"
    
    
    Nant = len(glob.glob(Cpath +"/*.dat"))
    
    Cdata = dict()#glob.glob(Cpath +"/*.dat")
    Gdata = dict()#glob.glob(Gpath +"/antenna*")
    
    for i in range(Nant):
        
        Cdata[i] = Cpath + "/raw_ch%.d.dat" %i
        

    #sys.exit()
    
    #Nlay = 42
    Pos =np.genfromtxt(Pospath, delimiter=' ')[:,2:5]
    
    Traces_C = dict()
    
    for i in range(Nant):
        print(i)
        if(i%10==0): print("%.d/%.d" %(i,Nant))
    
        Traces_C[i] = np.loadtxt(Cdata[i])
    
    #Traces_G = np.array(Traces_G)
   #Traces_C = np.array(Traces_C)
    
    
    return Nant, Traces_C, Pos

def GetPeakTraces(Traces, Nant):
    
    Etot = np.zeros(Nant)
    Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    
    for i in range(Nant):
        
        Ex[i] = max(abs(Traces[i][:,1]))
        Ey[i] = max(abs(Traces[i][:,2]))
        Ez[i] = max(abs(Traces[i][:,3]))
        Etot[i] = max(np.sqrt((Traces[i][:,1])**2 + \
             (Traces[i][:,2])**2 + (Traces[i][:,3])**2)) 

    return Ex, Ey, Ez, Etot

def GetIntTraces(Traces, Nant):
    
    Etot = np.zeros(Nant)
    Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    binT = round((Traces[0][1,0] -Traces[0][0,0])*1e10)/1e10
    for i in range(Nant):
        
        Etot_all = np.sqrt(Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2)
        extent = 10000
        peak_id = np.argmax(Etot_all)
        minid = peak_id -extent
        maxid = peak_id + extent
        if(minid<0): minid = 0
        if(maxid>len( Traces[i][:,0])): maxid =len( Traces[i][:,0])
        
        time = np.arange(0, len(Traces[i][minid:maxid,0]))*binT
        
        Ex[i] = simps(abs(hilbert(Traces[i][minid:maxid,1])), time)*1e9
        Ey[i] = simps(abs(hilbert(Traces[i][minid:maxid,2])), time)*1e9
        Ez[i] = simps(abs(hilbert(Traces[i][minid:maxid,3])), time)*1e9
        Etot[i] = Ex[i]**2 + Ey[i]**2 + Ez[i]**2

    return Ex, Ey, Ez, Etot

def GetIntTracesSum(Traces, Nant):
    
    Etot = np.zeros(Nant)
    Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    binT = round((Traces[0][1,0] -Traces[0][0,0])*1e10)/1e10
    
    for i in range(Nant):
        
        Etot_all = np.sqrt(Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2)
        
        extent = 10000
        peak_id = np.argmax(Etot_all)
        minid = peak_id -extent
        maxid = peak_id + extent
        if(minid<0): minid = 0
        if(maxid>len( Traces[i][:,0])): maxid =len( Traces[i][:,0])
        
        
        Ex[i] =  np.sum(binT*abs(hilbert(Traces[i][minid:(maxid-1),2])))*1e9#simps(abs(Traces[i][minid:maxid,1]), Traces[i][minid:maxid,0])
        Ey[i] = np.sum(binT*abs(hilbert(Traces[i][minid:(maxid-1),2])))*1e9
        Ez[i] = np.sum(binT*abs(hilbert(Traces[i][minid:(maxid-1),2])))*1e9
        Etot[i] = Ex[i]**2 + Ey[i]**2 + Ez[i]**2
        
        
        #plt.plot(Traces[i][minid:maxid,0], Traces[i][minid:maxid,2])
        #plt.show()
    return Ex, Ey, Ez, Etot


def Traces_cgs_to_si(Traces):
    
    Nant = len(Traces)
    k = 29979.24588*1e6 # from statVolt/cm to µV/m 
    
    for i in range(Nant):
        #print(Traces[i].shape, i)
        Traces[i][:,1:] = Traces[i][:,1:]*k
    
    return Traces

def CorrectScaling(Depth, E, theta):
    
      
      k0 = GetDepthScaling(100)*GetEnergyScaling(0.1)/np.cos(47*np.pi/180.0)
    
          
      k = GetDepthScaling(Depth)*GetEnergyScaling(E)/np.cos(theta*np.pi/180.0)
    
      return k/k0
  
def CorrectLength(Traces_C, Correction):
    if(Correction):
        for i in range(len(Traces_C)):
            if(i%10==0): print(i)
            np.savetxt("/Users/chiche/Desktop/DeepCrSearch"\
             + "/Simulations/DeepCrLib/Rectangle_Proton_0.0316_10_0_1/NewCoreas/raw_ch%.d.dat" \
             %i, Traces_C[i][:5000,:])
    


def CombineTraces(Nant, Traces_C, Traces_G):
    TracesTot = dict()

    for i in range(Nant):
        if(i%100==0): print(i)

        binT = round((Traces_C[i][1,0] -Traces_C[i][0,0])*1e10)/1e10
        
        Ex = [] 
        Ey = [] 
        Ez = [] 
        Twindow = []
        
        if(max(Traces_C[i][:,0])< min(Traces_G[i][:,0])):   
            
            Ex = np.concatenate((Traces_C[i][:,1], [0,0], Traces_G[i][:,1]))
            Ey = np.concatenate((Traces_C[i][:,2],[0,0], Traces_G[i][:,2]))
            Ez = np.concatenate((Traces_C[i][:,3],[0,0], Traces_G[i][:,3]))
            gap1 = Traces_C[i][-1,0] + binT
            gap2 = Traces_G[i][0,0] - binT
            Twindow = np.concatenate((Traces_C[i][:,0], [gap1, gap2], Traces_G[i][:,0]))
            
            TracesTot[i] = np.array([Twindow, Ex, Ey, Ez]).T
            
        if((max(Traces_C[i][:,0])> min(Traces_G[i][:,0])) &\
           (max(Traces_C[i][:,0])<max(Traces_G[i][:,0]))):
            
            argminG = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][0,0]))
            argmaxC = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][-1,0]))
            
            Twindow = np.concatenate((Traces_C[i][:argminG,0], Traces_G[i][:,0]))
            
            Ex_air, Ey_air, Ez_air = Traces_C[i][:argminG,1], Traces_C[i][:argminG,2],\
            Traces_C[i][:argminG,3]
            
            tmin = Traces_C[i][argminG,0]
            t = tmin
            k = 0
            GapTrace = False
            Ex_air_ice, Ey_air_ice, Ez_air_ice = [], [], []
    
            for j in range(argminG, len(Traces_C[i][:,0]), 1):
                argmint = np.argmin(abs(t - Traces_G[i][:,0]))
                if(abs(t - Traces_G[i][argmint,0])<=binT):
                    
                    Ex_air_ice.append(Traces_C[i][j,1] + Traces_G[i][k,1])
                    Ey_air_ice.append(Traces_C[i][j,2] + Traces_G[i][k,2])
                    Ez_air_ice.append(Traces_C[i][j,3] + Traces_G[i][k,3])
                    
                    k = k +1
                    t = t + binT
                    kmaxG =k
                    
                else:
                    Ex_air_ice.append(Traces_C[i][j,1])
                    Ey_air_ice.append(Traces_C[i][j,2])
                    Ez_air_ice.append(Traces_C[i][j,3])
                    
                    k = k +1
                    t = t + binT
                    GapTrace = True
            
            if(GapTrace): Twindow = np.concatenate((Traces_C[i][:,0], Traces_G[i][kmaxG:,0]))
                
            Ex_ice, Ey_ice, Ez_ice = Traces_G[i][kmaxG:,1], \
            Traces_G[i][kmaxG:,2], Traces_G[i][kmaxG:,3]
            
            Ex = np.concatenate((Ex_air, Ex_air_ice, Ex_ice))
            Ey = np.concatenate((Ey_air, Ey_air_ice, Ey_ice))
            Ez = np.concatenate((Ez_air, Ez_air_ice, Ez_ice))
    
            TracesTot[i] = np.array([Twindow, Ex, Ey, Ez]).T
            
        if((max(Traces_C[i][:,0])>max(Traces_G[i][:,0]))):
            argminG = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][0,0]))
            argmaxC = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][-1,0]))
        
            Twindow = Traces_C[i][:,0]
        
            Ex_air, Ey_air, Ez_air = Traces_C[i][:argminG,1], Traces_C[i][:argminG,2],\
            Traces_C[i][:argminG,3]
            
            tmin = Traces_C[i][argminG,0]
            t = tmin
            k = 0
            
            Ex_air_ice, Ey_air_ice, Ez_air_ice = [], [], []
            
            for j in range(argminG, argmaxC, 1):
                
                argmint = np.argmin(abs(t - Traces_G[i][:,0]))
                if(abs(t - Traces_G[i][argmint,0])<=binT):
                    
                    Ex_air_ice.append(Traces_C[i][j,1] + Traces_G[i][k,1])
                    Ey_air_ice.append(Traces_C[i][j,2] + Traces_G[i][k,2])
                    Ez_air_ice.append(Traces_C[i][j,3] + Traces_G[i][k,3])
                    
                    k = k +1
                    t = t + binT
                    
                else:
                    Ex_air_ice.append(Traces_C[i][j,1])
                    Ey_air_ice.append(Traces_C[i][j,2])
                    Ez_air_ice.append(Traces_C[i][j,3])
                    
                    k = k +1
                    t = t + binT
                
            Ex_air2, Ey_air2, Ez_air2 = Traces_C[i][argmaxC:,1], \
            Traces_C[i][argmaxC:,2], Traces_C[i][argmaxC:,3]
            
            Ex = np.concatenate((Ex_air, Ex_air_ice, Ex_air2))
            Ey = np.concatenate((Ey_air, Ey_air_ice, Ey_air2))
            Ez = np.concatenate((Ez_air, Ez_air_ice, Ez_air2))
    
            TracesTot[i] = np.array([Twindow, Ex, Ey, Ez]).T
    return TracesTot 

def GetAntLine(Pos, Nplane):
    k =0
    for i in range(len(Pos)):
    
        if(Pos[i,0]*Pos[i+1,0]<0):
            k = k +1
        if(k ==2):
            NantLine = (i+1)
            break
    Nlines = int(Nplane/NantLine)
    return NantLine, Nlines



def GetRadioExtent(Nlay, Nplane, Pos, Etot_int):
    
# Boucle sur le nombre de layers. Pour chaque layer on trouve la zone ou on a 99% de l'énergie
# Garder la ligne avec l'intégrale la plus grande et ensuite classer par |x|
# Code specifique à phi =0, ex_max selon l'axe x
#Nlay =5
# Nplane = 729
# Depth = [100, 80, 60, 40, 0]


    NantLine, Nlines = GetAntLine(Pos, Nplane)

    extent = np.zeros(Nlay)
    maxpos = np.zeros(Nlay)
    xminlay = np.zeros(Nlay)
    xmaxlay = np.zeros(Nlay)
    
    for i in range(Nlay):
        IntAll = np.zeros(Nlines)
        for j in range(Nlines):
            argmin = j*NantLine + i*Nplane
            argmax = (j+1)*NantLine + i*Nplane
            IntAll[j] = np.sum(Etot_int[argmin:argmax])
        
        Lmax = np.argmax(IntAll)
        
        argfracmin = Lmax*NantLine + i*Nplane
        argfracmax = (Lmax+1)*NantLine + i*Nplane   
        
        #plt.scatter(Pos[argfracmin:argfracmax, 0], Etot_int[argfracmin:argfracmax])
        #plt.show()
        
        Frac = Etot_int[argfracmin +np.argsort\
                        (Etot_int[argfracmin:argfracmax])[::-1]]/IntAll[Lmax]
        SumFrac = np.cumsum(Frac)
        ilow = np.searchsorted(SumFrac, 0.99)
        xlow= Pos[argfracmin + ilow, 0]
        imax = np.argmax(Etot_int[argfracmin:argfracmax])
        xmax = Pos[argfracmin + imax, 0]
        maxpos[i] = xmax
        xminlay[i] = min(Pos[i*Nplane:(i+1)*Nplane,0])
        xmaxlay[i] = max(Pos[i*Nplane:(i+1)*Nplane,0])
        extent[i]= int(abs(xmax - xlow))
        
        radioextent = 2*extent
        simextent = abs(xmaxlay-xminlay)
        
        # amplitude along the line with the highest integrated signal
        #plt.scatter(Pos[Lmax*NantLine:(Lmax+1)*NantLine,0],  \
                #Etot_int[Lmax*NantLine:(Lmax+1)*NantLine])

    return radioextent, simextent, extent, maxpos, xminlay, xmaxlay


def GetEnergyFracvsRcore(Nlay, Nplane, Pos, Etot_int):
    
# Boucle sur le nombre de layers. Pour chaque layer on trouve la zone ou on a 99% de l'énergie
# Garder la ligne avec l'intégrale la plus grande et ensuite classer par |x|
# Code specifique à phi =0, ex_max selon l'axe x
#Nlay =5
# Nplane = 729
# Depth = [100, 80, 60, 40, 0]


    NantLine, Nlines = GetAntLine(Pos, Nplane)

    extent = np.zeros(Nlay)
    maxpos = np.zeros(Nlay)
    xminlay = np.zeros(Nlay)
    xmaxlay = np.zeros(Nlay)
    
    IntFrac = dict()
    IntFracLine = 0
    Nsample = 100
    for i in range(Nlay):
        IntAll = np.zeros(Nlines)
        for j in range(Nlines):
            argmin = j*NantLine + i*Nplane
            argmax = (j+1)*NantLine + i*Nplane
            IntAll[j] = np.sum(Etot_int[argmin:argmax])

            for k in range(Nsample):
                IntFracLine

        
        Lmax = np.argmax(IntAll)
        
        argfracmin = Lmax*NantLine + i*Nplane
        argfracmax = (Lmax+1)*NantLine + i*Nplane   
        
        #plt.scatter(Pos[argfracmin:argfracmax, 0], Etot_int[argfracmin:argfracmax])
        #plt.show()
        
        Frac = Etot_int[argfracmin +np.argsort\
                        (Etot_int[argfracmin:argfracmax])[::-1]]/IntAll[Lmax]
        SumFrac = np.cumsum(Frac)
        ilow = np.searchsorted(SumFrac, 0.99)
        xlow= Pos[argfracmin + ilow, 0]
        imax = np.argmax(Etot_int[argfracmin:argfracmax])
        xmax = Pos[argfracmin + imax, 0]
        maxpos[i] = xmax
        xminlay[i] = min(Pos[i*Nplane:(i+1)*Nplane,0])
        xmaxlay[i] = max(Pos[i*Nplane:(i+1)*Nplane,0])
        extent[i]= int(abs(xmax - xlow))
        
        radioextent = 2*extent
        simextent = abs(xmaxlay-xminlay)
        
        # amplitude along the line with the highest integrated signal
        #plt.scatter(Pos[Lmax*NantLine:(Lmax+1)*NantLine,0],  \
                #Etot_int[Lmax*NantLine:(Lmax+1)*NantLine])

    return radioextent, simextent, extent, maxpos, xminlay, xmaxlay