#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:43:13 2024

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt


def EfieldMap(Pos, Nlay, Nplanes, E, sim, save, energy, theta, path):
    
    for i in range(Nlay):
        antmin = i*Nplanes
        antmax = (i+1)*Nplanes
        plt.scatter(Pos[antmin:antmax,0], Pos[antmin:antmax,1], \
                    c= E[antmin:antmax], cmap = "jet", s=10)
        cbar = plt.colorbar()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        cbar.set_label("E [$\mu V/m$]")
        depth = 3216 - Pos[Nplanes*i,2]
        #plt.xlim(-250,250)
        #plt.ylim(-200,200)
        plt.legend(["Depth = %.f m" %(depth)], loc ="upper right")
        plt.title(sim + " map (E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =14)
        if(save):
            plt.savefig\
            (path + sim + "EfieldMap_E%.2f_th%.1f_depth%1.f.pdf" \
             %(energy, theta, depth), bbox_inches = "tight")
        plt.show()


def PlotLDF(Pos, Nplane, E, sim, Nlay):
    
    for i in range(Nlay):
        antmin = i*Nplane
        antmax = int((i+0.5)*Nplane)
        plt.plot(Pos[antmin:antmax,0], E[antmin:antmax])
        #plt.scatter(Pos[:int(Nlay/2),0], EtotC[int(Nlay/2):Nlay])
        plt.xlabel("x [m]")
        plt.ylabel("E [$\mu V/m$]")
        plt.title(sim + " LDF")
        plt.show()
    
    
def PlotTraces(Traces, start, stop):
    
    for i in range(start, stop, 1):
    
        plt.plot(Traces[i][:, 0]*1e9,Traces[i][:, 3])
        plt.xlabel("Time [ns]")
        plt.ylabel("E [$\mu V/m$]")
        plt.title(r"Proton  E = 0.01 EeV - $\theta = 0^{\circ}$ $\phi = 0^{\circ}$")
        plt.show()


def PlotGivenTrace(Traces, arg, ax):
    
    if(ax == "x"): ax =1
    if(ax == "y"): ax =2
    if(ax == "z"): ax =3
    plt.plot(Traces[arg][:, 0]*1e9,Traces[arg][:, ax])
    plt.xlabel("Time [ns]")
    plt.ylabel("E [$\mu V/m$]")
    #plt.title(r"Proton  E = 0.01 EeV - $\theta = 0^{\circ}$ $\phi = 0^{\circ}$")
    plt.show()

def plot_polarisation(vxb, vxvxb, Etot_sp, Evxb, Evxvxb, path):
    
    # function that plots the normalised polarisation in a given plane
    
    #r = np.sqrt(Evxb**2 + Evxvxb**2)
    plt.scatter(vxb, vxvxb, color = "white")
    #cbar = plt.colorbar()
    plt.xlabel('v x b [m]')
    plt.ylabel('v x (v x b) [m]')
    #plt.xlim(-200,200)
    #plt.ylim(-200,200)
    #plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    #cbar.set_label(r"$ E\ [\mu V/m]$")
    plt.quiver(vxb, vxvxb, Evxb/20, Evxvxb/20)
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.tight_layout()
    plt.savefig(path + 'polarisation_sp.png', dpi = 500)
    plt.show()
    
def PlotMaxTraces(Traces, Etot, NantPlot):
    
    MaxId = np.argsort(abs(Etot))
    for i in range(NantPlot):
        arg = MaxId[-(i+1)]
        PlotTraces(Traces, arg, arg+1)
        
def PlotAllTraces(Nant, Traces, Threshold, Nmax):
    k =0
    
    for i in range(Nant):
        if(k<Nmax):
            if(max(abs(Traces[i][:, 2]))>Threshold):
                PlotTraces(Traces, i, i+1)
                print(i)
                k = k +1

def PlotLayer(Pos, k, Nplane, path):
    plt.scatter(Pos[k*Nplane:(k+1)*Nplane,0], Pos[k*Nplane:(k+1)*Nplane,1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(path + "RectGrid.pdf")
    plt.show()
    

def PlotAllChannels(Traces, ID):
   
    plt.plot(Traces[ID][:, 0]*1e9,Traces[ID][:, 1], label ="x-channel")
    plt.plot(Traces[ID][:, 0]*1e9,Traces[ID][:, 2], label ="y-channel")
    plt.plot(Traces[ID][:, 0]*1e9,Traces[ID][:, 3], label ="z-channel")
    plt.xlabel("Time [ns]")
    plt.ylabel("E [$\mu V/m$]")
    plt.legend()
    #plt.xlim(500,700)
    plt.xlim(630,650)
    #plt.title(r"Proton  E = 0.01 EeV - $\theta = 0^{\circ}$ $\phi = 0^{\circ}$")
    plt.tight_layout()
    plt.savefig("/Users/chiche/Desktop/AllChannelsInIce_%.d.pdf" %ID)
    plt.show()
    
    
    


