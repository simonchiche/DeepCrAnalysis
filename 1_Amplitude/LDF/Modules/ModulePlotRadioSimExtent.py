#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:31:01 2024

@author: chiche
"""

import numpy as np
import matplotlib.pyplot as plt


def PlotMaxLDF(PosAirLDF, PosIceLDF, Eair_LDF, Eice_LDF, channel, Depth, Shower, OutputPath):


    if(channel == "x"): k= 0
    if(channel == "y"): k= 1

    plt.scatter(PosAirLDF[:,k], Eair_LDF,label = "in-air")
    plt.scatter(PosIceLDF[:,k], Eice_LDF,label = "in-ice")
    plt.yscale("log")
    xlab = channel + " [m]"
    plt.xlabel(xlab)
    plt.ylabel("LDF [$\mu V/m$]")
    plt.grid()
    plt.legend()
    plt.title("E=%.2f EeV, $\\theta=%.d^{\circ}$, |z| =%.d" %(Shower.energy, Shower.zenith, Depth))
    plt.savefig(OutputPath + "LDF_E%.2f_th%.d_|z|%.d_" %(Shower.energy, Shower.zenith, Depth) + channel + ".pdf", bbox_inches = "tight")
    plt.show()

def PlotRadioSimExtent(Depths, radioextent, simextent):
    
    # radio extent and simulation extent vs depth
    plt.plot(3216-np.array(Depths), radioextent, label = "radio")
    plt.plot(3216-np.array(Depths), simextent, label = "sim")
    plt.xlabel("Depth [m]")
    plt.ylabel("Extent [m]")
    #plt.title("E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$" %(energy, theta), size =14)
    plt.legend()
    #plt.savefig(OutputPath + "LayoutExtent_E%.2f_th%.1f.pdf" \
     #            %(energy, theta), bbox_inches = "tight")
    plt.show()


def PlotAirIceExtent(Depths, airextent, iceextent, simextent, energy, theta):
    
    # radio extent and simulation extent vs depth
    plt.plot(3216-np.array(Depths), airextent, label = "in-air")
    plt.plot(3216-np.array(Depths), iceextent, label = "in-ice")
    plt.plot(3216-np.array(Depths), simextent, label = "sim")
    plt.xlabel("Depth [m]")
    plt.ylabel("Extent [m]")
    #plt.title("E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$" %(energy, theta), size =14)
    plt.legend()
    plt.savefig("/Users/chiche/Desktop/" + "AirIce_E%.2f_th%.1f.pdf" \
                 %(energy, theta), bbox_inches = "tight")
    plt.show()

def PlotFillingFactor(Depths, radioextent, simextent):

    # filling factor
    plt.scatter(3216-np.array(Depths), radioextent/simextent)
    plt.xlabel("Depth [m]")
    plt.ylabel("Filling factor [%]")
    #plt.title("E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$" %(energy, theta), size =14)
    #plt.savefig(OutputPath + "FillingFactor_E%.2f_th%.1f.pdf" \
    #             %(energy, theta), bbox_inches = "tight")
    plt.show()