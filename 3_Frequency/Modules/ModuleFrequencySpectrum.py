#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:43:13 2024

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import gaussian_filter

def compute_spectrum(
    t, E,
    window='hann',
    detrend='mean',      # 'none' | 'mean' | 'linear' | 'poly'
    poly_deg=2,          # used if detrend='poly'
    fmin=None,           # drop frequencies below fmin in the return
    onesided=True        # for real-valued signals, keep positive freqs
):
    t = np.asarray(t)
    E = np.asarray(E)
    # --- 1) Ensure roughly uniform sampling (for this simple FFT path)
    dt = np.median(np.diff(t))
    # --- 2) Detrend / de-mean
    if detrend == 'mean':
        E = E - np.mean(E)
    elif detrend == 'linear':
        p = np.polyfit(t, E, 1)
        E = E - np.polyval(p, t)
    elif detrend == 'poly':
        p = np.polyfit(t, E, poly_deg)
        E = E - np.polyval(p, t)
    elif detrend == 'none':
        pass
    else:
        raise ValueError("detrend must be 'none'|'mean'|'linear'|'poly'.")

    # --- 3) Windowing (reduce leakage from the finite record)
    N = len(E)
    if window in (None, 'rect'):
        w = np.ones(N)
    elif window == 'hann':
        w = np.hanning(N)
    elif window == 'hamming':
        w = np.hamming(N)
    else:
        raise ValueError("window must be 'rect'|'hann'|'hamming' or None")
    Ew = E * w

    # --- 4) FFT and frequency axis
    if onesided:
        X = np.fft.rfft(Ew)
        f = np.fft.rfftfreq(N, d=dt)
    else:
        X = np.fft.fft(Ew)
        f = np.fft.fftfreq(N, d=dt)

    # --- 5) Amplitude scaling (simple, readable)
    A = np.abs(X) / N
    if onesided:
        if N % 2 == 0:
            A[1:-1] *= 2.0
        else:
            A[1:] *= 2.0

    # --- 6) Optional low-frequency cut
    if fmin is not None:
        keep = f >= fmin
        f, A = f[keep], A[keep]
    
    
    pos = f[:N // 2]      # same as np.fft.fftfreq then [:N//2]
    Amp = A[:N // 2]

    return pos, Amp

def GourierTransform(signal, dt):
    N = len(signal)
    # Compute the Fourier Transform using FFT
    fft_result = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, dt)

    # Compute the magnitude of the FFT result
    fft_magnitude = np.abs(fft_result) / N

    # Since FFT output is symmetrical, take only the positive half
    positive_freqs = fft_freqs[:N // 2]
    positive_magnitude = fft_magnitude[:N // 2]

    return positive_freqs, positive_magnitude


def PlotAllSpectra(Traces):
    for i in range(len(Traces)):
        time = Traces[i][:,0]
        signal =Traces[i][:,2] 
        #Ex, Ey, Ez =Traces[i][:,1], Traces[i][:,1], Traces[i][:,3]
        #signal = np.sqrt(Ex*2 + Ey**2 + Ez**2)   
        f, A = compute_spectrum(time, signal, window='rect',     detrend='none',    # do not remove mean
            onesided=False     # full FFT; we'll slice positive half like you did
        )

        #plt.figure(figsize=(10, 5))
        plt.plot(f/1e6, A)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Amplitude spectrum")
        plt.grid()
        plt.xlim(0, 1500)
    plt.show()
    return

def PlotAllSignals(Traces):
    for i in range(len(Traces)):
        time = Traces[i][:,0]
        signal =Traces[i][:,2] 
        #plt.figure(figsize=(10, 5))
        plt.plot(time, signal)
        plt.xlabel("temps")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.xlim(1.18e-6, 1.2e-6)
    plt.show()
    return


def PlotFrequencyHeatmap(Trace_Surface, radius, radius_idx, Shower, OutputPath, label, rmax=None, Save=False, merge_factor=1):
    """
    Trace_Surface : dict {i: array Nx3} contenant [t, Ex, Ey, Ez]
    radius        : tableau (Nant,) des distances au core
    radius_idx    : indices triés (np.argsort(radius))
    rmax          : rayon maximal à afficher (m). Si None -> tous.
    merge_factor  : pour éventuellement fusionner plusieurs rayons consécutifs (ex: 2 => fusionne 2 anneaux)
    """

    NFREQ = 300
    FMAX_MHZ = 1500.0
    FMIN_MHZ = 20.0
    LOG_GRID = True

    # ---------- 1) Tri et filtrage par rayon ----------
    order = radius_idx
    r_sorted = radius[order]
    if rmax is not None:
        sel = r_sorted <= rmax
        order = order[sel]
        r_sorted = r_sorted[sel]
    Nant = len(order)

    # ---------- 2) Grille de fréquences ----------
    if LOG_GRID:
        fgrid_mhz = np.logspace(np.log10(max(FMIN_MHZ, 1e-3)),
                                np.log10(FMAX_MHZ), NFREQ)
    else:
        fgrid_mhz = np.linspace(0.0, FMAX_MHZ, NFREQ)

    # ---------- 3) Calcul spectres par antenne ----------
    H = np.full((Nant, NFREQ), np.nan)
    for ii, i in enumerate(order):
        time = Trace_Surface[i][:, 0]
        signal = Trace_Surface[i][:, 2]
        #Ex, Ey, Ez =Trace_Surface[i][:,1], Trace_Surface[i][:,1], Trace_Surface[i][:,3]
        #signal = np.sqrt(Ex*2 + Ey**2 + Ez**2)   
        f_hz, A = compute_spectrum(time, signal,
                                   window='rect', detrend='none', onesided=False)
        mpos = f_hz > 0 #only positive frequencies
        f_mhz = f_hz[mpos] / 1e6 # convert to MHz
        A = A[mpos] 

        #Amax = np.max(A)
        #if Amax > 0:
        #    A /= Amax

        orderf = np.argsort(f_mhz)
        f_mhz = f_mhz[orderf]
        A = A[orderf]
        valid = (fgrid_mhz >= f_mhz[0]) & (fgrid_mhz <= f_mhz[-1])
        H[ii, valid] = np.interp(fgrid_mhz[valid], f_mhz, A)

    # ---------- 4) Détermination automatique des bins ----------
    # Rayon unique (valeurs discrètes de ta grille polaire)
    unique_r = np.unique(np.round(r_sorted, 6))  # arrondi pour tolérance numérique

    # Si plusieurs antennes à même rayon, on les moyenne
    Nunique = len(unique_r)
    H_binned = np.full((Nunique, NFREQ), np.nan)

    for i, r_val in enumerate(unique_r):
        sel = np.isclose(r_sorted, r_val, atol=1e-3)
        if np.any(sel):
            H_binned[i, :] = np.nanmean(H[sel, :], axis=0)

    # ---------- 5) Optionnel : fusion de plusieurs rayons voisins ----------
    if merge_factor > 1:
        Nmerged = Nunique // merge_factor
        H_merged = np.full((Nmerged, NFREQ), np.nan)
        r_merged = np.full(Nmerged, np.nan)
        for i in range(Nmerged):
            start = i * merge_factor
            stop = start + merge_factor
            H_merged[i, :] = np.nanmean(H_binned[start:stop, :], axis=0)
            r_merged[i] = np.nanmean(unique_r[start:stop])
        H_binned = H_merged
        unique_r = r_merged

    # ---------- 6) Construction des edges ----------
    f_edges = np.empty(NFREQ + 1)
    f_edges[1:-1] = 0.5 * (fgrid_mhz[1:] + fgrid_mhz[:-1])
    f_edges[0] = fgrid_mhz[0] - (fgrid_mhz[1] - fgrid_mhz[0])
    f_edges[-1] = fgrid_mhz[-1] + (fgrid_mhz[-1] - fgrid_mhz[-2])

    # y_edges selon les rayons réels
    dr_local = np.diff(unique_r)
    r_edges = np.concatenate([[unique_r[0] - dr_local[0]/2],
                              unique_r[:-1] + dr_local/2,
                              [unique_r[-1] + dr_local[-1]/2]])

    # ---------- 7) Affichage ----------
    plt.figure(figsize=(8, 4.5))
    F, R = np.meshgrid(f_edges, r_edges)
    H_smooth = H_binned = gaussian_filter(H_binned, sigma=(1.0, 2))
    denom = np.nanmax(H_smooth)
    pc = plt.pcolormesh(F, R, H_smooth/denom, shading='auto', cmap='viridis', rasterized=True)

    if LOG_GRID:
        plt.xscale('log')

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Distance to core [m]')
    plt.colorbar(pc, label='Amplitude (normalized)')
    #plt.title('Frequency–distance heatmap')
    plt.title(label + " E=$10^{17.5}\,$eV" +f", $\\theta$={Shower.zenith:.1f}$^\circ$")
    plt.tight_layout()
    #plt.ylim(5,150)
    if Save:
        energy=Shower.energy
        theta=Shower.zenith
        plt.savefig(OutputPath + label + "FrequencyHeatmap" + f"E{int(energy)}_th{int(theta)}.pdf", bbox_inches='tight', dpi=200)
    plt.show()

    return fgrid_mhz, unique_r, H_binned

def GetPeakTraces(Traces):

    Nant = len(Traces)
    #print(Nant)
    Etot = np.zeros(Nant)
    Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    
    for i in range(Nant):
        
        Ex[i] = max(abs(Traces[i][:,1]))
        Ey[i] = max(abs(Traces[i][:,2]))
        Ez[i] = max(abs(Traces[i][:,3]))
        Etot[i] = max(np.sqrt((Traces[i][:,1])**2 + \
            (Traces[i][:,2])**2 + (Traces[i][:,3])**2)) 
        
    return Ex, Ey, Ez, Etot

def PlotAllSpectra_rbin(Traces, title, OutputPath):
    current_max = 0
    for i in range(len(Traces)):
        time = Traces[i][:,0]
        signal =Traces[i][:,2] 
        #Ex, Ey, Ez =Traces[i][:,1], Traces[i][:,1], Traces[i][:,3]
        #signal = np.sqrt(Ex*2 + Ey**2 + Ez**2)   
        f, A = compute_spectrum(time, signal, window='rect',     detrend='none',    # do not remove mean
            onesided=False     # full FFT; we'll slice positive half like you did
        )
        current_max = max(max(A), current_max)
    for i in range(len(Traces)):
        time = Traces[i][:,0]
        signal =Traces[i][:,2] 
        #Ex, Ey, Ez =Traces[i][:,1], Traces[i][:,1], Traces[i][:,3]
        #signal = np.sqrt(Ex*2 + Ey**2 + Ez**2)   
        f, A = compute_spectrum(time, signal, window='rect',     detrend='none',    # do not remove mean
            onesided=False     # full FFT; we'll slice positive half like you did
        )
        #plt.figure(figsize=(10, 5))
        plt.plot(f/1e6, A/current_max)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Normalized amplitude")
        
    plt.title(title, fontsize=14)
    plt.xlim(0, 1500)
    plt.grid()
    plt.savefig(OutputPath + f"spectra_{title}_rbin.pdf", bbox_inches = "tight")
    plt.show()
    return
