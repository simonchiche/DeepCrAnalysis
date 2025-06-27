import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

def PlotSurfaceFootprint(footprint, Shower, Save, BatchID, OutputPath):
    Energy, zenith, azimuth = Shower.energy, Shower.zenith, Shower.azimuth
    plt.figure(figsize=(6, 6))
    plt.plot(footprint[:, 0], footprint[:, 1], label='Surface footprint', color='blue')
    #plt.scatter(Xmax[0], Xmax[1], color='red', label='Xmax projection')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'E ={Energy} EeV, $\\theta$={zenith}°, $\\varphi$={azimuth}°', fontsize=12)
    plt.legend(loc = "upper left")
    plt.grid(True)
    plt.savefig(OutputPath +  f"{BatchID}_GroundFootprint_E{Energy}_theta{zenith}_phi{azimuth}.pdf", bbox_inches = "tight") if Save else None
    plt.show()


def CompareFootprints(footprint, Shower, all_xray, all_yray, Save, BatchID, OutputPath):

    Energy, zenith, azimuth = Shower.energy, Shower.zenith, Shower.azimuth

    # Display the footprint
    plt.figure(figsize=(6, 6))
    plt.plot(footprint[:, 0], footprint[:, 1], label='Surface footprint', color='blue')
    plt.plot(all_xray, all_yray, color='red', label='$100\,$m-deep footprint')
    #plt.scatter(0, 0, color='red', label='Projection verticale de Xmax')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'E ={Energy} EeV, $\\theta$={zenith}°, $\\varphi$={azimuth}°', fontsize=12)
    plt.legend(loc = "upper left")    
    plt.grid(True)
    plt.savefig(OutputPath + f"{BatchID}_GroundvsDeepFootprint_E{Energy}_theta{zenith}_phi{azimuth}.pdf", bbox_inches = "tight") if Save else None
    plt.show()

def PlotFootprintPolygons(polygon_surface, polygon_deep, Shower, Save, BatchID, OutputPath):

    Energy, zenith, azimuth = Shower.energy, Shower.zenith, Shower.azimuth
    difference = polygon_deep.difference(polygon_surface)
    # Area of each zone
    area_depth = polygon_deep.area
    area_outside = difference.area
    fraction_outside = area_outside / area_depth if area_depth > 0 else 0


    # Plot both polygons
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot surface footprint in blue
    plot_polygon(polygon_surface, ax=ax, add_points=False, facecolor='lightblue', edgecolor='blue', alpha=0.4)
    # Plot depth footprint in red
    plot_polygon(polygon_deep, ax=ax, add_points=False, facecolor='lightcoral', edgecolor='red', alpha=0.4)

    plt.title(f'E ={Energy} EeV, $\\theta$={zenith}°, $\\varphi$={azimuth}°', fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    plt.grid(True)
    plt.legend(["Surface", "100m-deep"], loc = "upper right")
    plt.text(
        0.05, 0.95,  # x and y in axes coordinates (0 to 1)
        f"Deep triggers only: {fraction_outside:.2%}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    plt.savefig(OutputPath + f"{BatchID}_GroundvsDeepAreaFootprint_E{Energy}_theta{zenith}_phi{azimuth}.pdf", bbox_inches = "tight") if Save else None
    plt.show()


def PlotAmplitudeDilution(all_xray_samples, all_yray_samples, SurfaceDeepRatio, Shower, Save, BatchID, OutputPath):
 
    Energy, zenith, azimuth = Shower.energy, Shower.zenith, Shower.azimuth
    # Amplitude dilution scatter plot
    plt.scatter(all_xray_samples, all_yray_samples, c=SurfaceDeepRatio*100, cmap='jet', s=1, label='Ray paths')
    cbar = plt.colorbar()
    cbar.set_label('$[\%]$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'E ={Energy} EeV, $\\theta$={zenith}°, $\\varphi$={azimuth}°', fontsize=12)
    plt.savefig(OutputPath + f"{BatchID}_AmplitudeDistribuition_E{Energy}_theta{zenith}_phi{azimuth}.pdf", bbox_inches = "tight") if Save else None
    plt.show()

def PlotTimeDelayDistribution(all_dt_samples,  Shower, Save, BatchID, OutputPath, Nbins=10):
    """
    Plot the histogram of time delay distribution.
    """
    Energy, zenith, azimuth = Shower.energy, Shower.zenith, Shower.azimuth
    #plt.figure(figsize=(8, 6))
    plt.hist(np.array(all_dt_samples)*1e9, bins=Nbins, color='sandybrown', alpha=0.7, edgecolor='black')
    plt.xlabel('Time Delay [ns]')
    plt.ylabel('Count')
    plt.title(f'E ={Energy} EeV, $\\theta$={zenith}°, $\\varphi$={azimuth}°', fontsize=12)
    plt.grid(True)
    plt.savefig(OutputPath + f"{BatchID}_TimeDelayDistribution_E{Energy}_theta{zenith}_phi{azimuth}.pdf", bbox_inches = "tight") if Save else None
    plt.show()

def PlotSampledFootprint(footprint_samples,  Shower, Save, BatchID, OutputPath):
    """
    Plot the sampled surface footprint.
    """
    Energy, zenith, azimuth = Shower.energy, Shower.zenith, Shower.azimuth
    plt.figure(figsize=(6, 6))
    plt.scatter(footprint_samples[:, 0], footprint_samples[:, 1], color='blue', s=1, label='Ground sampled positions')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'E ={Energy} EeV, $\\theta$={zenith}°, $\\varphi$={azimuth}°', fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(OutputPath + f"{BatchID}_SampledFootprint_E{Energy}_theta{zenith}_phi{azimuth}.pdf", bbox_inches = "tight") if Save else None
    plt.show()


def PlotXmaxDistanceVsZenith(EnergyAll, ZenithAll, XmaxDistAll):
    """
    Plot the Xmax distance as a function of the zenith angle for different energies.
    """
    plt.figure(figsize=(10, 6))
    Ebins = np.unique(EnergyAll)
    for i in range(len(Ebins)):
        sel = EnergyAll == Ebins[i]
        arg = np.argsort(ZenithAll[sel])
        plt.plot(ZenithAll[sel][arg], XmaxDistAll[sel][arg]/1e3, '-o', label=f"E={Ebins[i]} EeV")
    plt.xlabel("$\\theta$ [degrees]")
    plt.ylabel("Xmax distance [km]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.yscale('log')
    plt.show()

def PlotDeepTriggersVsZenith(EnergyAll, ZenithAll, DeepTriggerAll):
    """
    Plot the deep trigger fraction as a function of the zenith angle for different energies.
    """
    #plt.figure(figsize=(10, 6))
    Ebins = np.unique(EnergyAll)
    for i in range(len(Ebins)):
        sel = EnergyAll == Ebins[i]
        arg = np.argsort(ZenithAll[sel])
        plt.plot(ZenithAll[sel][arg], DeepTriggerAll[sel][arg]*100, 'o', label=f"E={Ebins[i]} EeV")
    plt.xlabel("$\\theta$ [degrees]")
    plt.ylabel("Deep trigger only events [$\%$]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.yscale('log')
    plt.show()

def PlotMeanTimedelayEbin(ZenithAll, EnergyAll, meandeltat, stddeltat):
    
    Ebins = np.unique(EnergyAll)
    for i in range(len(Ebins)):
        sel = EnergyAll == Ebins[i]
        arg = np.argsort(ZenithAll[sel])
        plt.errorbar(ZenithAll[sel][arg], meandeltat[sel][arg]*1e9, yerr=stddeltat[sel][arg]*1e9, fmt='o', label=f"E={Ebins[i]} EeV")
    plt.ylabel("Time delay [ns]")
    plt.xlabel("zenith [Deg.]")
    plt.legend()
    plt.show()

def PlotTimeDistributionAllsimsperEbin(ZenithAll, EnergyAll, dt_all_sims, selE):
    
    zenithmask = np.array([10, 28, 39, 50])
    bins = np.linspace(350, 750, 30 + 1)

    argsort = np.argsort(ZenithAll)

    for i in range(len(ZenithAll[argsort])):
        if(ZenithAll[argsort][i] not in zenithmask): continue
        if(EnergyAll[argsort][i] != selE): continue
        print(i)
        dt_all_sims_sorted = [dt_all_sims[i] for i in argsort]

        plt.hist(dt_all_sims_sorted[i]*1e9, bins = bins, alpha=0.7, edgecolor='black', label=f"$\\theta$={ZenithAll[argsort][i]}$^\\circ$")
    plt.xlabel("Time delay [ns]")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
