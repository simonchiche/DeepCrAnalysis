import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

def PlotSurfaceFootprint(footprint, Xmax, zenith, azimuth, Save, BatchID):
    plt.figure(figsize=(6, 6))
    plt.plot(footprint[:, 0], footprint[:, 1], label='Surface footprint', color='blue')
    #plt.scatter(Xmax[0], Xmax[1], color='red', label='Xmax projection')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'(zenith={zenith}°, azimuth={azimuth}°)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{BatchID}_GroundFootprint.pdf", bbox_inches = "tight") if Save else None
    plt.show()


def CompareFootprints(footprint, zenith, azimuth, all_xray, all_yray, Save, BatchID):

    # Display the footprint
    plt.figure(figsize=(6, 6))
    plt.plot(footprint[:, 0], footprint[:, 1], label='Surface footprint', color='blue')
    plt.plot(all_xray, all_yray, color='red', label='In-ice footprint')
    #plt.scatter(0, 0, color='red', label='Projection verticale de Xmax')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'(zenith={zenith}°, azimuth={azimuth}°)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{BatchID}_GroundvsDeepFootprint.pdf", bbox_inches = "tight") if Save else None
    plt.show()

def PlotFootprintPolygons(polygon_surface, polygon_deep):

    difference = polygon_deep.difference(polygon_surface)
    # Area of each zone
    area_depth = polygon_deep.area
    area_outside = difference.area
    fraction_outside = area_outside / area_depth if area_depth > 0 else 0


    # Plot both polygons
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot surface footprint in blue
    plot_polygon(polygon_surface, ax=ax, add_points=False, facecolor='lightblue', edgecolor='blue', alpha=0.4, label='Surface')
    # Plot depth footprint in red
    plot_polygon(polygon_deep, ax=ax, add_points=False, facecolor='lightcoral', edgecolor='red', alpha=0.4, label='Depth')

    ax.set_title("Surface vs 100m-deep footprint")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    plt.grid(True)
    plt.legend(["Surface", "100m-deep"])
    plt.text(
        0.05, 0.95,  # x and y in axes coordinates (0 to 1)
        f"Deep triggers only: {fraction_outside:.2%}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    plt.show()


def PlotAmplitudeDilution(all_xray_samples, all_yray_samples, SurfaceDeepRatio):
 
    # Amplitude dilution scatter plot
    plt.scatter(all_xray_samples, all_yray_samples, c=SurfaceDeepRatio*100, cmap='jet', s=1, label='Ray paths')
    cbar = plt.colorbar()
    cbar.set_label('$[\%]$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Amplitude fraction at the deep antennas')
    plt.show()

def PlotTimeDelayDistribution(all_dt_samples,Nbins=10):
    """
    Plot the histogram of time delay distribution.
    """
    #plt.figure(figsize=(8, 6))
    plt.hist(np.array(all_dt_samples)*1e9, bins=Nbins, color='sandybrown', alpha=0.7, edgecolor='black')
    plt.xlabel('Time Delay [ns]')
    plt.ylabel('Count')
    plt.title('Time Delay Distribution')
    plt.grid(True)
    #plt.savefig(f"{OutputPath}_TimeDelayDistribution.pdf", bbox_inches="tight") if Save else None
    plt.show()

def PlotSampledFootprint(footprint_samples, Shower):
    """
    Plot the sampled surface footprint.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(footprint_samples[:, 0], footprint_samples[:, 1], color='blue', s=1, label='Sampled positions')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Sampled surface footprint ($\\theta$={Shower.zenith}°, $\\varphi$={Shower.azimuth}°)', fontsize =12)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    #plt.savefig(f"{OutputPath}_SampledFootprint.pdf", bbox_inches="tight") if Save else None
    plt.show()