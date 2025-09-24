import numpy as np

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

import numpy as np

def GetMaxLDF(Pos, Etot, Depth,channel):

    if(channel == "x"): k= 1
    if(channel == "y"): k= 0
    # Returns the antenna postiions and signl amplitude along the line
    # Extract unique x positions (horizontal positions)
    sel = (Pos[:,2] == Depth)
    Pos = Pos[sel]
    Etot = Etot[sel]
    unique_x = np.unique(Pos[:, k])
    
    # Dictionary to store summed amplitudes per vertical line
    amplitude_sum = {}
    
    # Iterate over unique x positions
    for x in unique_x:
        mask = Pos[:, k] == x  # Select antennas in the vertical line at x
        amplitude_sum[x] = np.sum(Etot[mask])
    
    # Find x with the maximum summed amplitude
    best_x = max(amplitude_sum, key=amplitude_sum.get)
    
    # Extract positions and amplitudes for the best vertical line
    mask = Pos[:, k] == best_x
    best_positions = Pos[mask]
    best_amplitudes = Etot[mask]
    #print(len(mask), "mask")
    
    return best_positions, best_amplitudes

def GetMaxLDFGeneric(Pos, Etot, Depth,channel):

    if(channel == "x"): k= 0
    if(channel == "y"): k= 1
    # Returns the antenna postiions and signl amplitude along the line
    # Extract unique x positions (horizontal positions)
    sel = (Pos[:,2] == Depth)
    Pos = Pos[sel]
    Etot = Etot[sel]

    min_ax = np.min(Pos[:, k])
    max_ax = np.max(Pos[:, k])
    delta_ax = 5
    ax_bins = np.arange(min_ax, max_ax + delta_ax, delta_ax)


    best_arg = np.argmax([np.sum(Etot[(Pos[:, k] >= ax_bins[i]) & (Pos[:, k] < ax_bins[i+1])]) for i in range(len(ax_bins)-1)])
    mask = (Pos[:, k] >= ax_bins[best_arg]) & (Pos[:, k] < ax_bins[best_arg+1])
    best_positions = Pos[mask]
    best_amplitudes = Etot[mask]        

    
    return best_positions, best_amplitudes

def GetMaxLDFx(Pos, Etot, Depth,channel):


    # Returns the antenna postiions and signl amplitude along the line
    # Extract unique x positions (horizontal positions)
    sel = (Pos[:,2] == Depth)
    Pos = Pos[sel]
    Etot = Etot[sel]

    min_y = np.min(abs(Pos[:, 1]))
    delta_ax = 5
    print(min_y)
    mask = (Pos[:, 1] >= (min_y - delta_ax) ) & (Pos[:, 1] < (min_y + delta_ax)) 
    best_positions = Pos[mask]
    best_amplitudes = Etot[mask]        
    
    return best_positions, best_amplitudes

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
        
    print(radioextent/simextent)

    return radioextent, simextent, extent, maxpos, xminlay, xmaxlay



def GetCaracExtent(x, I, frac=0.99):
    """
    Returns the symmetric x-limits around x=0 containing the given fraction of total intensity.
    
    Parameters:
    - x: 1D array of positions (can be positive or negative, centered on 0)
    - I: corresponding intensity values (same length as x)
    - frac: fraction of total intensity to include (default is 0.95)

    Returns:
    - extent: scalar, symmetric bound such that [-extent, extent] contains `frac` of total intensity
    """
 
    # Order by proximity to 0
    indexes = np.argsort(np.abs(x))
    
    # Cumulative intensity fraction
    I_sorted = I[indexes]
    cum_I = np.cumsum(I_sorted)
    print(cum_I)
    
    cum_I_frac = cum_I / cum_I[-1]
    print(cum_I_frac)
    print(x)
    
    # Find how many points needed to reach desired fraction
    idx = np.searchsorted(cum_I_frac, frac)
    
    # Corresponding x values
    x_included = np.abs(x[indexes[:idx+1]])
    extent = np.max(x_included)

    xmax = abs(x[np.argmax(I)])

    return extent, xmax


def central_interval(x, I):
    # Sort x and I together by x (in case they aren't already)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    I_sorted = I[sorted_indices]

    # Compute the differential dx between points
    dx = np.gradient(x_sorted)
    
    # Compute cumulative intensity
    intensity = I_sorted * dx
    total_intensity = np.sum(intensity)
    
    cum_intensity = np.cumsum(intensity)
    
    # Normalize cumulative intensity to go from 0 to 1
    cum_intensity /= total_intensity

    # Find indices where cumulative intensity goes from 2.5% to 97.5%
    lower_idx = np.searchsorted(cum_intensity, 0.025)
    upper_idx = np.searchsorted(cum_intensity, 0.975)

    return x_sorted[upper_idx] #x_sorted[lower_idx], x_sorted[upper_idx]
