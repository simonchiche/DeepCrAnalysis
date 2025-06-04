import numpy as np
import matplotlib.pyplot as plt

def generate_equal_arc_polar_grid(radii, arc_step):
    x_all, y_all = [], []

    for r in radii:
        n_phi = int(np.round(2 * np.pi * r / arc_step))#max(1, int(np.round(2 * np.pi * r / arc_step)))
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        x_all.extend(r * np.cos(phis))
        y_all.extend(r * np.sin(phis))

    return np.array(x_all), np.array(y_all)



def GenerateDenseCoreArray(r99):

    # positions for the inner parts
    rinmax = 250
    rcoremax = 150
    corestep = 10 
    instep = 15
    arcstep_1 = corestep

    radii_1 =  np.arange(10, rcoremax, corestep)
    arcstep_2 = instep
    radii_2 =  np.arange(rcoremax, rinmax, instep)

    xant_1, yant_1 = generate_equal_arc_polar_grid(radii_1, arcstep_1)
    xant_2, yant_2 = generate_equal_arc_polar_grid(radii_2, arcstep_2)

    # positions for the inner parts
    # radius for the outer parts
    rlogmin_out = np.log10(rinmax)
    rlogmax_out = np.log10(r99 + 100)
    rlog_out =  np.linspace(rlogmin_out, rlogmax_out, 15)
    rout = 10**rlog_out[1:]

    # angles for the outter parts
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    
    # positions for the outter parts
    Nangles = len(angles)
    Nradius = len(rout)

    Nant = Nradius*Nangles
    xant_3, yant_3 = np.zeros(Nant), np.zeros(Nant)

    for i in range(len(rout)):
        for j in range(len(angles)):

            xant_3[Nangles*i + j] = rout[i]*np.cos(angles[j])
            yant_3[Nangles*i + j] = rout[i]*np.sin(angles[j])

    xant = np.concatenate([xant_1, xant_2, xant_3])
    yant = np.concatenate([yant_1, yant_2, yant_3])

    return xant, yant

xant, yant = GenerateDenseCoreArray(1000)
plt.scatter(xant, yant, s = 5)
plt.show()