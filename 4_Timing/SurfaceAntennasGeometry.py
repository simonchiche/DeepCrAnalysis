import numpy as np
import matplotlib.pyplot as plt

def sph_to_cart(theta, phi):
    """Transforme des coordonnées sphériques en cartésiennes (unit vector)."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

def generate_footprint(Xmax, zenith_deg, azimuth_deg, theta_C_deg=1.0, n_rays=360):
    # Convert angles to radians
    zenith = np.radians(zenith_deg)
    azimuth = np.radians(azimuth_deg)
    theta_C = np.radians(theta_C_deg)

    # Direction de la gerbe (vecteur unitaire)
    shower_axis = sph_to_cart(zenith, azimuth)

    # Base locale orthonormée autour de l'axe de la gerbe
    z_axis = shower_axis
    # Trouver un vecteur perpendiculaire
    if np.allclose(z_axis, [0, 0, 1]):
        tmp = np.array([1, 0, 0])
    else:
        tmp = np.array([0, 0, 1])
    x_axis = np.cross(z_axis, tmp)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Échantillonnage d'angles autour du cône
    alphas = np.linspace(0, 2 * np.pi, n_rays)
    points = []

    for alpha in alphas:
        # Direction du rayon sur le cône de Cherenkov
        dir_vector = (
            np.cos(theta_C) * z_axis +
            np.sin(theta_C) * (np.cos(alpha) * x_axis + np.sin(alpha) * y_axis)
        )
        print(dir_vector)

        # Intersecter le rayon avec le plan z = 0 (sol)
        dz = -dir_vector[2]
        if dz == 0:
            continue  # rayon parallèle au sol, ne l’intersecte jamais
        t = -Xmax[2] / dz
        if t <= 0:
            print("test")
            continue  # rayon vers le haut

        point_on_ground = Xmax + t * dir_vector
        points.append(point_on_ground[:2])  # x, y

    return np.array(points)

# Paramètres de test
Xmax = np.array([-1250, 0.0, 700.0])  # Position du Xmax à 700 m d'altitude
zenith = 60.0  # en degrés
azimuth = 0.0  # en degrés (vers l'est)
theta_C = 1.5 # angle Cherenkov en degrés

# Générer la footprint
footprint = generate_footprint(Xmax, zenith, azimuth, theta_C*3)

# Affichage
plt.figure(figsize=(6, 6))
plt.plot(footprint[:, 0], footprint[:, 1], label='Footprint au sol')
#plt.scatter(0, 0, color='red', label='Projection verticale de Xmax')
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Footprint au sol (zenith={zenith}°, azimuth={azimuth}°)')
plt.legend()
plt.grid(True)
plt.show()
