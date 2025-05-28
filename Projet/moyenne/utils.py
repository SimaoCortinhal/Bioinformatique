import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_raw_membrane(coords, title="Raw Membrane Structure"):
    """
    Trace les coordonnées lipidiques brutes de la membrane en 3D.

    Paramètres :
        coords (np.array) : Tableau de coordonnées lipidiques (x, y, z).
        title (str) : Titre du graphique.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Diagramme de dispersion de toutes les coordonnées des lipides
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, color='blue', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def plot_membrane(coords, leaflets):
    """
    Représente la structure de la membrane en 3D avec des couleurs différentes pour chaque feuillet.

    Paramètres :
        coords (np.array) : Coordonnées des lipides (x, y, z).
        leaflets (liste) : Identifiants des lipides regroupés par feuillet.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red']  # Couleurs pour les feuillets
    
    for idx, leaflet in enumerate(leaflets):
        leaflet_coords = coords[list(leaflet)]
        ax.scatter(leaflet_coords[:, 0], leaflet_coords[:, 1], leaflet_coords[:, 2], 
                   label=f"Leaflet {idx+1}", s=10, color=colors[idx % len(colors)])
    
    ax.set_title("Membrane Structure")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def plot_z_coordinate_distribution(coords, num_bins=50):
    """
    Trace la distribution des coordonnées z des lipides.

    Paramètres :
        coords (np.array) : Coordonnées des lipides (x, y, z).
        num_bins (int) : Nombre de bins pour l'histogramme.
    """
    z_coords = coords[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.hist(z_coords, bins=num_bins, color='green', alpha=0.7, edgecolor='black')
    plt.title("Distribution of z-Coordinates")
    plt.xlabel("z-Coordinate")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_leaflet_distances(distances_flat, leaflet):
    """
    Trace la distribution des distances par paire au sein d'un feuillet.

    Paramètres :
        distances_flat (np.array) : Distances par paire à l'intérieur du feuillet.
        leaflet (str) : Nom du feuillet ('top' ou 'bottom').
    """
    plt.figure(figsize=(10, 6))
    plt.hist(distances_flat, bins=50, color='lightblue', edgecolor='black')
    plt.title(f'Pairwise Distance Distribution in {leaflet.capitalize()} Leaflet')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def plot_all_pairwise_distances(distances, title="Distribution of All Pairwise Distances"):
    """
    Trace la distribution de toutes les distances par paire dans le système.

    Args :
        distances (np.array) : Matrice des distances par paire (tableau 2D).
        title (str) : Titre du graphique.
    """
    # Extrait la partie triangulaire supérieure de la matrice des distances (pour éviter les distances en double)
    distances_flat = distances[np.triu_indices(len(distances), k=1)]

    # Affiche l'histogramme
    plt.figure(figsize=(10, 6))
    plt.hist(distances_flat, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return distances_flat
