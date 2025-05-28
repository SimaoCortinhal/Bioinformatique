import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_raw_membrane(coords, title="Raw Membrane Structure"):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Afficher les coordonnées des lipides
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, color='blue', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def plot_membrane(coords, leaflets, filename=""):
    #Récupérer que le nom du fichier
    file = filename.split("/")[-1]   
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red']
    
    for idx, leaflet in enumerate(leaflets):
        leaflet_coords = coords[list(leaflet)]
        ax.scatter(leaflet_coords[:, 0], leaflet_coords[:, 1], leaflet_coords[:, 2], 
                   label=f"Leaflet {idx+1}", s=10, color=colors[idx % len(colors)])
    
    title = "Membrane Structure: " + file
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def plot_z_coordinate_distribution(coords, num_bins=50):
    z_coords = coords[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.hist(z_coords, bins=num_bins, color='green', alpha=0.7, edgecolor='black')
    plt.title("Distribution of z-Coordinates")
    plt.xlabel("z-Coordinate")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_leaflet_distances(distances_flat, leaflet):
    plt.figure(figsize=(10, 6))
    plt.hist(distances_flat, bins=50, color='lightblue', edgecolor='black')
    plt.title(f'Pairwise Distance Distribution in {leaflet.capitalize()} Leaflet')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def plot_all_pairwise_distances(distances, title="Distribution of All Pairwise Distances"):
    distances_flat = distances[np.triu_indices(len(distances), k=1)]

    plt.figure(figsize=(10, 6))
    plt.hist(distances_flat, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return distances_flat
