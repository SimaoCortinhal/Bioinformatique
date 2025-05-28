import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_raw_membrane(coords, title="Raw Membrane Structure"):
    """
    Plots the membrane's raw lipid coordinates in 3D.

    Args:
        coords (np.array): Array of lipid coordinates (x, y, z).
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of all lipid coordinates
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, color='blue', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def plot_membrane(coords, leaflets):
    """
    Plots the membrane structure in 3D with different colors for each leaflet.

    Args:
        coords (np.array): Lipid coordinates (x, y, z).
        leaflets (list): Lipid IDs grouped by leaflet.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red']
    
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
    Plots the distribution of the z-coordinates of lipids.

    Args:
        coords (np.array): Lipid coordinates (x, y, z).
        num_bins (int): Number of bins for the histogram.
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
    Plot the distribution of pairwise distances within a leaflet.

    Args:
        distances_flat (np.array): Pairwise distances within the leaflet.
        leaflet (str): Name of the leaflet ('top' or 'bottom').
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
    Plot the distribution of all pairwise distances in the system.

    Args:
        distances (np.array): Pairwise distance matrix (2D array).
        title (str): Title of the plot.
    """
    distances_flat = distances[np.triu_indices(len(distances), k=1)]

    plt.figure(figsize=(10, 6))
    plt.hist(distances_flat, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return distances_flat

import matplotlib.pyplot as plt
import numpy as np

def plot_lipid_orientations(orientation_coords):
    """
    Plots the orientation of lipids based on their headgroup and tailgroup atom coordinates.

    Parameters:
    orientation_coords (list of tuples): A list where each element is a tuple of (headgroup_coord, tailgroup_coord),
                                          where each of these is a 3D coordinate (x, y, z).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    headgroup_x = []
    headgroup_y = []
    headgroup_z = []
    tailgroup_x = []
    tailgroup_y = []
    tailgroup_z = []

    for headgroup_coord, tailgroup_coord in orientation_coords:
        if headgroup_coord is not None and tailgroup_coord is not None:
            headgroup_x.append(headgroup_coord[0])
            headgroup_y.append(headgroup_coord[1])
            headgroup_z.append(headgroup_coord[2])
            
            tailgroup_x.append(tailgroup_coord[0])
            tailgroup_y.append(tailgroup_coord[1])
            tailgroup_z.append(tailgroup_coord[2])

    ax.scatter(headgroup_x, headgroup_y, headgroup_z, color='r', label='Headgroup')
    ax.scatter(tailgroup_x, tailgroup_y, tailgroup_z, color='b', label='Tailgroup')

    for i in range(len(headgroup_x)):
        ax.plot([headgroup_x[i], tailgroup_x[i]],
                [headgroup_y[i], tailgroup_y[i]],
                [headgroup_z[i], tailgroup_z[i]], color='k', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lipid Orientations')

    ax.legend()

    plt.show()

def visualize_lipid_orientations(ref_coords, orientation_vectors, scale=5, title="Lipid Orientations"):
    """
    Visualize the orientation of each lipid in a 3D space using arrows.
    
    Parameters:
        ref_coords (np.array): Array of reference coordinates for each lipid (Nx3 array).
        orientation_vectors (np.array): Array of orientation vectors for each lipid (Nx3 array).
        scale (float): Scale factor for the arrow length (default: 5).
        title (str): Title of the plot (default: "Lipid Orientations").
    """
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=16)

    x, y, z = ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2]
    u, v, w = orientation_vectors[:, 0], orientation_vectors[:, 1], orientation_vectors[:, 2]

    norms = np.linalg.norm(orientation_vectors, axis=1, keepdims=True)
    normalized_vectors = orientation_vectors / norms
    u, v, w = normalized_vectors[:, 0], normalized_vectors[:, 1], normalized_vectors[:, 2]

    ax.scatter(x, y, z, c='blue', s=10, label='Lipid Positions')
    
    ax.quiver(x, y, z, u, v, w, length=scale, color='red', label='Lipid Orientation Vectors', arrow_length_ratio=0.3)

    ax.set_xlabel('X (nm)', fontsize=12)
    ax.set_ylabel('Y (nm)', fontsize=12)
    ax.set_zlabel('Z (nm)', fontsize=12)

    ax.legend(fontsize=12)
    ax.grid(True)

    plt.show()
    
def plot_cosine_similarity_histogram(similarity_matrix):
    """
    Calculate and plot a histogram of cosine similarities between orientation vectors.
    """
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_matrix.flatten(), bins=20, color='b', edgecolor='k', alpha=0.7)
    plt.title("Histogram of Cosine Similarities between Lipid Orientation Vectors")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    
def plot_normal_vectors(ref_coords, normal_vectors, vector_scale=1.0):
    """
    Plot lipid positions and their normal vectors in 3D space.

    Parameters:
        ref_coords (np.array): Array of shape (n, 3) with lipid coordinates.
        normal_vectors (np.array): Array of shape (n, 3) with normal vectors.
        vector_scale (float): Scale factor for normal vectors (default is 1.0).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2], color='blue', label='Lipid Positions', alpha=0.6)

    for i in range(len(ref_coords)):
        start = ref_coords[i]
        vector = normal_vectors[i] * vector_scale
        ax.quiver(
            start[0], start[1], start[2],
            vector[0], vector[1], vector[2],
            color='red', length=vector_scale, normalize=True, alpha=0.8
        )
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lipid Positions and Normal Vectors')

    ax.legend()

    plt.show()