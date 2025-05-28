import numpy as np
import networkx as nx
import MDAnalysis as mda
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import cuda, float32
import math
import sys

from utils import *

def read_gromacs_file_with_mdanalysis(filename):
    """
    Lecture d'un fichier GROMACS (.gro) avec MDAnalysis pour sélectionner les lipides
    et récupérer les coordonnées du premier atome de chaque lipide, 
    leurs identifiants et les dimensions de la boîte.
    """
    u = mda.Universe(filename)
    # Définir les résidus non-lipidiques à exclure
    not_lipid = ["W", "NA+", "CL-", "SOL", "HOH","CL","ACE","LYS","LEU","GLY","ALA","NH2","MET","GLN","PRO","ILE","VAL","SER","TRP","TYR","GLU","ARG"]
    
    lipid_selection = u.select_atoms(
        " and ".join([f"not resname {res}" for res in not_lipid])
    )
    
    # Vérifier si la sélection est vide
    if len(lipid_selection) == 0:
        raise ValueError("Aucun lipide trouvé dans la sélection. Vérifiez le fichier ou la sélection.")
    
    lipid_residues = lipid_selection.residues
    
    # Extraire les coordonnées du premier atome de chaque lipide
    lipid_coords = [res.atoms[0].position for res in lipid_residues]
    lipid_ids = lipid_residues.resids
    
    box_dimensions = u.dimensions
    
    print(f"Nombre de lipides : {len(lipid_coords)}")
    print(f"Coordonnées des premiers atomes des lipides : {len(lipid_coords)}")  # Afficher les 5 premières pour vérification
    
    lipid_coords = np.array(lipid_coords)
    
    return lipid_coords, lipid_ids, box_dimensions

def compute_periodic_distances_with_mdanalysis(coords, box_dimensions):
    from MDAnalysis.lib.distances import apply_PBC

    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            distance = np.linalg.norm(diff)
            distances[i, j] = distances[j, i] = distance

    return distances

@cuda.jit
def compute_distances_kernel(coords, box_dimensions, distances, n):
    """
    Kernel CUDA pour calculer la matrice de distances.
    """
    i, j = cuda.grid(2)
    if i < n and j < n and i < j:  # Calcule uniquement la moitié supérieure de la matrice
        diff = cuda.local.array(3, dtype=float32)
        diff_in_box = cuda.local.array(3, dtype=float32)

        # Calcule les différences de coordonnées avec PBC
        for k in range(3):
            diff[k] = coords[i, k] - coords[j, k]
            diff_in_box[k] = diff[k] - round(diff[k] / box_dimensions[k]) * box_dimensions[k]

        distance = 0.0
        for k in range(3):
            distance += diff_in_box[k] ** 2
        distance = math.sqrt(distance)

        # Mise à jour de la matrice de distances
        distances[i, j] = distance
        distances[j, i] = distance

def compute_periodic_distances_with_mdanalysis_gpu(coords, box_dimensions):
    """
    Calcul de la matrice de distances en utilisant le GPU avec CUDA (numba).
    """
    n = coords.shape[0]

    coords = coords.astype(np.float32)
    box_dimensions = box_dimensions.astype(np.float32)

    # Allocation de la mémoire sur le GPU
    d_coords = cuda.to_device(coords)
    d_box_dimensions = cuda.to_device(box_dimensions)
    d_distances = cuda.device_array((n, n), dtype=np.float32)

    # Définition de la configuration du kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    compute_distances_kernel[blocks_per_grid, threads_per_block](d_coords, d_box_dimensions, d_distances, n)

    # Récupérer les résultats du GPU
    distances = d_distances.copy_to_host()
    return distances

def find_leaflets_by_graph(coords, ids, distances, distance_threshold):
    """
    Trouver les deux feuillets constitué de lipides
    """
    # Initialisation du graphe
    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, lipid_id=ids[i])

    # Construction des arêtes initiales
    mask = distances < distance_threshold
    np.fill_diagonal(mask, False)
    edges = np.column_stack(np.where(mask))
    G.add_edges_from(edges)

    print("Initial Number of edges:", len(G.edges))
    print("Initial Number of nodes:", len(G.nodes))

    # Itération pour ajuster le seuil de distance
    new_distance_threshold = distance_threshold
    size_limit = len(G.nodes) // 4 # Seuil pour les petites composantes
    print("Size limit for small components:", size_limit)
    
    while len(list(nx.connected_components(G))) > 2:
        components = list(nx.connected_components(G))
        component_sizes = [len(c) for c in components]
        
        print("Number of components:", len(components))
        print("Component sizes:", component_sizes)

        # Recalculer les petites et grandes composantes
        small_components = {node for comp in components if len(comp) <= size_limit for node in comp}
        large_components = {node for comp in components if len(comp) > size_limit for node in comp}
        
        # Mise à jour des connexions
        new_distance_threshold += 1
        mask = (distances < new_distance_threshold) & ~mask  # Arêtes supplémentaires
        new_edges = np.column_stack(np.where(mask))

        # Filtrer les nouvelles connexions
        filtered_edges = []
        for i, j in new_edges:
            # Si les deux nœuds sont dans des grandes composantes, on interdit la fusion
            if i in large_components and j in large_components:
                continue
            # Sinon, autoriser la fusion entre petites et grandes composantes
            elif (i in small_components and j in large_components) or (j in small_components and i in large_components) or (i in small_components and j in small_components):
                filtered_edges.append((i, j))
                G.add_edges_from(filtered_edges)
                # Mettre à jour les petites et grandes composantes après chaque ajout d'arête
            components = list(nx.connected_components(G))
            small_components = {node for comp in components if len(comp) <= size_limit for node in comp}
            large_components = {node for comp in components if len(comp) > size_limit for node in comp}

        print(f"Updated distance threshold: {new_distance_threshold}")
        print("Updated Number of edges:", len(G.edges))

    components = list(nx.connected_components(G))
    leaflets = components
    print("Final Number of components:", len(components))
    print("Final Component sizes:", [len(c) for c in components])

    return leaflets, G

def analyse_membrane(filename):
    # Lecture du fichier .gro
    coords, ids, box_dimensions = read_gromacs_file_with_mdanalysis(filename)
    # Affichage de la structure 
    plot_raw_membrane(coords, title="Raw Membrane Structure Before Leaflet Computation")
        
    print("Analyzing z-coordinate distribution...")
    plot_z_coordinate_distribution(coords)
    
    distance_threshold = 10
    
    print("Current distance threshold", distance_threshold)
    
    # Vérifier que l'on dispose d'un GPU:
    if not cuda.is_available():
        print("No GPU found.")
        distances = compute_periodic_distances_with_mdanalysis(coords, box_dimensions)
    else:
        print("GPU found.")
        distances = compute_periodic_distances_with_mdanalysis_gpu(coords, box_dimensions)
    
    distances_flat = plot_all_pairwise_distances(distances, title="Distribution of Pairwise Distances in the System")

    print(f"Max Distance: {np.max(distances_flat):.2f}")
    print(f"Min Distance: {np.min(distances_flat):.2f}")
    print(f"Mean Distance: {np.mean(distances_flat):.2f}")
    print(f"Median Distance: {np.median(distances_flat):.2f}")
    
    leaflets, graph = find_leaflets_by_graph(coords, ids, distances, distance_threshold)
    
    # Visualisation des feuillets trouvés
    plot_membrane(coords, leaflets, filename=filename)
    
    return leaflets, ids

def save_results(leaflets, ids, output_file):
    with open(output_file, 'w') as f:
        for i, leaflet in enumerate(leaflets, 1):
            # Convertir l'ensemble en liste pour permettre l'indexation
            leaflet = list(leaflet)
            
            # Mappez les indices des noeuds aux identifiants des lipides
            print(f"Leaflet {i}: {len(leaflet)}")
            lipid_ids = [ids[n] for n in leaflet]
            lipid_ids.sort()
            f.write(f"[ membrane_{1}_leaflet_{i} ]\n")
            # Écrire les lipides, 15 par ligne
            for j in range(0, len(lipid_ids), 15):
                row = lipid_ids[j:j+15]
                formatted_row = " ".join(f"{lipid:5d}" for lipid in row)
                f.write(f"   {formatted_row}\n")

def get_filename(arg):
    switcher = {
        "1": "bilayer_chol", 
        "2": "bilayer_peptide", 
        "3": "bilayer_prot", 
        "4": "dppc_vesicle", 
        "5": "large_plasma_membrane", 
        "6": "small_plasma_membrane",  
        "7": "model_vesicle"
    }
    return switcher.get(arg, "bilayer_chol")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python membrane-analyzer.py <filename>")
        sys.exit(1)
        
    filename = get_filename(sys.argv[1]) 
    print(f"Selected filename: {filename}")
    input_file = f"./data/{filename}/{filename}.gro"
    output_file = f"./output/leaflets_results{filename}.txt"
    
    
    leaflets, ids = analyse_membrane(input_file)
    save_results(leaflets, ids, output_file)
    
    print(f"Analysis complete. Results saved in {output_file}")
