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

    :param filename: Chemin vers le fichier GROMACS (.gro)
    :return: tuple contenant les coordonnées des premiers atomes des lipides, leurs identifiants, et les dimensions de la boîte
    """
    # Charge l'univers
    u = mda.Universe(filename)

    # Définis les résidus non-lipidiques à exclure
    not_lipid = ["W", "NA+", "CL-", "SOL", "HOH","CL","ACE","LYS","LEU","GLY","ALA","NH2","MET","GLN","PRO","ILE","VAL","SER","TRP","TYR","GLU","ARG"]

    # Construis une sélection MDAnalysis
    lipid_selection = u.select_atoms(
        " and ".join([f"not resname {res}" for res in not_lipid])
    )

    # Vérifie si la sélection est vide
    if len(lipid_selection) == 0:
        raise ValueError("Aucun lipide trouvé dans la sélection. Vérifiez le fichier ou la sélection.")

    # Récupére les résidus des lipides
    lipid_residues = lipid_selection.residues

    # Identifiants des résidus des lipides
    lipid_ids = lipid_residues.resids  

    #Calcul du centre de masse de chaque lipide
    lipid_coords = []
    for lipid in lipid_residues:
        
        #Récupération des masses et des positions
        atom_positions = np.array([atom.position for atom in lipid.atoms])
        atom_masses = np.array([atom.mass for atom in lipid.atoms])
        
        #Adiition des masses des atoms
        mass_sum = np.sum(atom_masses)
        #Calcul du centre de masses
        weighted_positions = np.sum(atom_positions * atom_masses[:, np.newaxis], axis=0) / mass_sum
        
        lipid_coords.append(weighted_positions)
    

    # On récupère les dimensions de la boîte
    box_dimensions = u.dimensions

    # Debugging ou information utile
    print(f"Nombre de lipides : {len(lipid_coords)}")
    print(f"Coordonnées des premiers atomes des lipides : {len(lipid_coords)}")  # Afficher les 5 premières pour vérification

    #On retransforme la liste en un tableau numpy
    lipid_coords = np.array(lipid_coords)

    # Retourner les résultats
    return lipid_coords, lipid_ids, box_dimensions


#fonction pour calculer la moyenne qu in'est pas utilisée
def read_gromacs_file_with_mdanalysis_and_mean(filename):

    u = mda.Universe(filename)
    # selection des lipides
    lipids = u.select_atoms("resname POPC or resname CHOL or resname DPPC")
    lipids_mean= []
    # calcul pour chaque lipide de la moyenne
    for lipid in lipids.residues:
        lipids_mean.append(lipid.atoms.positions.mean(axis=0))

    lipids_mean_numpy=np.array(lipids_mean)
    box_dimensions = u.dimensions

    return lipids_mean_numpy,lipids.residues.resids,box_dimensions


def compute_periodic_distances_with_mdanalysis(coords, box_dimensions):
    from MDAnalysis.lib.distances import apply_PBC

    n = len(coords)
    distances = np.zeros((n, n))

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            # On applique des conditions limites périodiques pour calculer la distance minimale de l'image
            diff = coords[i] - coords[j]
            diff_in_box = apply_PBC(diff, box_dimensions)
            distance = np.linalg.norm(diff)
            distances[i, j] = distances[j, i] = distance

    return distances

@cuda.jit
def compute_distances_kernel(coords, box_dimensions, distances, n):
    """
   On définit un noyau CUDA pour calculer les distances par paire avec des conditions limites périodiques.
    """
    i, j = cuda.grid(2)
    if i < n and j < n and i < j:  # Only calculate upper triangle
        diff = cuda.local.array(3, dtype=float32)
        diff_in_box = cuda.local.array(3, dtype=float32)

        # calcul de la différence élément par élément
        for k in range(3):
            diff[k] = coords[i, k] - coords[j, k]
            diff_in_box[k] = diff[k] - round(diff[k] / box_dimensions[k]) * box_dimensions[k]

        # Calcul de la distance
        distance = 0.0
        for k in range(3):
            distance += diff_in_box[k] ** 2
        distance = math.sqrt(distance)

        # Mise à jour de la matrice de distance de maière symétrique
        distances[i, j] = distance
        distances[j, i] = distance

def compute_periodic_distances_with_mdanalysis_gpu(coords, box_dimensions):
    """
    Calcul des distances périodiques par paire en utilisant Numba CUDA.

    Paramètres :
        coords (numpy.ndarray) : Tableau de forme (n, 3) avec les coordonnées atomiques.
        box_dimensions (numpy.ndarray) : Tableau de formes (3,) avec les dimensions de la boîte.

    Retourne :
        numpy.ndarray : Matrice de distance symétrique de la forme (n, n).
    """
    n = coords.shape[0]

    # Converti les entrées en float32 pour la compatibilité avec CUDA
    coords = coords.astype(np.float32)
    box_dimensions = box_dimensions.astype(np.float32)

    # Alloue la mémoire sur le GPU
    d_coords = cuda.to_device(coords)
    d_box_dimensions = cuda.to_device(box_dimensions)
    d_distances = cuda.device_array((n, n), dtype=np.float32)

    # Définis la taille des blocks et de la grille
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Lancement du kernel
    compute_distances_kernel[blocks_per_grid, threads_per_block](d_coords, d_box_dimensions, d_distances, n)

    # Copie les résultats sur l'hôte
    distances = d_distances.copy_to_host()
    return distances




def compute_leaflet_pairwise_distances(coords, ids, z_threshold, box_dimensions, leaflet='top'):
    """
    Calcule les distances par paire à l'intérieur d'un même feuillet et proposer un seuil de regroupement.
    """
    # Sépare les feuillets en fonction de la coordonnée z
    if leaflet == 'top':
        leaflet_coords = coords[coords[:, 2] >= z_threshold]
    elif leaflet == 'bottom':
        leaflet_coords = coords[coords[:, 2] < z_threshold]
    else:
        raise ValueError("Leaflet must be 'top' or 'bottom'.")

    #  Calcule les distances par paire pour le feuillet
    distances = compute_periodic_distances_with_mdanalysis_gpu(leaflet_coords, box_dimensions)

    # Aplatis la matrice des distances pour extraire les distances par paire
    distances_flat = distances[np.triu_indices(len(leaflet_coords), k=1)]

    # Suggére un seuil basé sur les distances observées
    max_distance = np.max(distances_flat)
    suggested_threshold = max_distance * 1.1  # Add 10% buffer to ensure all connections are captured

    return distances_flat, suggested_threshold

def find_leaflets_by_graph(coords, ids, distances, distance_threshold):
    """
    Trouve les feuillets en construisant un graphe de connexions sur la base d'un seuil de distance réglable,
    en veillant à ce que les petits composants soient fusionnés avec les plus grands, mais que les grands composants ne fusionnent pas entre eux.
    """
    # Initialisation du graphe
    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, lipid_id=ids[i])

    # Construction des arêtes initiales
    mask = distances < distance_threshold
    np.fill_diagonal(mask, False)  # Pas de boucles sur soi-même
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

        # Filtre les nouvelles connexions
        filtered_edges = []
        for i, j in new_edges:
            # Si les deux nœuds sont dans des grandes composantes, on interdit la fusion
            if i in large_components and j in large_components:
                continue
            # Sinon, on autorise la fusion entre petites et grandes composantes
            elif (i in small_components and j in large_components) or (j in small_components and i in large_components) or (i in small_components and j in small_components):
                filtered_edges.append((i, j))
                G.add_edges_from(filtered_edges)
                # Mets à jour les petites et grandes composantes après chaque ajout
            components = list(nx.connected_components(G))
            small_components = {node for comp in components if len(comp) <= size_limit for node in comp}
            large_components = {node for comp in components if len(comp) > size_limit for node in comp}


        print(f"Updated distance threshold: {new_distance_threshold}")
        print("Updated Number of edges:", len(G.edges))


    # Résultat final
    components = list(nx.connected_components(G))
    leaflets = components
    print("Final Number of components:", len(components))
    print("Final Component sizes:", [len(c) for c in components])

    return leaflets, G




def analyse_membrane(filename):
    print(filename)
    coords, ids, box_dimensions = read_gromacs_file_with_mdanalysis(filename)

    plot_raw_membrane(coords, title="Raw Membrane Structure Before Leaflet Computation")

    # Step 3: Analyze de la distribution
    print("Analyzing pairwise distances...")
    #plot_pairwise_distance_distribution(coords, box_dimensions)

    print("Analyzing z-coordinate distribution...")
    plot_z_coordinate_distribution(coords)

    # ce seuil obtient simplement la médiane de l'axe z. Pour une structure membranaire simple,
    # Cette coupure correspond à la coupure de l'axe z des coordonnées moyennes entre les deux feuillets.
    z_threshold = compute_z_threshold(coords)
    distance_threshold = 5

    print("Current distance threshold", distance_threshold)
    distances = compute_periodic_distances_with_mdanalysis_gpu(coords, box_dimensions)
    distances_flat = plot_all_pairwise_distances(distances, title="Distribution of Pairwise Distances in the System")

    print(f"Max Distance: {np.max(distances_flat):.2f}")
    print(f"Min Distance: {np.min(distances_flat):.2f}")
    print(f"Mean Distance: {np.mean(distances_flat):.2f}")
    print(f"Median Distance: {np.median(distances_flat):.2f}")

    leaflets, graph = find_leaflets_by_graph(coords, ids, distances, distance_threshold)

    # Visualise la structure de la membrane en 3D
    plot_membrane(coords, leaflets)

    return leaflets, ids

def save_results(leaflets, ids, output_file):
    """
    Save the leaflets to a file, listing lipid IDs instead of node indices.
    Sauvegarde les feuillets dans un fichier, liste les ID des lipides plûtot que l'indice des noeuds
    """
    with open(output_file, 'w') as f:
        for i, leaflet in enumerate(leaflets, 1):
            # Convertit l'ensemble en liste pour permettre l'indexation
            leaflet = list(leaflet)

            # Mappe les indices des noeuds aux identifiants des lipides
            lipid_ids = [ids[n] for n in leaflet]
            lipid_ids.sort()
            f.write(f"[ membrane_{1}_leaflet_{i} ]\n")
            # Écris les lipides, 15 par ligne
            for j in range(0, len(lipid_ids), 15):
                row = lipid_ids[j:j+15]
                formatted_row = " ".join(f"{lipid:5d}" for lipid in row)
                f.write(f"   {formatted_row}\n")

def compute_z_threshold(coords):
    z_median = np.median(coords[:, 2])
    return z_median


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
