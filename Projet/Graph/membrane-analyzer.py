import numpy as np
import networkx as nx
import MDAnalysis as mda
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import cuda, float32, prange, njit
import math
import sys
import argparse

from utils import *

def read_gromacs_file_with_mdanalysis(filename):
    """
    Lecture d'un fichier GROMACS (.gro) avec MDAnalysis pour sélectionner les lipides
    et récupérer les coordonnées du premier atome de chaque lipide,
    leurs identifiants et les dimensions de la boîte.

    :param filename: Chemin vers le fichier GROMACS (.gro)
    :return: tuple contenant les coordonnées des premiers atomes des lipides, leurs identifiants, et les dimensions de la boîte
    """
    # Charger l'univers
    u = mda.Universe(filename)
    
    # Définir les résidus non-lipidiques à exclure
    not_lipid = ["W", "NA+", "CL-", "SOL", "HOH","CL","ACE","LYS","LEU","GLY","ALA","NH2","MET","GLN","PRO","ILE","VAL","SER","TRP","TYR","GLU","ARG"]

    # Construire une sélection MDAnalysis
    lipid_selection = u.select_atoms(
        " and ".join([f"not resname {res}" for res in not_lipid])
    )

    # Vérifier si la sélection est vide
    if len(lipid_selection) == 0:
        raise ValueError("Aucun lipide trouvé dans la sélection. Vérifiez le fichier ou la sélection.")
    
    lipid_residues = lipid_selection.residues
    reflipid_coords = [res.atoms[0].position for res in lipid_residues]
    lipid_ids = lipid_selection.residues.resids
    box_dimensions = u.dimensions
    
    orientation_coords = []
    lipid_coords = []
    print(len(lipid_ids))
    for lipid_id in set(lipid_ids):
        lipid_atoms = u.select_atoms(f"resid {lipid_id}")
        #print(lipid_atoms)

        # Récupérer les coordonnées de tous les atomes du lipide
        all_atom_coords = lipid_atoms.positions
        lipid_coords.append(all_atom_coords)

        if len(lipid_atoms) >= 2:
            atom1_coord = lipid_atoms[0].position #head
            atom2_coord = lipid_atoms[-1].position #tail

            orientation_coords.append((atom1_coord, atom2_coord))
        else:
            orientation_coords.append((None, None))

    # Debugging ou information utile
    print(f"Nombre de lipides : {len(lipid_coords)}")
    
    reflipid_coords = np.array(reflipid_coords)

    # Retourner les résultats
    return reflipid_coords, lipid_ids, box_dimensions, lipid_coords, orientation_coords

def compute_periodic_distances_with_mdanalysis(coords, box_dimensions):
    """
    Calculer les distances par paire de lipides. 
    Utilisation de MDAnalysis pour appliquer les conditions aux limites périodiques.
    """
    from MDAnalysis.lib.distances import apply_PBC

    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            diff_in_box = apply_PBC(diff, box_dimensions)
            #distance = sqrt(x2​−x1​)² + (y2​−y1​)² + (z2​−z1​)²
            distance = np.linalg.norm(diff)
            #print(coords[i], coords[j], diff, diff_in_box, distance)
            distances[i, j] = distances[j, i] = distance

    return distances

@cuda.jit
def compute_distances_kernel(coords, box_dimensions, distances, n):
    """
    CUDA kernel to compute pairwise distances with periodic boundary conditions.
    """
    i, j = cuda.grid(2)
    if i < n and j < n and i < j:  # matrice triangulaire pour éviter les calculs redondants
        diff = cuda.local.array(3, dtype=float32)
        diff_in_box = cuda.local.array(3, dtype=float32)

        # difference pour chaque dimension
        for k in range(3):
            diff[k] = coords[i, k] - coords[j, k]
            diff_in_box[k] = diff[k] - round(diff[k] / box_dimensions[k]) * box_dimensions[k]

        # calcul distance
        distance = 0.0
        for k in range(3):
            distance += diff_in_box[k] ** 2
        distance = math.sqrt(distance)

        # ajout de la distance à la matrice
        distances[i, j] = distance
        distances[j, i] = distance #symétrie

def compute_periodic_distances_with_mdanalysis_gpu(coords, box_dimensions):
    """
    Calculer les distances périodiques par paire en utilisant Numba CUDA.
    
    Parameters :
        coords : Tableau de forme (n, 3) avec les coordonnées atomiques.
        box_dimensions : Tableau de formes (3,) avec les dimensions de la boîte.
    
    Returns :
        distances : Matrice de distance symétrique de la forme (n, n).
    """
    n = coords.shape[0]

    coords = coords.astype(np.float32)
    box_dimensions = box_dimensions.astype(np.float32)

    d_coords = cuda.to_device(coords)
    d_box_dimensions = cuda.to_device(box_dimensions)
    d_distances = cuda.device_array((n, n), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    compute_distances_kernel[blocks_per_grid, threads_per_block](d_coords, d_box_dimensions, d_distances, n)

    distances = d_distances.copy_to_host()
    return distances

def compute_orientation(orientation_coords, lipid_idx):
    """
    Calcule le vecteur d'orientation d'un lipide sur la base des atomes sélectionnés.
    `atom_indices` doit spécifier deux atomes pour définir le vecteur d'orientation.
    """
    ref_atom1 = orientation_coords[lipid_idx][0]
    ref_atom2 = orientation_coords[lipid_idx][1]
    
    # vecteur entre deux atomes
    orientation_vector = ref_atom2 - ref_atom1
    # normalisation
    orientation_vector /= np.linalg.norm(orientation_vector)

    return orientation_vector

def calculate_cosine_similarity(vector1, vector2):
    """
    Calculer la similarité en cosinus entre deux vecteurs.
    Cosine similarity = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    """
    #cosine = np.dot(A,B)/(norm(A)*norm(B))
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def compute_geometrical_center(coords):
    """
    Calculer le centre approximatif d'un lipide à partir des coordonnées de ses atomes.
    """
    return np.mean(coords, axis=0) #centroid

def get_ref_coords_from_geometry(coords):
    """
    Calculer les coordonnées de référence pour chaque lipide en fonction de sa géométrie.
    """
    ref_coords = []
    for lipid_coords in coords:
        ref_coords.append(compute_geometrical_center(lipid_coords))
    return ref_coords

def compute_normal_vectors(ref_coords, distances, distance_threshold, max_neighbors=12):
    """
    /!\ [DEPRECATED]
    Calculer les vecteurs normaux pour chaque lipide à partir des coordonnées de l'atome de référence
    en se basant sur les lipides environnants (voisins les plus proches).
    """
    normal_vectors = np.zeros((n, 3))
    n = len(ref_coords)
    for i in range(n):
        neighbours = []
        nb_neighbors = 0
        for j in range(i + 1, n):
            if distances[i, j] < distance_threshold:
                neighbour = ref_coords[j] - ref_coords[i]
                neighbours.append(neighbour)
                nb_neighbors += 1
        local_normal = np.mean(neighbours, axis=0)
        # normalisation
        local_normal /= np.linalg.norm(local_normal)
        normal_vectors[i] = local_normal

    return normal_vectors

def optimized_normal_vectors(ref_coords, distances, distance_threshold, max_neighbors=12):
    """
    Calculer les vecteurs normaux pour chaque lipide à partir des coordonnées de l'atome de référence
    en se basant sur les lipides environnants (voisins les plus proches).
    Cette version optimisée utilise numpy pour éviter les boucles et accélérer le calcul.
    Parameters :
        ref_coords : Coordonnées des atomes de référence pour chaque lipide.
        distances : Matrice de distances par paire.
        distance_threshold : Seuil de distance pour la connexion des lipides.
        max_neighbors : Nombre maximum de voisins à considérer pour chaque lipide.
    Returns :
        normal_vectors : Tableau de vecteurs normaux pour chaque lipide.
    """
    n = len(ref_coords)
    normal_vectors = np.zeros((n, 3))
    for i in range(n):
        neighbor_mask = (distances[i] < distance_threshold) & (distances[i] > 0)
        neighbors = ref_coords[neighbor_mask] - ref_coords[i]

        if len(neighbors) == 0:
            print(f"No neighbors found for lipid {i}")
            normal_vectors[i] = [0, 0, 0]
        else:
            # normal local
            local_normal = np.mean(neighbors, axis=0)

            # normalisation
            norm = np.linalg.norm(local_normal)
            if norm == 0:  # division par zéro
                normal_vectors[i] = [0, 0, 0]
            else:
                normal_vectors[i] = local_normal / norm

    return normal_vectors

def compute_orientation_vector(orientation_coords):
    """
    Calcule les vecteurs d'orientation pour chaque lipide sur la base des atomes sélectionnés.
    Parameters :
    orientation_coords : Une liste dont chaque élément est une liste de coordonnées 
        représentant les positions des atomes dans un lipide.
    Returns :
        Un tableau de vecteurs d'orientation pour chaque lipide.
    """
    orientation_vectors = []
    for lipid_idx in range(len(orientation_coords)):
        orientation_vector = compute_orientation(orientation_coords, lipid_idx)
        orientation_vectors.append(orientation_vector)
    return np.array(orientation_vectors)

@cuda.jit
def compute_cosine_similarity_matrix_gpu(orientation_vectors, similarity_matrix):
    """
    Kernel GPU pour calculer la matrice de similarité cosinus
    pour toutes les paires de vecteurs d'orientation.
    Paraleters :
        orientation_vectors : Tableau 2D de vecteurs d'orientation (Nx3).
        similarity_matrix : Tableau de sortie 2D pour stocker la similarité en cosinus (NxN).
    """
    i, j = cuda.grid(2)
    n = orientation_vectors.shape[0]

    if i < n and j < n and i < j:  # eviter les calculs redondants
        # charger les vecteurs d'orientation en mémoire locale
        v1 = orientation_vectors[i]
        v2 = orientation_vectors[j]

        # calculer le produit scalaire et les normes
        dot_product = 0.0
        norm_v1 = 0.0
        norm_v2 = 0.0
        for k in range(3): # x, y, z
            dot_product += v1[k] * v2[k]
            norm_v1 += v1[k] ** 2
            norm_v2 += v2[k] ** 2

        # calcule de la similarité
        norm_v1 = math.sqrt(norm_v1)
        norm_v2 = math.sqrt(norm_v2)
        if norm_v1 > 0 and norm_v2 > 0:
            similarity = dot_product / (norm_v1 * norm_v2)
        else:
            similarity = 0.0

        # stocker la similarité dans la matrice
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # symétrie

def compute_cosine_similarity_matrix(orientation_vectors):
    """
    Fonction permettant d'allouer la mémoire du GPU et de lancer le noyau 
    permettant de calculer la matrice de similarité par cosinus.
    Parameters :
        orientation_vectors : Tableau numpy 2D de vecteurs d'orientation (Nx3).
    Returns :
        similarity_matrix : Tableau numpy 2D de similarités en cosinus (NxN).
    """
    n = len(orientation_vectors)

    # convertir en float32 pour la compatibilité avec CUDA
    orientation_vectors = np.array(orientation_vectors, dtype=np.float32)

    # allocation de la mémoire sur le périphérique
    d_orientation_vectors = cuda.to_device(orientation_vectors)
    d_similarity_matrix = cuda.device_array((n, n), dtype=np.float32)

    # definition de la grille et des blocs
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # lancement du kernel
    compute_cosine_similarity_matrix_gpu[blocks_per_grid, threads_per_block](d_orientation_vectors, d_similarity_matrix)

    # copie du résultat sur l'hôte
    similarity_matrix = d_similarity_matrix.copy_to_host()

    return similarity_matrix

def find_leaflets_by_graph(ref_coords, ids, distances, distance_threshold, orientations, orientation_threshold, local_normals, normal_threshold, debug=False):
    """
    Trouve les feuillets en construisant un graphe de connexions basé sur une distance réglable et un seuil d'orientation,
    en veillant à ce que les petits composants soient fusionnés avec les plus grands.

    Parameters :
        ref_coords (np.array) : Coordonnées des points de référence des lipides.
        ids (np.array) : Identifiants des lipides.
        distances (np.array) : Matrice de distance par paire (tableau 2D).
        distance_threshold (float) : Seuil initial pour la connexion des lipides.
        orientations (np.array) : Matrice de similarité cosinus pour les orientations des lipides.
        orientation_threshold (float) : Seuil pour considérer les lipides comme étant dans le même feuillet sur la base de l'orientation.

    Returns :
        leaflets (liste d'ensembles) : Composantes connectées représentant les feuillets.
        G (nx.Graph) : Représentation graphique des connexions entre les lipides.
    """
    import networkx as nx

    n = len(ref_coords)
    
    # Pré-calcul des produits scalaires des vecteurs normaux
    scalar_products = np.dot(local_normals, local_normals.T)

    # graph init
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, lipid_id=ids[i])

    edges_dist = [0, 0]

    # Ajout des arêtes en fonction des distances et des orientations
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < distance_threshold:
                scalar = scalar_products[i, j]
                if scalar > normal_threshold or orientations[i, j] > orientation_threshold:
                    G.add_edge(i, j)
                    edges_dist[0] += 1
                else:
                    edges_dist[1] += 1
            else:
                edges_dist[1] += 1

    if debug:
        print("[DEBUG] Edges distribution:", edges_dist)
        nx.draw(G, with_labels=False, node_size=10)
        plt.title("Graph of Lipid Connections with Orientation Threshold")
        plt.show()

    print("Initial Distribution of edges:", edges_dist)
    print("Initial Number of edges:", len(G.edges))
    print("Initial Number of nodes:", len(G.nodes))

    # Ajout des arêtes pour fusionner les petites composantes
    new_distance_threshold = distance_threshold
    size_limit = len(G.nodes) // 4  # Seuil pour les petites composantes
    print("Size limit for small components:", size_limit)

    mask = distances < distance_threshold

    while len(list(nx.connected_components(G))) > 2:
        components = list(nx.connected_components(G))
        component_sizes = [len(c) for c in components]
        
        print("Number of components:", len(components))
        print("Component sizes:", component_sizes)

        # Recalculer les composantes
        small_components = {node for comp in components if len(comp) <= size_limit for node in comp}
        large_components = {node for comp in components if len(comp) > size_limit for node in comp}
        
        # Mettre à jour le seuil de distance
        new_distance_threshold += 1
        mask = (distances < new_distance_threshold) & ~mask
        new_edges = np.column_stack(np.where(mask))

        # Filtrer les arêtes pour les composantes de petite taille
        filtered_edges = []
        for i, j in new_edges:
            # Ignorer les arêtes qui fusionnent deux grandes composantes
            if i in large_components and j in large_components:
                continue

            scalar = np.dot(local_normals[i], local_normals[j])
            if scalar > normal_threshold or orientations[i, j] > orientation_threshold:
                # Autoriser les arêtes reliant des petits et des grands composants ou deux petits composants
                if (i in small_components and j in large_components) or \
                (j in small_components and i in large_components) or \
                (i in small_components and j in small_components):
                    filtered_edges.append((i, j))
        
        # Ajouter les arêtes filtrées au graphe
        G.add_edges_from(filtered_edges)

        print(f"Updated distance threshold: {new_distance_threshold}")
        print("Updated Number of edges:", len(G.edges))

    # résultats finaux
    components = list(nx.connected_components(G))
    leaflets = components
    print("Final Number of components:", len(components))
    print("Final Component sizes:", [len(c) for c in components])

    return leaflets, G

def analyse_membrane(filename, args):
    coords, ids, box_dimensions, lipid_coords, orientation_coords = read_gromacs_file_with_mdanalysis(filename)
    #print(coords, ids, box_dimensions, lipid_coords, orientation_coords)
    
    print("Analyzing raw membrane structure...")
    if args.show_plot:
        plot_raw_membrane(coords, title="Raw Membrane Structure Before Leaflet Computation")
    
    print("Analyzing lipid orientations...")
    if args.show_plot:
        plot_lipid_orientations(orientation_coords)

    orientation_vectors = compute_orientation_vector(orientation_coords)
    print(orientation_vectors.shape, coords.shape)

    print("Analyzing orientation vectors...")
    if args.show_plot:
        visualize_lipid_orientations(coords, orientation_vectors)

    print("Analyzing cosine similarity...")
    orientation_similarity_matrix = compute_cosine_similarity_matrix(orientation_vectors)
    if args.show_plot:
        plot_cosine_similarity_histogram(orientation_similarity_matrix)

    oritentation_threshold = np.percentile(np.ravel(orientation_similarity_matrix), 60)
    oritentation_threshold = 0.5

    print("Orientation threshold:", oritentation_threshold)

    print(f"Max Cosine: {np.max(orientation_similarity_matrix):.2f}")
    print(f"Min Cosine: {np.min(orientation_similarity_matrix):.2f}")
    print(f"Mean Cosine: {np.mean(orientation_similarity_matrix):.2f}")
    print(f"Median Cosine: {np.median(orientation_similarity_matrix):.2f}")
    
    print("Analyzing z-coordinate distribution...")
    if args.show_plot:
        plot_z_coordinate_distribution(coords)
    
    # ce seuil est simplement obtenu par la médiane des z. Pour une structure membranaire simple, 
    # cette coupure est la coupure de l'axe z des coordonnées médianes entre les 2 feuillets
    z_threshold = compute_z_threshold(coords)
    print("Leaflet Z-axis separation threshold:", z_threshold)

    distance_threshold = 10
    
    print("Current distance threshold", distance_threshold)
    distances = compute_periodic_distances_with_mdanalysis_gpu(coords, box_dimensions)

    print("Analyzing pairwise distances...")
    if args.show_plot:
        distances_flat = plot_all_pairwise_distances(distances, title="Distribution of Pairwise Distances in the System")

        max_distance = np.max(distances_flat)
        min_distance = np.min(distances_flat)
        mean_distance = np.mean(distances_flat)
        median_distance = np.median(distances_flat)
    
        print(f"Max Distance: {max_distance:.2f}")
        print(f"Min Distance: {min_distance:.2f}")
        print(f"Mean Distance: {mean_distance:.2f}")
        print(f"Median Distance: {median_distance:.2f}")
    
    print("Computing local normal vectors...")
    normal_vectors = optimized_normal_vectors(coords, distances, 25, max_neighbors=12)
    print(normal_vectors)

    if args.show_plot:
        plot_normal_vectors(coords, normal_vectors, 8)

    normal_threshold = 0.5

    print("Computing leaflets...")
    leaflets, graph = find_leaflets_by_graph(
        coords, ids, distances, distance_threshold,
        orientation_similarity_matrix, oritentation_threshold,
        normal_vectors, normal_threshold,
        debug=args.debug
    )

    if args.debug:
        nx.draw(graph, with_labels=False, node_size=10)
        plt.title("Graph of Lipid Connections")
        plt.show()
    
    # Afficher les résultats et la division des feuillets
    plot_membrane(coords, leaflets)
    
    return leaflets, ids

def save_results(leaflets, ids, output_file):
    """
    Enregistrer les feuillets dans un fichier (TXT), en indiquant les ID des lipides au lieu des indices des nœuds.
    """
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

def compute_z_threshold(coords):
    """
    Calculer le seuil de la coordonnée z pour séparer les feuillets.
    """
    z_median = np.median(coords[:, 2])
    return z_median

def get_filename(arg):
    """
    Obtenir le nom du fichier (indice) en fonction de l'argument d'entrée.
    """
    
    switcher = {
        "1": "bilayer_chol", # 1 diff
        "2": "bilayer_peptide", # 0 diff
        "3": "bilayer_prot", # 0 diff
        "4": "dppc_vesicle", # diff au premier leaflet (plus de lipide)
        "5": "large_plasma_membrane",  # 19 différences
        "6": "small_plasma_membrane",  # 15 différences
        "7": "model_vesicle" # 0 diff
    }
    return switcher.get(arg, "bilayer_chol")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membrane lipid leaflet analyzer.")
    parser.add_argument("filename", type=str, help="Name of the GROMACS file to analyze.")
    parser.add_argument("--show_plot", action="store_true", help="Display the plot if this flag is set.")
    parser.add_argument("--debug", action="store_true", help="Display the generated graphs if this flag is set.")
    parser.add_argument("--options", action="store_true", help="Display the filename options message.")
    args = parser.parse_args()
    
    if args.options:
        options = '''
        "1": "bilayer_chol",
        "2": "bilayer_peptide",
        "3": "bilayer_prot",
        "4": "dppc_vesicle",
        "5": "large_plasma_membrane",
        "6": "small_plasma_membrane",
        "7": "model_vesicle"
        '''
        print("Usage: python membrane-analyzer.py <filename> [--show_plot] [--debug]")
        print("Example: python membrane-analyzer.py bilayer_chol --show_plot")
        print("filename options:")
        print(options)

    filename = get_filename(sys.argv[1])
    print(f"Selected filename: {filename}")
    input_file = f"./data/{filename}/{filename}.gro"
    output_file = f"./output/leaflets_results{filename}.txt"
    
    leaflets, ids = analyse_membrane(input_file, args)
    save_results(leaflets, ids, output_file)
    
    print(f"Analysis complete. Results saved in {output_file}")
