from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.colors as mcolors
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import glob
import re
import shutil

def get_min_max_coordinates(pdb_file):
    # Crée un parser pour lire le fichier PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    
    # Initialise les valeurs min et max
    x_min = y_min = z_min = float('inf')
    x_max = y_max = z_max = float('-inf')
    
    # Listes pour stocker les coordonnées des atomes
    x_coords = []
    y_coords = []
    z_coords = []
    
    # Parcourir tous les atomes pour trouver les coordonnées min et max
    for atom in structure.get_atoms():
        x, y, z = atom.coord
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        x_min, y_min, z_min = min(x_min, x), min(y_min, y), min(z_min, z)
        x_max, y_max, z_max = max(x_max, x), max(y_max, y), max(z_max, z)
    
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
    }

def calculate_max_distance(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    max_distance = 0

    # Parcourir toutes les paires d'atomes pour calculer la distance maximale
    atoms = list(structure.get_atoms())
    for i, atom1 in enumerate(atoms):
        for atom2 in atoms[i+1:]:
            distance = atom1 - atom2
            max_distance = max(max_distance, distance)

    return max_distance

def get_cube_dimensions_with_external_margin(result, margin_distance, grid_size=3):
    # Utiliser les dimensions exactes de la molécule pour définir les limites
    x_min, x_max = result["x_min"], result["x_max"]
    y_min, y_max = result["y_min"], result["y_max"]
    z_min, z_max = result["z_min"], result["z_max"]

    # Ajouter la marge de distance maximale calculée depuis le second fichier PDB
    x_min -= margin_distance
    x_max += margin_distance
    y_min -= margin_distance
    y_max += margin_distance
    z_min -= margin_distance
    z_max += margin_distance

    # Calcul de la longueur des sous-cubes (moitié de la longueur totale après ajout de la marge)
    cube_length = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    return {
        "cube_length_with_overlap": cube_length,
        "number_of_cubes": grid_size ** 3,
        "total_grid_dimensions": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        }
    }

def plot_coordinates_with_grid(result, margin_distance, grid_size=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Afficher les points des atomes
    ax.scatter(result["x_coords"], result["y_coords"], result["z_coords"], c='blue', marker='o', s=5, label='Atomes')
    
    # Calcul des dimensions du cube avec marge de distance maximale du second fichier
    dimensions = get_cube_dimensions_with_external_margin(result, margin_distance)
    x_min = dimensions["total_grid_dimensions"]["x_min"]
    x_max = dimensions["total_grid_dimensions"]["x_max"]
    y_min = dimensions["total_grid_dimensions"]["y_min"]
    y_max = dimensions["total_grid_dimensions"]["y_max"]
    z_min = dimensions["total_grid_dimensions"]["z_min"]
    z_max = dimensions["total_grid_dimensions"]["z_max"]
    cube_length = dimensions["cube_length_with_overlap"]
    
    # Liste de couleurs pour les sous-cubes
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Affichage des points du premier sous-cube et calcul des distances
    first_cube_vertices = [
        [x_min, y_min, z_min],
        [x_min + cube_length, y_min, z_min],
        [x_min + cube_length, y_min + cube_length, z_min],
        [x_min, y_min + cube_length, z_min],
        [x_min, y_min, z_min + cube_length],
        [x_min + cube_length, y_min, z_min + cube_length],
        [x_min + cube_length, y_min + cube_length, z_min + cube_length],
        [x_min, y_min + cube_length, z_min + cube_length]
    ]
    
    print("Coordonnées des sommets du premier cube :")
    for vertex in first_cube_vertices:
        print(vertex)
    
    print("\nDistances entre les sommets adjacents du premier cube :")
    for i in range(4):  # Calculer les distances entre sommets sur la première face
        dist = math.dist(first_cube_vertices[i], first_cube_vertices[(i + 1) % 4])
        print(f"Distance entre le sommet {i} et {((i + 1) % 4)} : {dist}")
    print(f"Distance en profondeur entre le sommet 0 et 4 : {math.dist(first_cube_vertices[0], first_cube_vertices[4])}")

    # Générer les sous-cubes de taille moitié du cube englobant
    color_index = 0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Calculer les coordonnées de départ de chaque sous-cube pour bien les disposer
                x_start = x_min + i * (cube_length / 2)
                y_start = y_min + j * (cube_length / 2)
                z_start = z_min + k * (cube_length / 2)
                
                # Définir les sommets du sous-cube
                vertices = [
                    [x_start, y_start, z_start],
                    [x_start + cube_length, y_start, z_start],
                    [x_start + cube_length, y_start + cube_length, z_start],
                    [x_start, y_start + cube_length, z_start],
                    [x_start, y_start, z_start + cube_length],
                    [x_start + cube_length, y_start, z_start + cube_length],
                    [x_start + cube_length, y_start + cube_length, z_start + cube_length],
                    [x_start, y_start + cube_length, z_start + cube_length]
                ]
                
                # Définir les faces du sous-cube sans diagonales
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Face inférieure
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Face supérieure
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Face avant
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Face arrière
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Face droite
                    [vertices[4], vertices[7], vertices[3], vertices[0]]   # Face gauche
                ]
                
                # Créer le sous-cube avec une couleur différente pour chaque
                color = colors[color_index % len(colors)]
                sub_cube = Poly3DCollection(faces, color=color, alpha=0.1, edgecolors='k')
                ax.add_collection3d(sub_cube)
                color_index += 1
    
    # Ajouter des légendes et des axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Représentation")
    plt.savefig("grid_with_atoms.png")



def generate_and_run_for_cube(i, j, k, result, margin_distance, grid_size=3, base_output_dir="./"):
    # Créer un dossier pour chaque thread
    thread_dir = os.path.join(base_output_dir, f"cube_{i}_{j}_{k}")
    if not os.path.exists(thread_dir):
        os.makedirs(thread_dir)

    # Obtenir les dimensions et les coordonnées de la grille
    dimensions = get_cube_dimensions_with_external_margin(result, margin_distance)
    x_min = dimensions["total_grid_dimensions"]["x_min"]
    y_min = dimensions["total_grid_dimensions"]["y_min"]
    z_min = dimensions["total_grid_dimensions"]["z_min"]
    cube_length = dimensions["cube_length_with_overlap"] / 2  # Diviser pour obtenir des sous-cubes

    # Calcul des coordonnées de départ pour chaque sous-cube
    x_start = x_min + i * cube_length
    y_start = y_min + j * cube_length
    z_start = z_min + k * cube_length

    # Calcul des coordonnées du centre du sous-cube
    x_center = x_start + cube_length / 2
    y_center = y_start + cube_length / 2
    z_center = z_start + cube_length / 2

    # Définir les dimensions `npts` pour chaque sous-cube
    npts_x = npts_y = npts_z = int((cube_length + margin_distance) / 0.375)  # Ajuster selon le spacing

    # Créer le contenu du fichier pour le sous-cube
    file_content = f"""npts {npts_x} {npts_y} {npts_z}                     # num.grid points in xyz
gridfld 3thc.maps.fld                # grid_data_file
spacing 0.375                        # spacing(A)
receptor_types A C H HD N OA SA      # receptor atom types
ligand_types C HD OA                 # ligand atom types
receptor 3thc.pdbqt                  # macromolecule
gridcenter {x_center:.3f} {y_center:.3f} {z_center:.3f}         # xyz-coordinates or auto
smooth 0.5                           # store minimum energy w/in rad(A)
map 3thc.C.map                       # atom-specific affinity map
map 3thc.HD.map                      # atom-specific affinity map
map 3thc.OA.map                      # atom-specific affinity map
elecmap 3thc.e.map                   # electrostatic potential map
dsolvmap 3thc.d.map                  # desolvation potential map
dielectric -0.1465                   # <0, AD4 distance-dep.diel;>0, constant"""

    # Nommer le fichier pour chaque sous-cube
    file_name = f"cube_{i}_{j}_{k}.gpf"
    file_path = os.path.join(thread_dir, file_name)
    log_file = file_name.replace(".gpf", ".glg")

    # Écrire le contenu dans le fichier
    with open(file_path, "w") as f:
        f.write(file_content)
    print(f"Fichier généré : {file_path}")
    
    # Copie du fichier pdbqt
    subprocess.run(["cp", "3thc.pdbqt", thread_dir])
    subprocess.run(["cp", "galactose.pdbqt", thread_dir])
    print(f"3thc.pdbqt et galactose.pdbqt copiés dans : {thread_dir}")
    
    # Lancer autogrid4 dans le répertoire de travail spécifique
    subprocess.run(["autogrid4", "-p", file_name, "-l", log_file], cwd=thread_dir)
    print(f"Commande autogrid4 exécutée pour : {file_path}")

    # Lancer autodock4 pour le docking
    
    # Copie du fichier dpf
    dpf_file = "galactose.dpf"
    subprocess.run(["cp", dpf_file, thread_dir])
    
    # output_file = os.path.join(thread_dir, f"galactose_{i}_{j}_{k}.dlg")
    subprocess.run(["autodock4", "-p", dpf_file, "-l", f"galactose_{i}_{j}_{k}.dlg"], cwd=thread_dir)
    print(f"Docking exécutée pour : cube_{i}_{j}_{k}")


def find_top_three_lowest_binding_energies(dlg_directory, num_files=3):
    energy_pattern = re.compile(r"Estimated Free Energy of Binding\s+=\s+(-?\d+\.\d+)\s+kcal/mol")
    energy_data = []

    # Parcourir tous les fichiers .dlg dans le répertoire
    for dlg_file in glob.glob(os.path.join(dlg_directory, "*.dlg")):
        min_energy = float('inf')  # Initialiser à une valeur élevée pour chaque fichier

        # Exécuter grep pour extraire les lignes contenant "Estimated Free Energy of Binding"
        result = subprocess.run(["grep", "Estimated Free Energy of Binding", dlg_file], capture_output=True, text=True)
        
        # Vérifier chaque ligne pour trouver la plus basse énergie dans ce fichier
        for line in result.stdout.splitlines():
            match = energy_pattern.search(line)
            if match:
                energy = float(match.group(1))
                min_energy = min(min_energy, energy)  # Garder la plus basse énergie pour ce fichier

        # Ajouter la plus basse énergie et le nom du fichier à la liste
        if min_energy < float('inf'):
            energy_data.append((dlg_file, min_energy))

    # Trier les fichiers par énergie la plus basse et sélectionner les trois premiers
    top_files = sorted(energy_data, key=lambda x: x[1])[:num_files]

    # Afficher les trois fichiers avec les énergies les plus basses
    print(f"Les {num_files} fichiers avec les énergies de liaison les plus basses :")
    for file, energy in top_files:
        print(f"Fichier: {file}, Énergie: {energy} kcal/mol")


def main(result, margin_distance, grid_size=3, base_output_dir="./output_cubes"):
    with ThreadPoolExecutor(max_workers=grid_size**3) as executor:
        futures = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Planifier chaque tâche pour un sous-cube dans un sous-dossier unique
                    futures.append(executor.submit(generate_and_run_for_cube, i, j, k, result, margin_distance, grid_size, base_output_dir))
        
        # Attendre que toutes les tâches soient terminées
        for future in futures:
            future.result()



# Chargement des deux fichiers PDB
pdb_file_1 = "3thc.pdb"  # Remplacez par le chemin de votre premier fichier PDB
pdb_file_2 = "galactose.pdb"  # Remplacez par le chemin de votre second fichier PDB

# Calcul des coordonnées min/max pour le premier fichier
result = get_min_max_coordinates(pdb_file_1)

# Calcul de la distance maximale dans le second fichier PDB pour définir la marge
margin_distance = calculate_max_distance(pdb_file_2)
print(f"Marge de distance maximale provenant du second fichier PDB : {margin_distance}")

# Affichage de la grille avec la marge de distance maximale et affichage des sommets du premier cube
plot_coordinates_with_grid(result, margin_distance)
# generate_cube_files(result, margin_distance)
main(result, margin_distance)
# # Lancement des commandes autogrid et autodock

# Moove all dlg files to dlg_output
os.makedirs("dlg_output", exist_ok=True)

for dlg_file in glob.glob("output_cubes/*/galactose_*.dlg"):
    shutil.move(dlg_file, "dlg_output/")

find_top_three_lowest_binding_energies("dlg_output", 5)