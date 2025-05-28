import MDAnalysis as mda
import numpy as np

liste = []
liste.extend(["CL", "ACE", "LYS", "LEU", "GLY", "ALA", "NH2", "MET", "GLN", "PRO", "ILE", "VAL", "SER", "TRP", "TYR", "GLU", "ARG"])

# Charger le fichier GRO
gro_file_path = './data/bilayer_peptide/bilayer_peptide.gro'  # Remplacez par le chemin correct
u = mda.Universe(gro_file_path)

# Sélection des molécules d'eau (resname=W) et des lipides
water_atoms = u.select_atoms("resname SOL or resname W")  # Sélectionner les atomes d'eau
lipid_residues = u.select_atoms("not resname SOL and not resname W").residues  # Regrouper les atomes par résidu (molécule)

# Récupérer les coordonnées des molécules d'eau
water_coords = water_atoms.positions

# Calculer la moyenne des coordonnées Z pour séparer les parties haute et basse
z_mean = np.mean(water_coords[:, 2])

# Séparer les molécules d'eau
water_top = water_coords[water_coords[:, 2] > z_mean]
water_bottom = water_coords[water_coords[:, 2] <= z_mean]

# Associer chaque molécule de lipide à la partie d'eau la plus proche
lipid_to_water_mapping = {"top": [], "bottom": []}
for residue in lipid_residues:
    # Calculer la position moyenne (centre de masse approximé) de la molécule
    lipid_center = residue.atoms.positions.mean(axis=0)

    # Calculer les distances entre le centre de la molécule et les molécules d'eau
    distances_to_top = np.linalg.norm(water_top - lipid_center, axis=1)
    distances_to_bottom = np.linalg.norm(water_bottom - lipid_center, axis=1)

    # Associer la molécule au groupe d'eau le plus proche 
    #exclure mot de la liste
    if residue.resname not in liste:
        if np.min(distances_to_top) < np.min(distances_to_bottom) :
            lipid_to_water_mapping["top"].append(residue.resid)  # Utiliser le resid comme ID de la molécule
        else:
            lipid_to_water_mapping["bottom"].append(residue.resid)

# Formater les lignes pour qu'elles contiennent 15 valeurs et aient une largeur de 91 caractères
def format_lines(values):
    lines = []
    for i in range(0, len(values), 15):
        group = values[i:i+15]
        formatted_line = "   ".join(f"{x:5}" for x in group).ljust(91)  # Espacement pour atteindre 91 caractères
        lines.append(formatted_line)
    return lines

# Générer le contenu formaté
output = []


output.append("[ membrane_1 ]")
output.extend(format_lines(lipid_to_water_mapping["bottom"]))

output.append("[ membrane_2 ]")
output.extend(format_lines(lipid_to_water_mapping["top"]))

# Sauvegarder dans un fichier
output_file_path = 'lipid_leaflet_assignment_formatted.txt'
with open(output_file_path, 'w') as f:
    for line in output:
        f.write(line + '\n')

print(f"Fichier généré : {output_file_path}")