import MDAnalysis as mda

# Remplacez 'sample.gro' par le chemin de votre fichier .gro
file_path = "projet-1001/data/bilayer_chol/bilayer_chol.gro"
u = mda.Universe(file_path)

# Obtenir les dimensions affichées par MDAnalysis
box = u.dimensions[:3]

# Diviser par 10 si les dimensions sont interprétées en angströms (à convertir en nm)
if box[0] > 100:  # Heuristique simple
    box = [dim / 10 for dim in box]

lipid_resnames = ["CHOL", "POPC", "DPPC", "DOPC"]


for residue in u.residues:
    if residue.resname in lipid_resnames:
        if (residue.resname == "CHOL"):
            atome_ref = residue.atoms[0]
        else:
            atome_ref = residue.atoms[1]
        print(f"Molécule ID: {residue.resid}, Nom: {residue.resname} est un lipide, Position de l'atome de référence : {atome_ref.position}")


print(f"Dimensions corrigées (nm) : x={box[0]:.5f}, y={box[1]:.5f}, z={box[2]:.5f}")