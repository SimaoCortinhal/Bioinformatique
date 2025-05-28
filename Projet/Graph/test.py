import MDAnalysis as mda

u = mda.Universe("data/bilayer_chol/bilayer_chol.gro")

# select lipids
lipids = u.select_atoms("resname POPC or resname CHOL or resname DPPC")
    
# print the average of the coordinates of all atoms in each lipid
for lipid in lipids.residues:
    print(lipid.resname, lipid.atoms.positions.mean(axis=0))