# Dossier TP Docking -  CORTINHAL Simão, BATISTE Quentin, COGNE Romain 
**Année 2024/2025 — M2 CHPS — CHPS1001 – CRTP Docking**

Ce dossier contient l’ensemble des fichiers utilisés et produits pour le TP de docking moléculaire autour de l’interaction entre le galactose et la β-galactosidase, en utilisant Autodock/AutoGrid et VMD.

## Contenu du dossier

- **3thc.pdb, 3thc.pdbqt** : Fichiers de structure de la protéine cible (format PDB et PDBQT).
- **Complexe_BetaGal_Galactose.pdb** : Fichier de référence pour le complexe cible/ligand.
- **galactose.pdb, galactose.pdbqt, galactose.dpf** : Fichiers décrivant le ligand galactose, utilisés pour le docking.
- **clear.sh** : Script bash pour nettoyer les fichiers générés lors des simulations.
- **notes.txt** : Notes et remarques diverses liées au TP.
- **README.md** : Ce fichier d’explications sur l’organisation du dossier.
- **script.py** : Script Python permettant de découper la molécule cible en plusieurs grilles pour des tests de docking plus précis.
- **TP_CHPS_Docking_2022_2023.pdf** : Sujet ou support de TP de référence.

---

## Résumé du TP

- **Étape 1** : Prise en main d’Autodock, réalisation de dockings sur l’ensemble de la molécule avec 10 et 50 lancements de l’algorithme génétique, comparaison avec le complexe de référence via VMD.
- **Étape 2** : Développement d’un script Python pour découper l’espace en 27 sous-grilles, réalisation de dockings parallélisés et comparaison des résultats, démontrant une amélioration nette de la précision.

