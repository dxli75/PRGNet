import sys
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, PDBIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.PDBIO import Select
import os
import numpy as np

warnings.simplefilter('ignore', BiopythonWarning)


def get_all_hetatm_residues(structure):
    het_residues = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith('H_'):
                    resname = residue.resname.strip()
                    het_residues.setdefault(resname, []).extend(residue.get_atoms())
    return het_residues


def get_ligand_center(structure):
    het_residues = get_all_hetatm_residues(structure)

    exclude_list = ['HOH', 'SO4', 'PO4', 'CL', 'NA', 'MG']
    candidates = {k: v for k, v in het_residues.items() if k not in exclude_list}

    if not candidates:
        raise ValueError("No valid ligand found")

    main_ligand = next(iter(candidates.values()))
    return np.mean([atom.get_coord() for atom in main_ligand], axis=0)


def validate_pocket(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('validation', pdb_file)
    except Exception as e:
        print(f"Failed to parse pocket file: {str(e)}", file=sys.stderr)
        return False

    REQUIRED_ATOMS = {'N', 'CA', 'C', 'O'}
    missing_records = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith('H_'):  
                    continue
                if not is_aa(residue, standard=True):
                    continue
                present_atoms = {a.name.strip() for a in residue}

                missing = REQUIRED_ATOMS - present_atoms
                if missing:
                    res_info = f"{chain.id}/{residue.id[1]} ({residue.resname})"
                    missing_records.append((res_info, sorted(missing)))

    missing_ca = [rec for rec in missing_records if 'CA' in rec[1]]
    if missing_ca:
        print("Attempting to repair residues missing CA atoms...")
        for res in missing_ca:
            print(f"Residue {res[0]} missing atoms: {', '.join(res[1])}", file=sys.stderr)
        return False
    return True


def extract_pocket(input_pdb, output_pdb, output_ss_file, radius=15):
    parser = PDBParser()
    structure = parser.get_structure('complex', input_pdb)

    try:
        center = get_ligand_center(structure)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    het_residues = get_all_hetatm_residues(structure)

    try:
        model = structure[0]
        dssp = DSSP(model, input_pdb)
    except Exception as e:
        print(f"Error: DSSP computation failed: {str(e)}", file=sys.stderr)
        return 1

    keep_residues = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith('H_'):
                    continue
                for atom in residue:
                    if (atom.get_vector() - center).norm() <= radius:
                        keep_residues.add((chain.id, residue.id))
                        break

    class PocketSelect(Select):
        def __init__(self, het_residues):
            self.het_residues = het_residues

        def accept_residue(self, residue):
            if residue.id[0].startswith('H_'):
                return False
            return (residue.parent.id, residue.id) in keep_residues

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, PocketSelect(het_residues))

    with open(output_ss_file, 'w') as f:
        for chain in model:
            for residue in chain:
                if (chain.id, residue.id) in keep_residues and is_aa(residue, standard=True):
                    try:
                        key = (chain.id, residue.id)
                        ss = dssp[key][2]
                        res_info = f"{chain.id}/{residue.id[1]} ({residue.resname}): {ss}"
                        f.write(res_info + '\n')
                    except KeyError:
                        continue

    if not validate_pocket(output_pdb):
        print(f"Error: Pocket file {output_pdb} contains incomplete residues.", file=sys.stderr)
        os.remove(output_pdb)
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input.pdb output.pdb output_ss.txt")
        sys.exit(1)

    exit_code = extract_pocket(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(exit_code)
