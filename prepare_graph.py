import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors, GetPeriodicTable, rdmolops, rdchem
from rdkit.Chem.rdmolops import SanitizeFlags
import os
from tqdm import tqdm
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial.distance import cdist
from rdkit.Chem import rdMolDescriptors as rdMD
import warnings
import shutil

warnings.filterwarnings("ignore", category=BiopythonWarning, module='Bio.PDB')
warnings.filterwarnings("ignore", category=UserWarning, module='Bio.PDB.DSSP')
warnings.filterwarnings("ignore", message="'num_faces' is deprecated")

# ====================
PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 
                  'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 
                  'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 
                  'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

PERMITTED_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

STEREO_TYPES = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]

USE_CHIRALITY = True  
USE_STEREO = True     
INTERACTION_CUTOFF = 5.0  
SPATIAL_CUTOFF = 8.0  

# ====================
# Residual Symbol Mapping Table
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G',
           'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
           'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V',
           'TRP':'W', 'TYR': 'Y'}
res_table = list(ressymbl.values())

dssp_table = ["H", "B", "E", "G", "I", "T", "S", "-"]  # DSSP Encoding

pcp_dict = {'A':[ 0.62014, -0.18875, -1.23870, -0.083627,-1.32960, -1.38170, -0.44118],
            'C':[ 0.29007, -0.44041, -0.76847, -1.05000, -0.48930, -0.77494, -1.11480],
            'D':[-0.90020,  1.57290, -0.89497,  1.73760, -0.72498, -0.50189, -0.91814],
            'E':[-0.74017,  1.57290, -0.28998,  1.47740, -0.25361,  0.094051,-0.44710],
            'F':[ 1.19030, -1.19540,  1.18120, -1.16150,  1.17070,  0.88720,  0.02584],
            'G':[ 0.48011,  0.062916,-1.99490,  0.25088, -1.80090, -2.03180,  2.20220],
            'H':[-0.40009, -0.18875,  0.17751,  0.77123,  0.55590,  0.44728, -0.71617],
            'I':[ 1.38030, -0.84308,  0.57625, -1.16150,  0.10503, -0.018637,-0.21903],
            'K':[-1.50030,  1.57290,  0.75499,  1.10570,  0.44318,  0.95221, -0.27937],
            'L':[ 1.06020, -0.84308,  0.57625, -1.27300,  0.10503,  0.24358,  0.24301],
            'M':[ 0.64014, -0.59141,  0.59275, -0.97565,  0.46368,  0.46679, -0.51046],
            'N':[-0.78018,  1.06960, -0.38073,  1.21720, -0.42781, -0.35453, -0.46879],
            'P':[ 0.12003,  0.062916,-0.84272, -0.12080, -0.45855, -0.75977,  3.13230],
            'Q':[-0.85019,  0.16358,  0.22426,  0.80840,  0.04355,  0.24575,  0.20516],
            'R':[-2.53060,  1.57290,  0.89249,  0.80840,  1.18100,  1.60670,  0.11866],
            'S':[-0.18004,  0.21392, -1.18920,  0.32522, -1.16560, -1.12820, -0.48056],
            'T':[-0.050011,-0.13842, -0.58422,  0.10221, -0.69424, -0.63625, -0.50017],
            'V':[ 1.08020, -0.69208, -0.028737,-0.90132, -0.36633, -0.37620,  0.32502],
            'W':[ 0.81018, -1.64840,  2.00620, -1.08720,  2.39010,  1.82990,  0.032377],
            'Y':[ 0.26006, -1.09470,  1.23070, -0.78981,  1.25270,  1.19060, -0.18876]}

# From Kyte AND Doolittle:
hydrophobicity0 = {"A":1.80,"R":-4.5,"N":-3.50,"D":-3.5,"C":2.50,"Q":-3.50,"E":-3.50,"G":-0.40,"H":-3.2,"I":4.50,"L":3.8,"K":-3.9,"M":1.90,"F":2.8,"P":-1.60,"S":-0.80,"T":-0.70,"W":-0.90,"Y":-1.30,"V":4.2}
ATOM_FEAT_DIM = 74    
     
lig_properties_error_ids = []  
H_error_ids = []         
miss_data = []

# ========== Feature Functions ==========
def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]  
    return np.array([int(x == s) for s in permitted_list], dtype=np.float32)

def get_atom_features(atom, record_type_map, protein_valid_residues=None, protein_ss_onehot=None):
    # 1. atom_type（44）
    atom_type = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOMS)
    
    # 2. chirality（4）
    atom_idx = atom.GetIdx()
    is_ligand = record_type_map.get(atom_idx, "HETATM") == "HETATM"
    if USE_CHIRALITY:
        if is_ligand:
            chiral_tag = str(atom.GetChiralTag())
            chirality_enc = one_hot_encoding(chiral_tag, ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        else:
            chirality_enc = [1, 0, 0, 0]  
    else:
        chirality_enc = []

    # 3. heavy_neighbors（6）
    heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != 'H')
    n_heavy_enc = one_hot_encoding(
        heavy_neighbors if heavy_neighbors <= 4 else "MoreThanFour",
        [0, 1, 2, 3, 4, "MoreThanFour"]
    )
    
    # 4.charge（8）
    charge = atom.GetFormalCharge()
    charge_enc = one_hot_encoding(
        charge if -3 <= charge <= 3 else "Extreme",
        [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    )
    
    # 5. hybrid（7）
    hybrid = str(atom.GetHybridization()).upper()
    hybrid_enc = one_hot_encoding(
        hybrid,
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    )
    
    # 6. ring and aromatic(2)
    in_ring = [int(atom.IsInRing())]
    is_aromatic = [int(atom.GetIsAromatic())]
    
    # 7. Physical properties（3）
    atomic_mass = atom.GetMass()
    vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
    covalent_radius = Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())

    features = np.concatenate([
        atom_type, chirality_enc, n_heavy_enc, charge_enc, hybrid_enc,
        in_ring, is_aromatic,
        [atomic_mass], [vdw_radius], [covalent_radius]
    ])    # total：44+4+6+8+7+2+3=74
    return features

def get_bond_features(bond, atom_i, atom_j, record_type_map):
    # 1.  bond_type（4）
    bond_type_enc = one_hot_encoding(bond.GetBondType(), PERMITTED_BOND_TYPES)
    
    # 2. Conjugation and ring（2）
    conj_enc = [int(bond.GetIsConjugated())]
    ring_enc = [int(bond.IsInRing())]
    
    # 3. Stereochemistry（4）
    if USE_STEREO:
        stereo = str(bond.GetStereo()).upper()
        stereo_enc = one_hot_encoding(stereo, STEREO_TYPES)
    else:
        stereo_enc = []
        
    features = np.concatenate([bond_type_enc, conj_enc, ring_enc, stereo_enc])
    return features

# ========== Interaction ==========
interface_edge_types = [
    'cross_salt_bridge',  # 0（salt bridges）
    'cross_hbond',        # 1（hydrogen bonds）
    'cross_pi_stack',     # 2 （π-π stacking）
    'cross_hydrophobic',  # 3（hydrophobic interactions）
    'cross_vdw'           # 4（van der Waals interactions）
]
num_interface_edge_types = len(interface_edge_types)

pt = GetPeriodicTable()
def is_vdw(atom_i, atom_j, distance):
    vdw_i = pt.GetRvdw(atom_i.GetAtomicNum())
    vdw_j = pt.GetRvdw(atom_j.GetAtomicNum())
    sum_vdw = vdw_i + vdw_j
    min_vdw = 0.8 * sum_vdw
    max_vdw = 1.2 * sum_vdw

    return min_vdw <= distance <= max_vdw   

def is_salt_bridge(mol, i, j, distance, record_type_map):
    if not (2.5 <= distance <= 4):
        return False
    
    if record_type_map[i] == record_type_map[j]:
        return False

    atom_i = mol.GetAtomWithIdx(i)
    atom_j = mol.GetAtomWithIdx(j)
    charge_i = atom_i.GetFormalCharge()
    charge_j = atom_j.GetFormalCharge()
    
    if charge_i * charge_j >= 0:  
        return False
    if charge_i > 0:
        pos_atom = atom_i
        neg_atom = atom_j
    else:
        pos_atom = atom_j
        neg_atom = atom_i

    def is_positive_atom(atom):
        symbol = atom.GetSymbol()
        
        if symbol == 'N':
            neighbors = atom.GetNeighbors()
            carbon_count = sum(1 for n in neighbors if n.GetSymbol() == 'C')
            if carbon_count >= 3:  
                return True
            if carbon_count >= 1 and charge_i > 0:
                return True
        
        if symbol == 'C':
            n_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'N')
            if n_count == 3:
                return True
        
        if symbol == 'N' and atom.GetIsAromatic():
            for ring in mol.GetRingInfo().AtomRings():
                if atom.GetIdx() in ring and len(ring) == 5:
                    return True
        
        if symbol in ['Na', 'K', 'Ca', 'Mg', 'Zn', 'Fe', 'Mn', 'Cu']:
            return True
        
        return False
    
    def is_negative_atom(atom):
        symbol = atom.GetSymbol()
        
        if symbol == 'O':
            carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
            for c_atom in carbon_neighbors:
                o_neighbors = [n for n in c_atom.GetNeighbors() if n.GetSymbol() == 'O']
                if len(o_neighbors) >= 2:  
                    return True
        
        if symbol == 'O' or symbol == 'P':
            p_atom = atom
            if symbol == 'O':
                p_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'P']
                if not p_neighbors:
                    return False
                p_atom = p_neighbors[0]
            
            o_count = sum(1 for n in p_atom.GetNeighbors() if n.GetSymbol() == 'O')
            if o_count >= 3 and charge_j < 0:
                return True
        
        if symbol == 'O' or symbol == 'S':
            s_atom = atom
            if symbol == 'O':
                s_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'S']
                if not s_neighbors:
                    return False
                s_atom = s_neighbors[0]
            
            o_count = sum(1 for n in s_atom.GetNeighbors() if n.GetSymbol() == 'O')
            if o_count >= 3 and charge_j < 0:
                return True
        
        return False 

    return is_positive_atom(pos_atom) and is_negative_atom(neg_atom)

def is_hbond(mol, i, j, distance, record_type_map):
    atom_i = mol.GetAtomWithIdx(i)
    atom_j = mol.GetAtomWithIdx(j)
    
    if record_type_map[i] == record_type_map[j]:
        return False

    donor_atom, acceptor_atom, h_atom  = None, None, None
    if atom_i.GetSymbol() == 'H':
        h_atom = atom_i
        donors = [n for n in atom_i.GetNeighbors() if n.GetSymbol() != 'H']
        if len(donors) != 1:
            return False
        donor_atom = donors[0]
        acceptor_atom = atom_j

    elif atom_j.GetSymbol() == 'H':
        h_atom = atom_j
        acceptors = [n for n in atom_j.GetNeighbors() if n.GetSymbol() != 'H']
        if len(acceptors) != 1:
            return False
        donor_atom = acceptors[0]
        acceptor_atom = atom_i
    else:
        return False

    donor_type = donor_atom.GetSymbol()
    acceptor_type = acceptor_atom.GetSymbol()
    if donor_type not in ['O', 'N', 'F']:
        return False
    if acceptor_type not in ['O', 'N', 'S', 'F', 'Cl']:
        return False
    
    if not (2.2 <= distance <= 4.0):
        return False
    
    h_coord = np.array(mol.GetConformer().GetAtomPosition(h_atom.GetIdx()))
    donor_coord = np.array(mol.GetConformer().GetAtomPosition(donor_atom.GetIdx()))
    acceptor_coord = np.array(mol.GetConformer().GetAtomPosition(acceptor_atom.GetIdx()))
   
    def is_valid_coord(coord):
        return np.linalg.norm(coord) > 1e-6  
    if not (is_valid_coord(h_coord) and is_valid_coord(donor_coord) and is_valid_coord(acceptor_coord)):
        return False

    vec_dh = h_coord - donor_coord
    vec_ha = acceptor_coord - h_coord
    angle = np.degrees(np.arccos(np.clip(
        np.dot(vec_dh, vec_ha) / (np.linalg.norm(vec_dh)*np.linalg.norm(vec_ha)),-1,1
    )))
    
    return angle >= 120

def is_electrostatic(mol, i, j, distance):
    atom_i = mol.GetAtomWithIdx(i)
    atom_j = mol.GetAtomWithIdx(j)
    charge_i = atom_i.GetFormalCharge()
    charge_j = atom_j.GetFormalCharge()

    if not (2.5 <= distance <= 7.0):
        return False
    
    if charge_i != 0 and charge_j != 0:
        if charge_i * charge_j < 0 and 2.5 <= distance <= 4.0:
            return True
    
    dipole_i = get_atom_dipole_direction(atom_i)
    dipole_j = get_atom_dipole_direction(atom_j)
    dipole_norm_i = np.linalg.norm(dipole_i)
    dipole_norm_j = np.linalg.norm(dipole_j)

    if dipole_norm_i > 0 and dipole_norm_j > 0:
        cos_theta = np.dot(dipole_i, dipole_j)/(dipole_norm_i * dipole_norm_j)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
        if 3.5 <= distance <= 7.0 and (angle < 30 or angle > 150):
            return True
    
    if (abs(charge_i) > 0 and dipole_norm_j > 0.5) or (abs(charge_j) > 0 and dipole_norm_i > 0.5):
        if 2.5 <= distance <= 6.0:
            return True
    
    return False

def get_atom_dipole_direction(atom):
    from scipy.spatial import Delaunay
    from rdkit import Geometry
    
    mol = atom.GetOwningMol()
    conf = mol.GetConformer()
    atom_idx = atom.GetIdx()
    center = np.array(conf.GetAtomPosition(atom_idx))
    
    neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
    if not neighbors:
        return np.array([0,0,1])  
    
    neighbor_coords = [np.array(conf.GetAtomPosition(n.GetIdx())) for n in neighbors]
    
    hybrid = atom.GetHybridization()
    
    if hybrid == Chem.HybridizationType.SP3:
        if len(neighbors) >= 3:
            centroid = np.mean(neighbor_coords, axis=0)
            direction = center - centroid
        else:
            direction = np.array([0,0,1])
    
    elif hybrid == Chem.HybridizationType.SP2:
        if len(neighbors) >= 2:
            v1 = neighbor_coords[0] - center
            v2 = neighbor_coords[1] - center
            direction = np.cross(v1, v2)
        else:
            direction = np.array([0,0,1])
    
    elif hybrid == Chem.HybridizationType.SP:
        direction = center - neighbor_coords[0]
    
    else:
        try:
            points = np.vstack([center, neighbor_coords])
            tri = Delaunay(points)
            normals = []
            for simplex in tri.simplices:
                vec1 = points[simplex[1]] - points[simplex[0]]
                vec2 = points[simplex[2]] - points[simplex[0]]
                normal = np.cross(vec1, vec2)
                if np.linalg.norm(normal) > 1e-6:
                    normals.append(normal)
            direction = np.mean(normals, axis=0) if normals else np.array([0,0,1])
        except:
            direction = np.array([0,0,1])
    
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else np.array([0,0,1])

def is_pi_stack(mol, i, j, distance):
    atom_i = mol.GetAtomWithIdx(i)
    atom_j = mol.GetAtomWithIdx(j)
    
    if not (atom_i.GetIsAromatic() and atom_j.GetIsAromatic()):
        return False
    
    def get_aromatic_system(atom):
        mol = atom.GetOwningMol()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if atom.GetIdx() in ring:
                aromatic_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
                if all(a.GetIsAromatic() for a in aromatic_atoms):
                    return aromatic_atoms  
        return None

    ring_i = get_aromatic_system(atom_i)
    ring_j = get_aromatic_system(atom_j)
    
    if not (ring_i and ring_j):
        return False

    def calc_ring_properties(ring_atoms, conf):
        coords = np.array([conf.GetAtomPosition(a.GetIdx()) for a in ring_atoms])
        centroid = np.mean(coords, axis=0)
        cov_matrix = np.cov(coords - centroid, rowvar=False)
        _, _, vh = np.linalg.svd(cov_matrix)
        return centroid, vh[2]  

    conf = mol.GetConformer()
    centroid_i, normal_i = calc_ring_properties(ring_i, conf)
    centroid_j, normal_j = calc_ring_properties(ring_j, conf)

    distance = np.linalg.norm(centroid_i - centroid_j)

    cos_angle = np.abs(np.dot(normal_i, normal_j))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  
    angle = np.degrees(np.arccos(cos_angle))
    
    centroid_vector = centroid_j - centroid_i
    normal_projection = np.dot(centroid_vector, normal_i) * normal_i
    offset = np.linalg.norm(centroid_vector - normal_projection)

    parallel = (3.4 <= distance <= 4.5) and (angle < 30) and (offset < 1.5)
    perpendicular = (4.0 <= distance <= 5.0) and (60 <= angle <= 120)

    return parallel or perpendicular

def is_hydrophobic(lig_atom, prot_atom, distance):
    def is_ligand_nonpolar(atom):
        symbol = atom.GetSymbol()
        if symbol == 'C':
            hybrid = atom.GetHybridization()
            if hybrid == Chem.HybridizationType.SP3:
                polar_neighbors = [n for n in atom.GetNeighbors()
                                if n.GetSymbol() in ['O', 'N', 'S', 'F']]
                if not polar_neighbors:
                    return True
            
            elif hybrid == Chem.HybridizationType.SP2 and atom.GetIsAromatic():
                polar_substituents = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() in ['O', 'N', 'S', 'F']:
                        polar_substituents = True
                        break
                if not polar_substituents:
                    return True

        elif symbol == 'S':
            if not any(n.GetSymbol() == 'O' for n in atom.GetNeighbors()):
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'C':
                        return True
                return True
        
        elif symbol == 'F':
            if sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'C') >= 3:
                return True  
        return False

    def is_protein_nonpolar(atom):
        res_info = atom.GetPDBResidueInfo()
        if res_info is None:
            return False

        res_name = res_info.GetResidueName()
        res_code = ressymbl.get(res_name, 'X')
        hydrophobic_residues = ['A', 'V', 'L', 'I', 'F', 'W', 'M', 'P', 'Y']
        
        if res_code in hydrophobic_residues:
            if res_code in ['F', 'W', 'Y'] and atom.GetSymbol() == 'C':
                return True
            elif atom.GetSymbol() == 'C':
                polar_neighbors = [n for n in atom.GetNeighbors()
                                if n.GetSymbol() in ['O', 'N', 'S', 'F']]
                return len(polar_neighbors) == 0
    
    if (is_ligand_nonpolar(lig_atom) and 
        is_protein_nonpolar(prot_atom) and
        3.0 <= distance <= 5.0):

        res_name = prot_atom.GetPDBResidueInfo().GetResidueName()
        res_code = ressymbl.get(res_name, 'X')
        res_hydrophobicity = hydrophobicity0.get(res_code, 0.0)
        if res_hydrophobicity >= 1.0:
            return True
        if res_code in ['A', 'V', 'L', 'I', 'F', 'W', 'M', 'P', 'Y']:
            return True

    return False

def get_interface_edge_features(edge_type_idx, edge_category, distance):
    interaction_types = {
        'atom': [ 'cross_salt_bridge', 'cross_hbond', 'cross_pi_stack', 'cross_hydrophobic', 'cross_vdw']
    }
    interaction_name = interaction_types[edge_category][edge_type_idx]
    standard_type_map = {
        'cross_salt_bridge': 0,
        'cross_hbond': 1,
        'cross_pi_stack': 2,
        'cross_hydrophobic': 3,
        'cross_vdw': 4,
    }
    standard_type_idx = standard_type_map.get(interaction_name, None)
    permitted_list = interaction_types[edge_category]
    if edge_type_idx < 0 or edge_type_idx >= len(permitted_list):
        edge_type_idx = len(permitted_list) - 1  
    type_enc = np.zeros(len(permitted_list), dtype=np.float32)
    type_enc[edge_type_idx] = 1.0

    strength_ranges = {
        0: [(2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],    # salt_bridge  
        1: [(2.2, 2.5), (2.5, 3.2), (3.2, 4.0)],    # hbond
        2: [(3.4, 4.0), (4.0, 4.5), (4.5, 5.0)],    # pi-stack
        3: [(3.0, 3.7), (3.7, 4.4), (4.4, 5.0)],     # hydrophobic
        4: [(1.0, 2.5), (2.5, 3.5), (3.5, 5.0)],    # vdw
    }
    
    strength = 0
    if standard_type_idx in strength_ranges:
        ranges = strength_ranges[standard_type_idx]
        for level, (low, high) in enumerate(ranges):
            if low <= distance < high:
                strength = level
                break
        if strength is None and distance == ranges[-1][1]:
            strength = len(ranges) - 1
    
    if strength is not None:
        strength_enc = one_hot_encoding(strength, [0, 1, 2])
    else:
        pass

    feat = np.concatenate([type_enc, strength_enc, [distance], [0.0]], dtype=np.float32 )  
    if len(feat) < 10:
        feat = np.pad(feat, (0, 24 - len(feat)), 'constant')
    return feat

# ==========protein edge types ==========
protein_edge_types = [
    'chemical_bond',      
    'sequence_adjacent',  
    'spatial_proximity',  
    'secondary_structure',
    'disulfide' 
]
num_protein_edge_types = len(protein_edge_types)

def get_protein_edge_features(edge_type_idx, distance):
    encoding = [0.0] * num_protein_edge_types
    encoding[edge_type_idx] = 1.0
    return np.concatenate([encoding, [distance], [0.0]*4], dtype=np.float32)  

class ProteinFeatureExtractor:
    def __init__(self, protein_file, complex_id, ss2_file):
        self.protein_file = protein_file
        self.complex_id = complex_id
        self.parser = PDBParser()
        self.structure = self.parser.get_structure('protein', protein_file)
        self.model = self.structure[0]
        self.residues = []
        self.ca_coords, self.sequence = self.parse_structure()
        self.ss_onehot, self.ss_seq = self.get_dssp_features(ss2_file)
    
    def parse_structure(self):
        ca_coords = []
        sequence = []
        self.valid_residues = []
        self.residues = []  

        for chain in self.model:
            chain_id = chain.id
            for residue in chain:
                res_id = residue.get_id()
                resnum = res_id[1]
                resname = residue.get_resname().strip()

                if resname not in ressymbl:
                    continue

                ca = None
                backup_atoms = ['CA', 'N', 'C', 'O']
                for atom_name in backup_atoms:
                    if residue.has_id(atom_name):
                        ca = residue[atom_name]
                        if atom_name != 'CA':
                            print(f"Residue {resname}{res_id[1]} uses backup atom {atom_name}")
                        break

                if ca is None:
                    print(f"Residue {resname}{res_id[1]} is missing backbone atoms and has been skipped")
                    continue
                if ca is not None:
                    self.residues.append(residue) 
                    ca_coords.append(ca.get_coord())
                    sequence.append(ressymbl[resname])
                    self.valid_residues.append((chain_id, resnum))
        
        if len(sequence) ==0 :
            print(f"Warning: protein file {self.protein_file} contains no valid residues")
        
        return np.array(ca_coords), ''.join(sequence)

    def get_dssp_features(self, ss2_file):
        ss_seq = []
        ss_onehot = np.zeros((len(self.sequence), DSSP_FEAT_DIM))
        residue_index_map = {res_id: idx for idx, res_id in enumerate(self.valid_residues)}
        try:
            with open(ss2_file, 'r') as f:  
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) < 2:
                        print(f"Format Error Row: {line.strip()}")
                        continue
                    res_info, ss = parts[0].strip(), parts[1].strip()
                    chain_id, res_num = res_info.split('/')
                    res_num = int(res_num.split(' ')[0])
                    res_id = (chain_id, res_num)
                    ss_seq.append(ss)
                    if res_id in residue_index_map:
                        idx = residue_index_map[res_id]
                        ss_onehot[idx] = one_hot_encoding(ss, dssp_table)
                    else:
                        print(f"No residual information found in the sequence: {res_info}")

        except Exception as e:
            print(f"Failure to read ss2 file: {str(e)}")
            ss_seq = ['C'] * len(self.sequence)
            ss_onehot = np.zeros((len(self.sequence), DSSP_FEAT_DIM))

        return ss_onehot, ss_seq

    def get_ss_edges(self, ss):
        edges = []
        edge_attrs = []
        n = len(ss)
        for i in range(n-1):
            if str(ss[i]).strip() == 'E' and str(ss[i+1]).strip() == 'E':
                edges.extend([(i, i+1), (i+1, i)])
                distance = np.linalg.norm(self.ca_coords[i] - self.ca_coords[i+1])
                features = get_protein_edge_features(3, distance)
                edge_attrs.extend([features, features])
        return edges, edge_attrs
    
    def get_disulfide_edges(self):
        edges = []
        edge_attrs = []
        cys_indices = [i for i, res in enumerate(self.sequence) if res == 'C']
        for i in cys_indices:
            for j in cys_indices:
                if i != j and np.linalg.norm(self.ca_coords[i]-self.ca_coords[j]) < 5.0:
                    edges.extend([(i,j), (j,i)])
                    distance = np.linalg.norm(self.ca_coords[i]-self.ca_coords[j])
                    features = get_protein_edge_features(4, distance)
                    edge_attrs.extend([features, features])
        return edges, edge_attrs
    
    def get_spatial_edges(self, cutoff=SPATIAL_CUTOFF):
        edges = []
        edge_attrs = []
        dist_matrix = cdist(self.ca_coords, self.ca_coords)
        n = len(self.ca_coords)
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i,j] <= cutoff:
                    edges.extend([(i,j), (j,i)])
                    distance = dist_matrix[i,j]
                    features = get_protein_edge_features(2, distance)
                    edge_attrs.extend([features, features])
        return edges, edge_attrs
    
    def get_all_edges(self, ss):
        all_edges = []
        all_attrs = []
        
        seq_edges = [(i,i+1) for i in range(len(self.sequence)-1)]
        for i,j in seq_edges:
            distance = np.linalg.norm(self.ca_coords[i] - self.ca_coords[j])
            features = get_protein_edge_features(1, distance) 
            all_edges.extend([(i,j), (j,i)])
            all_attrs.extend([features, features])

        edge_methods = [
            (self.get_spatial_edges, 2),   
            (self.get_ss_edges, 3),        
            (self.get_disulfide_edges, 4)  
        ]
        
        for method, edge_type in edge_methods:
            try:
                if edge_type == 3:
                    edges, attrs = method(ss)  
                else:
                    edges, attrs = method()

                if not (isinstance(edges, list) and isinstance(attrs, list)):
                    raise TypeError("Edge generation function must return two lists")

                all_edges.extend(edges)
                all_attrs.extend(attrs)
            except Exception as e:
                print(f"Failed to generate edge type {edge_type}: {str(e)}")
                continue
        
        return all_edges, all_attrs

#=============================
def get_residue_features(sequence, ss_onehot):
    if len(sequence) != ss_onehot.shape[0]:
        raise ValueError(f"Feature dimension mismatch: sequence length ({len(sequence)}) != number of SS features ({ss_onehot.shape[0]})")

    seq_onehot = np.zeros((len(sequence), len(res_table)))
    for i, res in enumerate(sequence):
        if res in res_table:
            seq_onehot[i, res_table.index(res)] = 1
    
    physchem = []
    for res in sequence:
        hydro0 = hydrophobicity0.get(res, 0.0)
        charge = charge_dict.get(res, 0.0)
        pcp = pcp_dict.get(res, [0.0]*7)
        physchem.append([hydro0, charge] + pcp)
    physchem = np.array(physchem)

    pos_enc = np.zeros((len(sequence), 4))
    for i in range(len(sequence)):
        pos_enc[i] = [np.sin(i / 10000**(2*j/4)) for j in range(4)]

    return np.hstack([seq_onehot, ss_onehot, physchem, pos_enc]).astype(np.float32)

def get_protein_features(protein_file, complex_id, ss2_file):
    try:
        extractor = ProteinFeatureExtractor(protein_file, complex_id, ss2_file)
        if not extractor.res_features:
            raise ValueError("No residue features were extracted")
        if len(extractor.sequence) == 0:
            raise ValueError("Empty protein structure")
        
        try:
            ss_onehot, ss_seq = extractor.get_dssp_features(ss2_file)
            if len(ss_seq) != len(extractor.sequence):
                print(f"Warning: DSSP length ({len(ss_seq)}) does not match sequence length ({len(extractor.sequence)}) ")
        except Exception as dssp_error:
            print(f"Failed to compute DSSP features: {str(dssp_error)}")
            ss_onehot = np.zeros((len(extractor.sequence), DSSP_FEAT_DIM))
            ss_seq = ['C'] * len(extractor.sequence)
        
        print(f'[DEBUG] get_protein_features: valid_residues sample={extractor.valid_residues[:5] if hasattr(extractor, "valid_residues") else None}, ss_onehot shape={ss_onehot.shape}')

        residue_features = get_residue_features(extractor.sequence, ss_onehot)
        if len(residue_features) == 0:
            raise ValueError("Empty residue features")

        edges, edge_attrs = extractor.get_all_edges(ss_seq)
        edges = np.array(edges)  
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        #edge_attrs_np = np.array(edge_attrs, dtype=np.float32)
        edge_attr = torch.from_numpy(np.array(edge_attrs, dtype=np.float32))

        protein_data = Data(
            x=torch.tensor(residue_features, dtype=torch.float),
            edge_index=edges,
            edge_attr=edge_attr.clone().detach(),  
            ca_coords=torch.tensor(extractor.ca_coords, dtype=torch.float),
            sequence=extractor.sequence,
            structure=extractor.structure
        )
        protein_data.valid_residues = extractor.valid_residues
        protein_data.ss_onehot = ss_onehot

        return protein_data
    except Exception as e:
        print(f"Protein processing failed: {str(e)}")
        return None

# ========== ligand properties ==========
def calculate_ligand_properties(ligand_mol, complex_id):
    try:
        mol_copy = Chem.Mol(ligand_mol)
        rdmolops.SanitizeMol(mol_copy, sanitizeOps=SanitizeFlags.SANITIZE_SETAROMATICITY)
        return {
            'logP': Descriptors.MolLogP(mol_copy),
            'TPSA': Descriptors.TPSA(mol_copy),
            'HDonors': rdMD.CalcNumHBD(mol_copy),
            'HAcceptors': rdMD.CalcNumHBA(mol_copy),
            'rotatable_bonds': rdMD.CalcNumRotatableBonds(mol_copy),
            'QED': QED.qed(mol_copy),
            'MW': Descriptors.MolWt(mol_copy),
            'Ro5_violations': sum([
                int(Descriptors.MolWt(mol_copy) > 500),
                int(Descriptors.MolLogP(mol_copy) > 5),
                int(rdMD.CalcNumHBD(mol_copy) > 5),
                int(rdMD.CalcNumHBA(mol_copy) > 10),
                int(rdMD.CalcNumRotatableBonds(mol_copy) > 10)
            ])
        }
    except Exception as e:
        print(f"[{complex_id}] Ligand property calculation failed: {str(e)}")
        lig_properties_error_ids.append(complex_id)  # record error ID
        return None

#========atom graph==========
def build_atom_graph(mol, ligand_mol, label, record_type_map, complex_id, ligand_props=None, protein_valid_residues=None, protein_ss_onehot=None):
    if mol is None or ligand_mol is None:
        return None
    conf = mol.GetConformer()
    if conf is None:
        print("Invalid molecular conformer")
        return None

    n_atoms = mol.GetNumAtoms()
    atom_coords = np.array([[conf.GetAtomPosition(i).x, 
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] 
                          for i in range(n_atoms)])

    x = []
    ligand_mask = []
    for atom in mol.GetAtoms():
        features = get_atom_features(
            atom, record_type_map,
            protein_valid_residues=protein_valid_residues,
            protein_ss_onehot=protein_ss_onehot
        )
        x.append(features)
        is_ligand = record_type_map[atom.GetIdx()] == "HETATM"
        ligand_mask.append(is_ligand)
    
    x = np.array(x, dtype=np.float32)
    ligand_mask = torch.tensor(ligand_mask, dtype=torch.bool)

    edges = []
    edge_attrs = []
    
    # 1. bond
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        features = get_bond_features(bond, atom_i, atom_j, record_type_map)
        edges.extend([(i, j), (j, i)])
        edge_attrs.extend([features, features])

    # 2. non-bonding interactions
    from scipy.spatial import KDTree
    coords = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(n_atoms)])
    ligand_indices = [i for i in range(n_atoms) if record_type_map[i] == "HETATM"]
    protein_indices = [i for i in range(n_atoms) if record_type_map[i] == "ATOM"]
    kd_tree = KDTree(coords)
    
    cross_pairs = []
    for lig_idx in ligand_indices:
        neighbors = kd_tree.query_ball_point(coords[lig_idx], INTERACTION_CUTOFF)
        for prot_idx in neighbors:
            if prot_idx in protein_indices:
                cross_pairs.append((lig_idx, prot_idx))

    for i, j in cross_pairs:
        if mol.GetBondBetweenAtoms(i, j) is not None:
            continue
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        distance = np.linalg.norm(coords[i] - coords[j])

        is_ligand_i = record_type_map[i] == "HETATM"
        is_ligand_j = record_type_map[j] == "HETATM"

        if is_ligand_i and not is_ligand_j:
            lig_atom, prot_atom = atom_i, atom_j
        elif is_ligand_j and not is_ligand_i:
            lig_atom, prot_atom = atom_j, atom_i
        else:
            continue

        if is_salt_bridge(mol, i, j, distance, record_type_map):
            edge_type = 0
        elif is_hbond(mol, i, j, distance, record_type_map):
            edge_type = 1
        elif is_pi_stack(mol, i, j, distance):
            edge_type = 2
        elif is_hydrophobic(lig_atom, prot_atom, distance):
            edge_type = 3
        elif is_vdw(atom_i, atom_j, distance):
            edge_type = 4
        else:
            edge_type = None

        if edge_type is not None:
            features = get_interface_edge_features(edge_type, 'atom', distance)
            edges.extend([(i, j), (j, i)])
            edge_attrs.extend([features, features])

    if not edges:
        return None
    edge_attrs_np = np.array(edge_attrs, dtype=np.float32)
    edge_attr = torch.tensor(edge_attrs_np, dtype=torch.float)

    global_features = [
        ligand_props['logP'], ligand_props['TPSA'],
        ligand_props['HDonors'], ligand_props['HAcceptors'],
        ligand_props['rotatable_bonds'], ligand_props['QED'],
        ligand_props['MW'], ligand_props['Ro5_violations']
    ]
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=edge_attr,
        pos=torch.tensor(atom_coords, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.float),
        global_features=torch.tensor(global_features),
        ligand_mask=ligand_mask,
        mol=mol
    )

# ========== Main processing pipeline ==========
def process_complex(complex_id, label, complex_dir, output_dir, ss2_dir, debug=True):
    complex_file = os.path.join(complex_dir, f"{complex_id}_pocket_ligand.pdb")
    ss2_file = os.path.join(ss2_dir, f"{complex_id}.ss2")
    
    print(f"Processing complex: {complex_id}")
    if not os.path.exists(complex_file):
        print(f"Complex file {complex_file} does not exist, skipping this complex")
        miss_data.append(complex_id)
        return None
    if not os.path.exists(ss2_file):
        print(f"ss2 file {ss2_file} does not exist, skipping this complex")
        miss_data.append(complex_id)
        return None

    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"[{complex_id}] Starting protein–ligand separation")
    protein_file = os.path.join(temp_dir, f"{complex_id}_protein.pdb")
    ligand_file = os.path.join(temp_dir, f"{complex_id}_ligand.pdb")
    
    with open(complex_file, 'r') as f:
        lines = f.readlines()

    with open(protein_file, 'w') as f_prot:
        for line in lines:
            if line.startswith("ATOM"):
                f_prot.write(line)
    
    lig_lines = []
    for line in lines:
        if line.startswith("HETATM"):
            res_name = line[17:20].strip().upper()
            if res_name == "LIG":
                lig_lines.append(line)
    
    if not lig_lines:
        print(f"Complex {complex_id} does not contain a ligand")
        return None
    
    with open(ligand_file, 'w') as f_lig:
        f_lig.writelines(lig_lines)

    print(f"[{complex_id}] Reading molecules ")
    ligand_mol = Chem.MolFromPDBFile(ligand_file, sanitize=False, removeHs=False)
    protein_mol = Chem.MolFromPDBFile(protein_file, sanitize=False, removeHs=False)
    print(f"配体原子数: {ligand_mol.GetNumAtoms() if ligand_mol else 0}")
    ligand_props = calculate_ligand_properties(ligand_mol, complex_id)
    if ligand_props is None:
        print(f"[{complex_id}] Ligand property calculation failed, skipping complex ")
        return None

    print(f"[{complex_id}] Starting hydrogen processing ")
    ligand_mol = process_hydrogens(ligand_mol, complex_id, is_ligand=True) 
    protein_mol = process_hydrogens(protein_mol, complex_id, is_ligand=False) 
    print(f"Ligand atom count after adding hydrogens: {ligand_mol.GetNumAtoms() if ligand_mol else 0}")

    if protein_mol is None or ligand_mol is None:
        print(f"[{complex_id}] Hydrogen processing failed for protein or ligand ")
        return None
    print(f"[{complex_id}] Merging molecules")
    mol = Chem.CombineMols(ligand_mol, protein_mol)
    if mol is None :
        print(f"Failed to read molecule file for {complex_id}")
        return None

    record_type_map = {}
    ligand_atom_count = ligand_mol.GetNumAtoms()
    for atom in ligand_mol.GetAtoms():
        record_type_map[atom.GetIdx()] = "HETATM"
    for atom in protein_mol.GetAtoms():
         record_type_map[ligand_atom_count + atom.GetIdx()] = "ATOM"

    try:
        protein_data = get_protein_features(protein_file, complex_id, ss2_file)
        if protein_data is None or protein_data.x.size(0) == 0:
            print(f"Protein data for complex {complex_id} is invalid or empty")
            return None
        if not all(hasattr(protein_data, attr) for attr in ['x', 'edge_index', 'ca_coords', 'res_features']):
            print(f"Incomplete protein data for complex {complex_id}")
            return None

        protein_valid_residues = getattr(protein_data, 'valid_residues', None)
        protein_ss_onehot = getattr(protein_data, 'ss_onehot', None)
        print(f'[DEBUG] process_complex: protein_valid_residues={protein_valid_residues[:5] if protein_valid_residues is not None else None}, protein_ss_onehot shape={protein_ss_onehot.shape if protein_ss_onehot is not None else None}')
        atom_graph = build_atom_graph(
            mol, ligand_mol, label, record_type_map, complex_id, ligand_props,
            protein_valid_residues=protein_valid_residues,
            protein_ss_onehot=protein_ss_onehot
        )
        if atom_graph is None or atom_graph.x.size(0) == 0:
            print(f"Incomplete atom graph data for complex {complex_id}")
            return None

        return atom_graph

# ========== Hydrogen processing function ==========
def process_hydrogens(mol, complex_id, is_ligand=True): 
    if mol is None:
        return None
    try:
        print(f"[{complex_id}] Original atom count ({'ligand' if is_ligand else 'protein'}): {mol.GetNumAtoms()}")
        
        if is_ligand:
            mol = Chem.RemoveHs(mol, implicitOnly=False)
            print(f"[{complex_id}] Atom count after removing all hydrogens: {mol.GetNumAtoms()}")
            
            polar_atoms = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in ['O', 'N', 'S', 'P']:
                    current_bonds = len(atom.GetBonds())
                    max_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())

                    charge = atom.GetFormalCharge()
                    if charge > 0:  
                        max_valence += abs(charge)

                    if current_bonds < max_valence:
                        polar_atoms.append(atom.GetIdx())
                        print(f"Atom {atom.GetIdx()} ({symbol}) can accept hydrogens: {current_bonds}/{max_valence} bonds ")

            print(f"[{complex_id}] Number of atoms requiring polar hydrogens: {len(polar_atoms)}")
            
            if polar_atoms:
                original_atom_indices = {a.GetIdx() for a in mol.GetAtoms()}
                temp_mol = Chem.AddHs(mol, onlyOnAtoms=polar_atoms, addCoords=True)
                added_hs = [a for a in temp_mol.GetAtoms() if a.GetAtomicNum() == 1] 
                print(f"[{complex_id}] Atom count after adding polar hydrogens: {temp_mol.GetNumAtoms()}")
                if len(added_hs) > 0:
                    mol = temp_mol
                else:
                    print(f"[{complex_id}] Warning: failed to add hydrogens, using original molecule ")
        else:
            pass

        try:
            print(f"[{complex_id}]  Starting molecule sanitization ")
            mol.UpdatePropertyCache(strict=False)
            rdmolops.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_ADJUSTHS)
        except Exception as sanitize_error:
            print(f"[{complex_id}] Bond perception failed: {str(sanitize_error)}")
            rdmolops.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_FINDRINGS | SanitizeFlags.SANITIZE_SETAROMATICITY)
        print(f"[{complex_id}] Atom count after sanitization: {mol.GetNumAtoms()}")
        return mol

    except Exception as e:
        print(f"[{complex_id}] Final hydrogen processing failed: {str(e)}")
        H_error_ids.append(complex_id)
        return None

# ========== Dataset processing ==========
def process_dataset(complex_dir, key_file, output_dir, ss2_dir):
    complex_infos = []
    with open(key_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    complex_infos.append((parts[0], float(parts[1])))
                except ValueError:
                    print(f"Invalid line: {line.strip()}")

    atom_graphs = []
    skipped_complexes = []
    processed_count = 0     

    for idx, (cid, label) in enumerate(tqdm(complex_infos, desc="Processing complexes")):
        try:
            atom_g = process_complex(cid, label, complex_dir, output_dir, ss2_dir, debug=True)
            if atom_g:
                atom_graphs.append(atom_g)
                processed_count += 1
            else:
                skipped_complexes.append(cid)
        except Exception as e:
            print(f"Critical error occurred while processing complex {cid} ")
            import traceback
            traceback.print_exc()
            skipped_complexes.append(cid)

    print(f"\n===== Processing completed =====")
    print(f"Successfully processed complexes: {processed_count}")
    print(f"Complexes with missing files: {len(miss_data)}")
    print(f"List：{miss_data}")
    print(f"Complexes with ligand property calculation failure: {len(lig_properties_error_ids)}")
    print(f"List: {lig_properties_error_ids}")
    print(f"Complexes with hydrogen processing failure: {len(H_error_ids)}")
    print(f"List: {H_error_ids}")
    print(f"Skipped during graph construction: {len(skipped_complexes)}个")
    print(f"List: {skipped_complexes}")

    # Clean up temporary directory
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    return atom_graphs

def save_graphs(graph_list, path):
    saved_data = []

    for graph in graph_list:
        data_dict = {
            'y': graph.y.detach().cpu().numpy() if hasattr(graph, 'y') else None,
            'global_features': graph.global_features.detach().cpu().numpy() if hasattr(graph, 'global_features') else None,
            'metadata': {}
        }
        
        if isinstance(graph, Data):
            data_dict['type'] = 'homogeneous'
            data_dict.update({
                'x': graph.x.detach().cpu().numpy(),
                'edge_index': graph.edge_index.detach().cpu().numpy(),
                'edge_attr': graph.edge_attr.detach().cpu().numpy(),
                'pos': graph.pos.detach().cpu().numpy() if hasattr(graph, 'pos') else None,
                'ligand_mask': graph.ligand_mask.detach().cpu().numpy() if hasattr(graph, 'ligand_mask') else None
            })
            
        data_dict['metadata'] = {
            'num_nodes': graph.num_nodes if hasattr(graph, 'num_nodes') else None,
            'num_edges': graph.num_edges if hasattr(graph, 'num_edges') else None,
            'device': str(graph.x.device) if hasattr(graph, 'x') else 'unknown'
        }
        
        saved_data.append(data_dict)
    
    np.save(path, saved_data, allow_pickle=True)

# ========== Main entry ==========
if __name__ == "__main__":
    config = {
        "complex_dir": "./dataset/v2020/complex-pocket-ligand",
        "ss2_dir": "./dataset/v2020/dssp",
        "key_file": "./dataset/v2020/PDBBind_v2020_labels.txt",
        "output_dir":"./dataset/v2020/feature_graph.npy"
    }

    atom_graphs = process_dataset(
        complex_dir=config["complex_dir"],
        ss2_dir=config["ss2_dir"],
        key_file=config["key_file"],
        output_dir=config["output_dir"]
    )
    
    os.makedirs(config["output_dir"], exist_ok=True)

    save_graphs(atom_graphs, f"{config['output_dir']}/atom_graphs.npy")


