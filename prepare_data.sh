#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SOURCE_DIR="./dataset/PDBbind_v2020"
TARGET_DIR="./dataset/v2020/complex_dir_PDBbind_v2020"
POCKET_DIR="./dataset/v2020/pocket_dir_PDBbind_v2020"
COMPLEX_DIR="./dataset/v2020/complex_pocket_ligand"
LIGAND_DIR="./dataset/v2020/ligand_dir_PDBbind_v2020"
PROTEIN_FIX_DIR="./dataset/v2020/protein_fix"
DSSP_DIR="./dataset/v2020/dssp"

mkdir -p "$TARGET_DIR" "$POCKET_DIR" "$COMPLEX_DIR" "$LIGAND_DIR" "$PROTEIN_FIX_DIR" "$DSSP_DIR"
PY_SCRIPT="pocket_dssp.py"

PARALLEL_JOBS=40

process_folder() {
    local folder="$1"
    local folder_name=$(basename "$folder")
    
    local ligand_mol2="$folder/${folder_name}_ligand.mol2"
    local protein_pdb="$folder/${folder_name}_protein.pdb"
    local ligand_pdb="$LIGAND_DIR/${folder_name}_ligand.pdb"
    local complex_pdb="$TARGET_DIR/${folder_name}.pdb"
    local pocket_pdb="$POCKET_DIR/${folder_name}_pocket.pdb"
    local complex_pocket_ligand="$COMPLEX_DIR/${folder_name}_pocket_ligand.pdb"
    local protein_fixed_pdb="$PROTEIN_FIX_DIR/${folder_name}_protein_fixed.pdb"
    local dssp_file="$DSSP_DIR/${folder_name}.ss2"

    echo -e "\n=== processing $folder_name ==="

    # 1. Check required input files
    for f in "$ligand_mol2" "$protein_pdb"; do
        if [[ ! -f "$f" ]]; then
            echo "Missing required file: $(basename "$f")" >&2
            return 1
        fi
    done

    # Fix missing atoms in protein structure
    echo "Fixing missing atoms in protein structure..."

    if [[ ! -s "$protein_pdb" ]]; then
        echo "Input protein file is empty:  $protein_pdb" >&2
        return 1
    fi

    local pdbfixer_options="--add-atoms=all --keep-heterogens=none --output=$protein_fixed_pdb" 
    if ! timeout 300 pdbfixer "$protein_pdb" $pdbfixer_options 2>&1; then
        echo "Protein structure fixing failed" >&2
        [[ -f "$protein_fixed_pdb" ]] && rm -f "$protein_fixed_pdb"
        return 1
    fi

    if [[ ! -s "$protein_fixed_pdb" ]]; then
        echo "Fixed protein file is empty" >&2
        return 1
    fi

    # 2. Convert ligand file format
    echo "Converting ligand file..."
    if ! obabel -imol2 "$ligand_mol2" -opdb -xr -xc -O "$ligand_pdb" 2>/dev/null; then
        echo "Ligand conversion failed" >&2
        return 1
    fi

    sed -i -E 's/^(ATOM  )/HETATM/g' "$ligand_pdb"
    sed -i -E 's/^(.{17}).../\1LIG/g' "$ligand_pdb"

    # 3. Merge protein and ligand into complex structure
    echo "Merging receptor and ligand..."
    {
        grep '^ATOM' "$protein_fixed_pdb"
        grep '^HETATM' "$ligand_pdb"
        echo "TER"
    } > "$complex_pdb"

    # 4. Extract binding pocket
    echo "Extracting binding pocket..."
    if ! python "$PY_SCRIPT" "$complex_pdb" "$pocket_pdb" "$dssp_file"; then
        echo "Pocket extraction failed or validation did not pass" >&2
        rm -f "$pocket_pdb"
        return 1
    fi

    # 5. Validate pocket extraction results
    if [[ -s "$pocket_pdb" ]]; then
        atom_count=$(grep -c '^ATOM' "$pocket_pdb")
        het_count=$(grep -c '^HETATM' "$pocket_pdb")
        echo "Successfully extracted pocket containing ${atom_count} protein atoms and ${het_count} ligand atoms"
    else
        echo "Generated pocket file is empty" >&2
        return 1
    fi

    # 6. Merge pocket protein and original ligand
    echo "Merging pocket protein and ligand..."
    {
	grep '^ATOM' "$pocket_pdb"
        grep '^HETATM' "$ligand_pdb" 
        echo "TER"
    } > "$complex_pocket_ligand"

    # Validate merged complex
    if [[ -s "$complex_pocket_ligand" ]]; then
        atom_count=$(grep -c '^ATOM' "$complex_pocket_ligand")
        het_count=$(grep -c '^HETATM' "$complex_pocket_ligand")
        echo "Successfully generated complex file（${atom_count} protein atoms + ${het_count} ligand atoms）"
    else
        echo "Failed to merge pocket protein and ligand" >&2
        return 1
    fi
}

export -f process_folder
export TARGET_DIR POCKET_DIR COMPLEX_DIR LIGAND_DIR PROTEIN_FIX_DIR PY_SCRIPT DSSP_DIR

find "$SOURCE_DIR" -maxdepth 1 -type d -name '????' | \
parallel --jobs $PARALLEL_JOBS \
        --progress \
        --joblog "$POCKET_DIR/parallel_joblog.txt" \
        --tagstring "Job {#} ({/.})" \
        --eta \
        "process_folder {}"

# Generate summary report
echo "===== Processing completed ====="
echo "Valid pocket files: $(find "$POCKET_DIR" -name '*_pocket.pdb' -type f | wc -l)"
echo "Valid complex files:  $(find "$COMPLEX_DIR" -name '*_pocket_ligand.pdb' -type f | wc -l)"
echo "Failed jobs: $(grep -c 'ERROR' "$POCKET_DIR"/parallel_joblog.txt)"
