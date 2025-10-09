#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SOURCE_DIR="./dataset/PDBbind_v2020"
TARGET_DIR="./dataset/complex_dir_PDBbind_v2020"
POCKET_DIR="./dataset/pocket_dir_PDBbind_v2020"
COMPLEX_DIR="./dataset/complex_pocket_ligand"
LIGAND_DIR="./dataset/ligand_dir_PDBbind_v2020"
PROTEIN_FIX_DIR="./dataset/protein_fix"
DSSP_DIR="./dataset/dssp"

# 创建目标目录
mkdir -p "$TARGET_DIR" "$POCKET_DIR" "$COMPLEX_DIR" "$LIGAND_DIR" "$PROTEIN_FIX_DIR" "$DSSP_DIR"
PY_SCRIPT="pocket_dssp.py"

# 获取CPU核心数
PARALLEL_JOBS=40

process_folder() {
    local folder="$1"
    local folder_name=$(basename "$folder")
    
    # 文件路径定义
    local ligand_mol2="$folder/${folder_name}_ligand.mol2"
    local protein_pdb="$folder/${folder_name}_protein.pdb"
    local ligand_pdb="$LIGAND_DIR/${folder_name}_ligand.pdb"
    local complex_pdb="$TARGET_DIR/${folder_name}.pdb"
    local pocket_pdb="$POCKET_DIR/${folder_name}_pocket.pdb"
    local complex_pocket_ligand="$COMPLEX_DIR/${folder_name}_pocket_ligand.pdb"
    local protein_fixed_pdb="$PROTEIN_FIX_DIR/${folder_name}_protein_fixed.pdb"
    local dssp_file="$DSSP_DIR/${folder_name}.ss2"

    echo -e "\n=== 处理 $folder_name ==="

    # 1. 检查必要文件
    for f in "$ligand_mol2" "$protein_pdb"; do
        if [[ ! -f "$f" ]]; then
            echo "❌ 缺少文件: $(basename "$f")" >&2
            return 1
        fi
    done

    # 修复缺失原子（修正蛋白质 protein_pdb）
    echo "     修正蛋白质结构缺失原子..."

    if [[ ! -s "$protein_pdb" ]]; then
        echo "❌ 输入文件为空: $protein_pdb" >&2
        return 1
    fi

    local pdbfixer_options="--add-atoms=all --keep-heterogens=none --output=$protein_fixed_pdb" 
    if ! timeout 300 pdbfixer "$protein_pdb" $pdbfixer_options 2>&1; then
        echo "❌ 蛋白质结构修复失败" >&2
        [[ -f "$protein_fixed_pdb" ]] && rm -f "$protein_fixed_pdb"
        return 1
    fi

    if [[ ! -s "$protein_fixed_pdb" ]]; then
        echo "❌ 修复文件为空" >&2
        return 1
    fi

    # 2. 转换配体文件
    echo "     转换配体文件..."
    if ! obabel -imol2 "$ligand_mol2" -opdb -xr -xc -O "$ligand_pdb" 2>/dev/null; then
        echo "❌ 配体转换失败" >&2
        return 1
    fi

    sed -i -E 's/^(ATOM  )/HETATM/g' "$ligand_pdb"
    sed -i -E 's/^(.{17}).../\1LIG/g' "$ligand_pdb"

    # 3. 合并复合物文件
    echo "     合并受体和配体..."
    {
        grep '^ATOM' "$protein_fixed_pdb"
        grep '^HETATM' "$ligand_pdb"
        echo "TER"
    } > "$complex_pdb"

    # 4. 提取活性口袋
    echo "    ️ 提取活性口袋..."
    if ! python "$PY_SCRIPT" "$complex_pdb" "$pocket_pdb" "$dssp_file"; then
        echo "❌ 口袋提取失败或验证未通过" >&2
        rm -f "$pocket_pdb"
        return 1
    fi

    # 5. 结果验证
    if [[ -s "$pocket_pdb" ]]; then
        atom_count=$(grep -c '^ATOM' "$pocket_pdb")
        het_count=$(grep -c '^HETATM' "$pocket_pdb")
        echo "✅  成功提取包含 ${atom_count} 蛋白原子,${het_count}配体原子"
    else
        echo "❌ 生成空口袋文件" >&2
        return 1
    fi

    # 6. 合并口袋和配体（确保仅包含口袋蛋白+原配体）
    echo "     合并口袋和配体..."
    {
	grep '^ATOM' "$pocket_pdb"
        grep '^HETATM' "$ligand_pdb"  # 使用原始转换后的配体
        echo "TER"
    } > "$complex_pocket_ligand"

    # 验证合并结果
    if [[ -s "$complex_pocket_ligand" ]]; then
        atom_count=$(grep -c '^ATOM' "$complex_pocket_ligand")
        het_count=$(grep -c '^HETATM' "$complex_pocket_ligand")
        echo "✅ 成功生成复合物文件（${atom_count} 蛋白原子 + ${aa_count} 个氨基酸残基 + ${het_count} 配体原子）"
    else
        echo "❌ 合并口袋和配体失败" >&2
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

# 生成统计报告
echo "===== 处理完成 ====="
echo "有效口袋文件数: $(find "$POCKET_DIR" -name '*_pocket.pdb' -type f | wc -l)"
echo "有效复合物文件数: $(find "$COMPLEX_DIR" -name '*_pocket_ligand.pdb' -type f | wc -l)"
echo "失败任务数: $(grep -c 'ERROR' "$POCKET_DIR"/parallel_joblog.txt)"
