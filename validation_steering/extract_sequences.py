#!/usr/bin/env python3
"""Extract amino acid sequences from generated CIF files.

This script parses CIF files from BoltzGen outputs and extracts
the designed chain sequences.
"""

import argparse
import csv
from pathlib import Path

import gemmi


def extract_sequence_from_cif(cif_path: Path) -> str | None:
    """Extract amino acid sequence from a CIF file.
    
    Parameters
    ----------
    cif_path : Path
        Path to the CIF file.
        
    Returns
    -------
    str | None
        Amino acid sequence as single-letter codes, or None if parsing fails.
    """
    try:
        structure = gemmi.read_structure(str(cif_path))
        structure.setup_entities()
        
        # Get all sequences from entities
        sequences = []
        for entity in structure.entities:
            if entity.entity_type.name == "Polymer":
                # Get sequence from entity
                if entity.full_sequence:
                    seq = "".join(entity.full_sequence)
                    sequences.append(seq)
        
        # Return the first sequence found (usually the designed chain)
        if sequences:
            return sequences[0]
        
        # Fallback: extract from actual structure
        for chain in structure[0]:
            seq = []
            for residue in chain:
                if residue.name not in ["HOH", "WAT"]:  # Skip waters
                    # Convert 3-letter code to 1-letter
                    three_letter = residue.name
                    one_letter = three_letter_to_one_letter(three_letter)
                    if one_letter:
                        seq.append(one_letter)
            if seq:
                return "".join(seq)
                
        return None
    except Exception as e:
        print(f"Warning: Failed to parse {cif_path}: {e}")
        return None


def three_letter_to_one_letter(three_letter: str) -> str | None:
    """Convert 3-letter amino acid code to 1-letter code."""
    mapping = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return mapping.get(three_letter.upper())


def extract_sequences_from_directory(output_dir: Path, output_csv: Path):
    """Extract sequences from all CIF files in a directory.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing generated CIF files (usually in a subdirectory).
    output_csv : Path
        Output CSV file path.
    """
    sequences = []
    
    # Look for CIF files in common output locations
    cif_files = []
    if (output_dir / "designs").exists():
        cif_files.extend((output_dir / "designs").glob("*.cif"))
    if (output_dir / "intermediate_designs").exists():
        cif_files.extend((output_dir / "intermediate_designs").glob("*.cif"))
    cif_files.extend(output_dir.glob("*.cif"))
    cif_files.extend(output_dir.rglob("*.cif"))
    
    print(f"Found {len(cif_files)} CIF files in {output_dir}")
    
    for cif_file in sorted(cif_files):
        sequence = extract_sequence_from_cif(cif_file)
        if sequence:
            sequences.append({
                "file": cif_file.name,
                "sequence": sequence,
                "length": len(sequence),
            })
        else:
            print(f"Warning: Could not extract sequence from {cif_file}")
    
    # Write to CSV
    if sequences:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "sequence", "length"])
            writer.writeheader()
            writer.writerows(sequences)
        print(f"Extracted {len(sequences)} sequences to {output_csv}")
    else:
        print(f"Warning: No sequences extracted from {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequences from BoltzGen output CIF files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing generated CIF files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.output_dir}")
    
    extract_sequences_from_directory(args.output_dir, args.output)


if __name__ == "__main__":
    main()

