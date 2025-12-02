#!/usr/bin/env python
"""
Postprocess and classify generated molecules from finetuning experiments.

This script:
1. Loads all epoch files (good_molecules.*) from a training directory
2. Deduplicates molecules (keeping highest log k)
3. Calculates additional metrics (SA score, normalized values)
4. Classifies carbanions using CarbanionClassifier
5. Exports a classified CSV file similar to accumulated_10k_all_unique_molecules_classified.csv
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from tqdm.auto import tqdm
import re
import os
import argparse
import glob
from carbanion_classifier import CarbanionClassifier


def calculate_sa_score(smiles):
    """Calculate Synthetic Accessibility score for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return sascorer.calculateScore(mol)
    except:
        return np.nan


def count_carbanion_sites(smi):
    """Count number of carbanion sites in a SMILES string."""
    return len(re.findall(r'\[(?:C|CH|CH2)-\]', smi))


def normalize(series):
    """Min-max normalization of a pandas Series."""
    if series.min() == series.max():
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def load_epoch_files(base_dir, pattern='good_molecules.*'):
    """
    Load all epoch files from a directory.

    Args:
        base_dir: Directory containing epoch files
        pattern: Glob pattern for epoch files (default: 'good_molecules.*')

    Returns:
        DataFrame with all molecules from all epochs
        DataFrame with per-epoch statistics
    """
    print("=" * 80)
    print("LOADING EPOCH FILES")
    print("=" * 80)
    print(f"Directory: {base_dir}")
    print(f"Pattern: {pattern}")

    # Find all matching files
    file_pattern = os.path.join(base_dir, pattern)
    epoch_files = sorted(glob.glob(file_pattern))

    if not epoch_files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    print(f"Found {len(epoch_files)} epoch files")

    all_data = []
    epoch_stats = []

    for file_path in tqdm(epoch_files, desc="Loading epochs"):
        # Extract epoch number from filename
        basename = os.path.basename(file_path)
        try:
            epoch = int(basename.split('.')[-1])
        except:
            print(f"⚠️  Warning: Could not parse epoch number from {basename}, skipping")
            continue

        try:
            df_epoch = pd.read_csv(file_path, sep='\t', header=0)
            df_epoch['epoch'] = epoch
            all_data.append(df_epoch)

            epoch_stats.append({
                'epoch': epoch,
                'n_molecules': len(df_epoch),
                'mean_logk': df_epoch['logk_CO2'].mean(),
                'max_logk': df_epoch['logk_CO2'].max(),
                'mean_N': df_epoch['N'].mean(),
                'mean_sN': df_epoch['sN'].mean()
            })

            print(f"  Epoch {epoch:2d}: {len(df_epoch):6,} molecules | "
                  f"Mean log k: {df_epoch['logk_CO2'].mean():.3f} | "
                  f"Max log k: {df_epoch['logk_CO2'].max():.3f}")

        except Exception as e:
            print(f"❌ Error loading {basename}: {e}")

    if not all_data:
        raise ValueError("No data loaded successfully")

    df_all = pd.concat(all_data, ignore_index=True)
    df_epoch_stats = pd.DataFrame(epoch_stats)

    print(f"\n✅ Total molecules across all epochs: {len(df_all):,}")
    print(f"✅ Unique SMILES: {df_all['SMILES'].nunique():,}")

    return df_all, df_epoch_stats


def deduplicate_molecules(df):
    """
    Deduplicate molecules by SMILES, keeping the one with highest log k.

    Args:
        df: DataFrame with 'SMILES' and 'logk_CO2' columns

    Returns:
        Deduplicated DataFrame
    """
    print("\n" + "=" * 80)
    print("DEDUPLICATION")
    print("=" * 80)

    print(f"Molecules before deduplication: {len(df):,}")
    print(f"Unique SMILES: {df['SMILES'].nunique():,}")

    # Sort by logk_CO2 (descending) and keep first occurrence of each SMILES
    df_dedup = df.sort_values('logk_CO2', ascending=False).drop_duplicates('SMILES', keep='first')

    print(f"Molecules after deduplication: {len(df_dedup):,}")
    print(f"Removed: {len(df) - len(df_dedup):,} duplicates")
    print(f"Retention rate: {len(df_dedup)/len(df)*100:.1f}%")

    return df_dedup


def calculate_metrics(df):
    """
    Calculate additional metrics for molecules.

    Args:
        df: DataFrame with 'SMILES', 'logk_CO2', and 'SA_score' columns

    Returns:
        DataFrame with added metric columns
    """
    print("\n" + "=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)

    # SA scores
    if 'SA_score' not in df.columns:
        print("Calculating SA scores...")
        df['SA_score'] = [calculate_sa_score(smi) for smi in tqdm(df['SMILES'], desc="SA scores")]

    # Carbanion site count
    if 'n_anion_sites' not in df.columns:
        print("Counting carbanion sites...")
        df['n_anion_sites'] = df['SMILES'].apply(count_carbanion_sites)

    # Normalized values
    print("Calculating normalized scores...")
    df['logk_normalized'] = normalize(df['logk_CO2'])
    df['SA_normalized'] = normalize(df['SA_score'])
    df['combined_score'] = df['logk_normalized'] - df['SA_normalized']

    return df


def classify_molecules(df, smiles_column='SMILES'):
    """
    Classify molecules using CarbanionClassifier.

    Args:
        df: DataFrame with SMILES column
        smiles_column: Name of column containing SMILES strings

    Returns:
        DataFrame with classification columns added
    """
    print("\n" + "=" * 80)
    print("CLASSIFYING CARBANIONS")
    print("=" * 80)

    classifier = CarbanionClassifier()
    df_classified = classifier.classify_dataframe(df, smiles_column=smiles_column)

    print(f"✅ Classification complete!")
    print(f"   Added {len([c for c in df_classified.columns if c not in df.columns])} classification columns")

    return df_classified


def print_statistics(df):
    """Print summary statistics for the dataset."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    print(f"\nTotal unique molecules: {len(df):,}")
    print(f"\nlog k (CO2):")
    print(f"  Mean: {df['logk_CO2'].mean():.3f} ± {df['logk_CO2'].std():.3f}")
    print(f"  Range: [{df['logk_CO2'].min():.3f}, {df['logk_CO2'].max():.3f}]")

    print(f"\nSA Score:")
    print(f"  Mean: {df['SA_score'].mean():.3f} ± {df['SA_score'].std():.3f}")
    print(f"  Range: [{df['SA_score'].min():.3f}, {df['SA_score'].max():.3f}]")

    print(f"\nCombined Score (logk_norm - SA_norm):")
    print(f"  Mean: {df['combined_score'].mean():.3f} ± {df['combined_score'].std():.3f}")

    print(f"\nMayr Parameters:")
    print(f"  N:  {df['N'].mean():.3f} ± {df['N'].std():.3f}")
    print(f"  sN: {df['sN'].mean():.4f} ± {df['sN'].std():.4f}")

    # Classification statistics
    if 'combination' in df.columns:
        print(f"\nTop 10 Alpha-Substituent Combinations:")
        combo_counts = df['combination'].value_counts().head(10)
        for combo, count in combo_counts.items():
            pct = count / len(df) * 100
            print(f"  {combo:20s}: {count:6,} ({pct:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Postprocess and classify generated molecules from finetuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10k batch
  python postprocess_and_classify_generated.py \\
      --input_dir /path/to/finetune_output \\
      --output molecules_classified.csv

  # Process with custom pattern
  python postprocess_and_classify_generated.py \\
      --input_dir /path/to/output \\
      --pattern "generated_*.tsv" \\
      --output results.csv
        """
    )

    parser.add_argument('--input_dir', required=True,
                       help='Directory containing epoch files (good_molecules.*)')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    parser.add_argument('--pattern', default='good_molecules.*',
                       help='Glob pattern for epoch files (default: good_molecules.*)')
    parser.add_argument('--save_epoch_stats', action='store_true',
                       help='Save per-epoch statistics to separate CSV')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Load all epoch files
    df_all, df_epoch_stats = load_epoch_files(args.input_dir, args.pattern)

    # Deduplicate
    df_unique = deduplicate_molecules(df_all)

    # Calculate metrics
    df_metrics = calculate_metrics(df_unique)

    # Classify
    df_classified = classify_molecules(df_metrics)

    # Print statistics
    print_statistics(df_classified)

    # Save outputs
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    df_classified.to_csv(args.output, index=False)
    print(f"✅ Saved classified molecules: {args.output}")
    print(f"   {len(df_classified):,} molecules × {len(df_classified.columns)} columns")

    if args.save_epoch_stats:
        epoch_stats_file = args.output.replace('.csv', '_epoch_stats.csv')
        df_epoch_stats.to_csv(epoch_stats_file, index=False)
        print(f"✅ Saved epoch statistics: {epoch_stats_file}")

    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
