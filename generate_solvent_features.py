#!/usr/bin/env python
"""Generate solvent-only features for Chemprop training

"""

import pandas as pd
import numpy as np
import argparse

# Solvent property descriptors (24 features)
# Values from:
# - CRC Handbook of Chemistry and Physics
# - Reichardt's "Solvents and Solvent Effects in Organic Chemistry" (4th ed.)
# - Kamlet-Taft parameters from J. Org. Chem. 1983, 48, 2877-2887
# - Gutmann DN/AN from Coord. Chem. Rev. 1976, 18, 225-255
# - NIST Chemistry WebBook for physical properties

_SOLVENT_DATA = {
    'water': {'MW': 18.015, 'logP': -1.38, 'HBD': 2, 'HBA': 2, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
              'Dielectric': 80.1, 'Dipole': 1.85, 'RefractiveIndex': 1.333, 'Viscosity': 0.890,
              'SurfaceTension': 72.8, 'Density': 0.997, 'MeltingPoint': 0.0, 'BoilingPoint': 100.0,
              'FlashPoint': np.nan, 'VaporPressure': 2.3, 'ET30': 63.1, 'Polarity': 10.2,
              'HBond_acidity': 1.17, 'HBond_basicity': 0.47, 'Polarizability': 1.45, 'DN': 33.0, 'AN': 54.8},

    'dichloromethane': {'MW': 84.93, 'logP': 1.25, 'HBD': 0, 'HBA': 0, 'TPSA': 0.0, 'RotBonds': 0, 'Aromatic': 0,
                        'Dielectric': 8.93, 'Dipole': 1.60, 'RefractiveIndex': 1.424, 'Viscosity': 0.413,
                        'SurfaceTension': 26.5, 'Density': 1.327, 'MeltingPoint': -96.7, 'BoilingPoint': 39.6,
                        'FlashPoint': np.nan, 'VaporPressure': 47.4, 'ET30': 40.7, 'Polarity': 3.1,
                        'HBond_acidity': 0.13, 'HBond_basicity': 0.10, 'Polarizability': 6.48, 'DN': 0.0, 'AN': 20.4},

    'mecn': {'MW': 41.05, 'logP': -0.34, 'HBD': 0, 'HBA': 1, 'TPSA': 23.79, 'RotBonds': 0, 'Aromatic': 0,
             'Dielectric': 37.5, 'Dipole': 3.92, 'RefractiveIndex': 1.344, 'Viscosity': 0.369,
             'SurfaceTension': 29.3, 'Density': 0.786, 'MeltingPoint': -45.7, 'BoilingPoint': 81.6,
             'FlashPoint': 6.0, 'VaporPressure': 9.71, 'ET30': 45.6, 'Polarity': 5.8,
             'HBond_acidity': 0.19, 'HBond_basicity': 0.40, 'Polarizability': 4.40, 'DN': 14.1, 'AN': 18.9},

    'dmso': {'MW': 78.13, 'logP': -1.35, 'HBD': 0, 'HBA': 2, 'TPSA': 36.28, 'RotBonds': 0, 'Aromatic': 0,
             'Dielectric': 46.7, 'Dipole': 3.96, 'RefractiveIndex': 1.479, 'Viscosity': 1.996,
             'SurfaceTension': 43.5, 'Density': 1.100, 'MeltingPoint': 18.5, 'BoilingPoint': 189.0,
             'FlashPoint': 95.0, 'VaporPressure': 0.06, 'ET30': 45.1, 'Polarity': 7.2,
             'HBond_acidity': 0.00, 'HBond_basicity': 0.76, 'Polarizability': 7.96, 'DN': 29.8, 'AN': 19.3},

    'thf': {'MW': 72.11, 'logP': 0.46, 'HBD': 0, 'HBA': 1, 'TPSA': 9.23, 'RotBonds': 0, 'Aromatic': 0,
            'Dielectric': 7.58, 'Dipole': 1.75, 'RefractiveIndex': 1.407, 'Viscosity': 0.456,
            'SurfaceTension': 26.4, 'Density': 0.889, 'MeltingPoint': -108.4, 'BoilingPoint': 66.0,
            'FlashPoint': -14.0, 'VaporPressure': 19.3, 'ET30': 37.4, 'Polarity': 4.0,
            'HBond_acidity': 0.00, 'HBond_basicity': 0.55, 'Polarizability': 6.97, 'DN': 20.0, 'AN': 8.0},

    'methanol': {'MW': 32.04, 'logP': -0.77, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
                 'Dielectric': 32.7, 'Dipole': 1.70, 'RefractiveIndex': 1.329, 'Viscosity': 0.544,
                 'SurfaceTension': 22.1, 'Density': 0.792, 'MeltingPoint': -97.6, 'BoilingPoint': 64.7,
                 'FlashPoint': 11.0, 'VaporPressure': 12.8, 'ET30': 55.4, 'Polarity': 5.1,
                 'HBond_acidity': 0.98, 'HBond_basicity': 0.66, 'Polarizability': 3.29, 'DN': 30.0, 'AN': 41.5},

    'meoh': {'MW': 32.04, 'logP': -0.77, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
             'Dielectric': 32.7, 'Dipole': 1.70, 'RefractiveIndex': 1.329, 'Viscosity': 0.544,
             'SurfaceTension': 22.1, 'Density': 0.792, 'MeltingPoint': -97.6, 'BoilingPoint': 64.7,
             'FlashPoint': 11.0, 'VaporPressure': 12.8, 'ET30': 55.4, 'Polarity': 5.1,
             'HBond_acidity': 0.98, 'HBond_basicity': 0.66, 'Polarizability': 3.29, 'DN': 30.0, 'AN': 41.5},

    'acetone': {'MW': 58.08, 'logP': -0.24, 'HBD': 0, 'HBA': 1, 'TPSA': 17.07, 'RotBonds': 0, 'Aromatic': 0,
                'Dielectric': 20.7, 'Dipole': 2.88, 'RefractiveIndex': 1.359, 'Viscosity': 0.306,
                'SurfaceTension': 23.7, 'Density': 0.791, 'MeltingPoint': -94.7, 'BoilingPoint': 56.2,
                'FlashPoint': -20.0, 'VaporPressure': 24.0, 'ET30': 42.2, 'Polarity': 5.1,
                'HBond_acidity': 0.08, 'HBond_basicity': 0.48, 'Polarizability': 6.33, 'DN': 17.0, 'AN': 12.5},

    'ethanol': {'MW': 46.07, 'logP': -0.31, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
                'Dielectric': 24.3, 'Dipole': 1.69, 'RefractiveIndex': 1.361, 'Viscosity': 1.074,
                'SurfaceTension': 22.1, 'Density': 0.789, 'MeltingPoint': -114.1, 'BoilingPoint': 78.4,
                'FlashPoint': 13.0, 'VaporPressure': 5.95, 'ET30': 51.9, 'Polarity': 5.2,
                'HBond_acidity': 0.86, 'HBond_basicity': 0.75, 'Polarizability': 5.11, 'DN': 32.0, 'AN': 37.1},

    'etoh': {'MW': 46.07, 'logP': -0.31, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
             'Dielectric': 24.3, 'Dipole': 1.69, 'RefractiveIndex': 1.361, 'Viscosity': 1.074,
             'SurfaceTension': 22.1, 'Density': 0.789, 'MeltingPoint': -114.1, 'BoilingPoint': 78.4,
             'FlashPoint': 13.0, 'VaporPressure': 5.95, 'ET30': 51.9, 'Polarity': 5.2,
             'HBond_acidity': 0.86, 'HBond_basicity': 0.75, 'Polarizability': 5.11, 'DN': 32.0, 'AN': 37.1},

    'dmf': {'MW': 73.09, 'logP': -0.87, 'HBD': 0, 'HBA': 1, 'TPSA': 20.31, 'RotBonds': 1, 'Aromatic': 0,
            'Dielectric': 36.7, 'Dipole': 3.82, 'RefractiveIndex': 1.430, 'Viscosity': 0.802,
            'SurfaceTension': 35.2, 'Density': 0.944, 'MeltingPoint': -60.5, 'BoilingPoint': 153.0,
            'FlashPoint': 58.0, 'VaporPressure': 0.52, 'ET30': 43.2, 'Polarity': 6.4,
            'HBond_acidity': 0.00, 'HBond_basicity': 0.69, 'Polarizability': 7.81, 'DN': 26.6, 'AN': 16.0},

    '1,2-dichloroethane': {'MW': 98.96, 'logP': 1.48, 'HBD': 0, 'HBA': 0, 'TPSA': 0.0, 'RotBonds': 1, 'Aromatic': 0,
                           'Dielectric': 10.4, 'Dipole': 1.83, 'RefractiveIndex': 1.444, 'Viscosity': 0.779,
                           'SurfaceTension': 32.2, 'Density': 1.253, 'MeltingPoint': -35.3, 'BoilingPoint': 83.5,
                           'FlashPoint': 13.0, 'VaporPressure': 8.7, 'ET30': 41.3, 'Polarity': 3.5,
                           'HBond_acidity': 0.00, 'HBond_basicity': 0.00, 'Polarizability': 8.0, 'DN': 0.0, 'AN': 16.7},

    # NEW: n-Propanol (1-propanol) - exact values from CRC Handbook & Reichardt
    'nproh': {'MW': 60.10, 'logP': 0.25, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 2, 'Aromatic': 0,
              'Dielectric': 20.8, 'Dipole': 1.68, 'RefractiveIndex': 1.385, 'Viscosity': 1.945,
              'SurfaceTension': 23.7, 'Density': 0.804, 'MeltingPoint': -126.1, 'BoilingPoint': 97.2,
              'FlashPoint': 15.0, 'VaporPressure': 2.0, 'ET30': 50.7, 'Polarity': 4.0,
              'HBond_acidity': 0.84, 'HBond_basicity': 0.90, 'Polarizability': 5.54, 'DN': 30.0, 'AN': 37.3},

    # NEW: i-Propanol (2-propanol, isopropanol) - exact values
    'iproh': {'MW': 60.10, 'logP': 0.05, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
              'Dielectric': 19.9, 'Dipole': 1.66, 'RefractiveIndex': 1.377, 'Viscosity': 2.04,
              'SurfaceTension': 21.7, 'Density': 0.785, 'MeltingPoint': -89.5, 'BoilingPoint': 82.3,
              'FlashPoint': 12.0, 'VaporPressure': 4.4, 'ET30': 48.4, 'Polarity': 3.9,
              'HBond_acidity': 0.76, 'HBond_basicity': 0.84, 'Polarizability': 5.48, 'DN': 31.5, 'AN': 33.5},

    # NEW: TFE (2,2,2-trifluoroethanol) - exact values from Reichardt
    'tfe': {'MW': 100.04, 'logP': 0.40, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 1, 'Aromatic': 0,
            'Dielectric': 26.7, 'Dipole': 2.52, 'RefractiveIndex': 1.291, 'Viscosity': 1.73,
            'SurfaceTension': 20.1, 'Density': 1.383, 'MeltingPoint': -43.0, 'BoilingPoint': 74.0,
            'FlashPoint': 23.0, 'VaporPressure': 8.5, 'ET30': 59.8, 'Polarity': 8.6,
            'HBond_acidity': 1.51, 'HBond_basicity': 0.00, 'Polarizability': 5.73, 'DN': 0.0, 'AN': 53.0},

    # NEW: HFIP (hexafluoroisopropanol, 1,1,1,3,3,3-hexafluoro-2-propanol) - exact values
    'hfip': {'MW': 168.04, 'logP': 1.24, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23, 'RotBonds': 0, 'Aromatic': 0,
             'Dielectric': 16.7, 'Dipole': 2.55, 'RefractiveIndex': 1.275, 'Viscosity': 1.27,
             'SurfaceTension': 15.4, 'Density': 1.596, 'MeltingPoint': -4.0, 'BoilingPoint': 58.2,
             'FlashPoint': 19.0, 'VaporPressure': 20.5, 'ET30': 65.3, 'Polarity': 9.3,
             'HBond_acidity': 1.96, 'HBond_basicity': 0.00, 'Polarizability': 6.50, 'DN': 0.0, 'AN': 62.4},
}

# Solvent mixture definitions with 50:50 ratio
# Format: 'mixture name': ('solvent1', 'solvent2')
_SOLVENT_MIXTURES = {
    'nproh-mecn mix': ('nproh', 'mecn'),
    'iproh-mecn mix': ('iproh', 'mecn'),
    'water-acetone mix': ('water', 'acetone'),
    'water-etoh mix': ('water', 'etoh'),
    'etoh-mecn mix': ('etoh', 'mecn'),
    'meoh-mecn mix': ('meoh', 'mecn'),
    'water-hfip mix': ('water', 'hfip'),
    'water-mecn mix': ('water', 'mecn'),
    'water-tfe mix': ('water', 'tfe'),
    'aq acetone': ('water', 'acetone'),
    'aq mecn': ('water', 'mecn'),
}

# Build final solvent properties dict
SOLVENT_PROPERTIES = _SOLVENT_DATA.copy()

def get_solvent_features(solvent_name):
    """Get 24-dimensional feature vector for a solvent.

    Handles both pure solvents and 50:50 mixtures.
    """
    solvent_lower = solvent_name.lower().strip()

    # Check if it's a mixture
    if solvent_lower in _SOLVENT_MIXTURES:
        solv1, solv2 = _SOLVENT_MIXTURES[solvent_lower]

        if solv1.lower() in SOLVENT_PROPERTIES and solv2.lower() in SOLVENT_PROPERTIES:
            props1 = SOLVENT_PROPERTIES[solv1.lower()]
            props2 = SOLVENT_PROPERTIES[solv2.lower()]

            # Calculate 50:50 weighted average
            mixed_props = {}
            for key in props1.keys():
                val1 = props1[key]
                val2 = props2[key]

                # Handle NaN values
                if pd.isna(val1) and pd.isna(val2):
                    mixed_props[key] = np.nan
                elif pd.isna(val1):
                    mixed_props[key] = val2
                elif pd.isna(val2):
                    mixed_props[key] = val1
                else:
                    mixed_props[key] = 0.5 * val1 + 0.5 * val2

            return list(mixed_props.values())
        else:
            print(f"Warning: Mixture components not found for '{solvent_name}', using zeros")
            return [0.0] * 24

    # Pure solvent
    if solvent_lower in SOLVENT_PROPERTIES:
        props = SOLVENT_PROPERTIES[solvent_lower]
        return list(props.values())
    else:
        # Return zeros for unknown solvents
        print(f"Warning: Unknown solvent '{solvent_name}', using zeros")
        return [0.0] * 24

def main():
    parser = argparse.ArgumentParser(description='Generate solvent features for Chemprop (v2 with 50:50 mixtures)')
    parser.add_argument('--data_csv', required=True, help='Input CSV with SMILES and Solvent columns')
    parser.add_argument('--out_csv', required=True, help='Output CSV with solvent features')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_csv)

    print(f"Loaded {len(df)} molecules")
    print(f"Solvents: {df['Solvent'].unique()}")

    # Generate solvent features
    solvent_features = []
    for solvent in df['Solvent']:
        features = get_solvent_features(solvent)
        solvent_features.append(features)

    # Create features dataframe
    feature_cols = ['MW', 'logP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Aromatic',
                    'Dielectric', 'Dipole', 'RefractiveIndex', 'Viscosity',
                    'SurfaceTension', 'Density', 'MeltingPoint', 'BoilingPoint',
                    'FlashPoint', 'VaporPressure', 'ET30', 'Polarity',
                    'HBond_acidity', 'HBond_basicity', 'Polarizability', 'DN', 'AN']

    features_df = pd.DataFrame(solvent_features, columns=feature_cols)

    # Replace NaN values with 0 (for missing flash points, etc.)
    features_df = features_df.fillna(0)

    # Save
    features_df.to_csv(args.out_csv, index=False)
    print(f"Saved solvent features to {args.out_csv}")
    print(f"Feature dimensions: {features_df.shape}")

if __name__ == '__main__':
    main()
