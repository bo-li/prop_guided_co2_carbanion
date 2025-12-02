#!/usr/bin/env python
"""
Carbanion Structure Classifier

A reusable class for classifying carbanion structures by their
alpha-substituents (electron-withdrawing groups).

Usage:
    from carbanion_classifier import CarbanionClassifier

    classifier = CarbanionClassifier()
    groups = classifier.get_alpha_groups('c1ccccc1[C-](C#N)C(=O)OC')
    # Returns: {'Ar', 'CN', 'CO2R'}

    # Or apply to a DataFrame
    df['alpha_groups'] = df['smiles'].apply(classifier.get_alpha_groups)
"""

from rdkit import Chem
from rdkit.Chem import Draw
from collections import Counter
import pandas as pd
import numpy as np


class CarbanionClassifier:
    """Classify carbanion structures by their alpha-substituents."""

    # Supported EWG types (halides grouped as 'X')
    # 'Aliphatic' is added for aliphatic carbon alpha-substituents
    # 'Other' is for unrecognized/weak substituents (S, isonitriles, etc.)
    EWG_TYPES = ['Ar', 'CN', 'CO2R', 'COR', 'C=N', 'NO2', 'SO2R', 'X', 'Aliphatic', 'Other']

    # Human-readable names
    EWG_NAMES = {
        'Ar': 'Aryl',
        'CN': 'Cyano',
        'CO2R': 'Ester',
        'COR': 'Ketone',
        'C=N': 'Imine',
        'NO2': 'Nitro',
        'SO2R': 'Sulfonyl',
        'X': 'Halide',
        'Aliphatic': 'Aliphatic',
        'Other': 'Other'
    }

    def __init__(self):
        """Initialize the classifier."""
        pass

    def find_carbanion_center(self, mol):
        """
        Find the carbanion center (C with -1 charge) in a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            int or None: Index of carbanion carbon, or None if not found
        """
        if mol is None:
            return None

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetFormalCharge() == -1:
                return atom.GetIdx()
        return None

    def get_substitution_degree(self, smiles):
        """
        Get the substitution degree (primary/secondary/tertiary) of carbanion center.

        Based on SMARTS pattern matching for explicit hydrogen count:
        - Methyl: [CH3-] (3 hydrogens)
        - Primary: [CH2-] (2 hydrogens)
        - Secondary: [CH-] (1 hydrogen)
        - Tertiary: [C-] (0 hydrogens)

        For cyclic/aromatic carbanions where SMARTS fails, uses carbon neighbor count:
        - 1 carbon neighbor → primary
        - 2 carbon neighbors → secondary
        - 3 carbon neighbors → tertiary

        Args:
            smiles: SMILES string

        Returns:
            str: 'primary', 'secondary', 'tertiary', 'methyl', or None if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Use SMARTS patterns to match specific carbanion types
        patterns = {
            'methyl': Chem.MolFromSmarts('[CH3-]'),
            'primary': Chem.MolFromSmarts('[CH2-]'),
            'secondary': Chem.MolFromSmarts('[CH-]'),
            'tertiary': Chem.MolFromSmarts('[C-]'),  # Note: This will also match CH3-, CH2-, CH-
        }

        # Check in order of specificity (most specific first)
        for degree in ['methyl', 'primary', 'secondary']:
            if mol.HasSubstructMatch(patterns[degree]):
                return degree

        # If none of the above, check for tertiary (least specific pattern)
        if mol.HasSubstructMatch(patterns['tertiary']):
            return 'tertiary'

        # Fallback for cyclic/aromatic carbanions: count carbon neighbors
        # This handles cases like indene, cyclopentadienyl, etc.
        carbanion_idx = self.find_carbanion_center(mol)
        if carbanion_idx is not None:
            carbanion_atom = mol.GetAtomWithIdx(carbanion_idx)
            num_carbons = sum(1 for n in carbanion_atom.GetNeighbors() if n.GetSymbol() == 'C')
            num_hydrogens = carbanion_atom.GetTotalNumHs()

            # Determine degree based on carbon neighbors and hydrogens
            if num_carbons == 1 and num_hydrogens == 2:
                return 'primary'
            elif num_carbons == 2 and num_hydrogens == 1:
                return 'secondary'  # Most cyclic carbanions fall here
            elif num_carbons == 3 and num_hydrogens == 0:
                return 'tertiary'
            elif num_carbons == 0 and num_hydrogens == 3:
                return 'methyl'

        return None

    def count_carbon_neighbors(self, smiles):
        """
        Count the number of carbon atoms directly attached to the carbanion center.

        Args:
            smiles: SMILES string

        Returns:
            int: Number of carbon neighbors (0-3), or -1 if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1

        carbanion_idx = self.find_carbanion_center(mol)
        if carbanion_idx is None:
            return -1

        carbanion_atom = mol.GetAtomWithIdx(carbanion_idx)
        return sum(1 for n in carbanion_atom.GetNeighbors() if n.GetSymbol() == 'C')

    def count_total_substituents(self, smiles):
        """
        Count the total number of heavy atom substituents on the carbanion center.

        Args:
            smiles: SMILES string

        Returns:
            int: Total number of non-hydrogen neighbors, or -1 if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1

        carbanion_idx = self.find_carbanion_center(mol)
        if carbanion_idx is None:
            return -1

        carbanion_atom = mol.GetAtomWithIdx(carbanion_idx)
        return len(list(carbanion_atom.GetNeighbors()))

    def count_aromatic_rings(self, smiles):
        """
        Count the number of aromatic rings in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            int: Number of aromatic rings, or 0 if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        ring_info = mol.GetRingInfo()
        aromatic_rings = 0
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_rings += 1
        return aromatic_rings

    def count_aliphatic_rings(self, smiles):
        """
        Count the number of aliphatic (non-aromatic) rings in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            int: Number of aliphatic rings, or 0 if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        ring_info = mol.GetRingInfo()
        aliphatic_rings = 0
        for ring in ring_info.AtomRings():
            if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aliphatic_rings += 1
        return aliphatic_rings

    def count_total_rings(self, smiles):
        """
        Count the total number of rings in the molecule (aromatic + aliphatic).

        Args:
            smiles: SMILES string

        Returns:
            int: Total number of rings, or 0 if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        return mol.GetRingInfo().NumRings()

    def get_ring_sizes(self, smiles):
        """
        Get the sizes of all rings in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            list: List of ring sizes (e.g., [6, 6, 5] for two benzene rings and a cyclopentane)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        return sorted([len(ring) for ring in ring_info.AtomRings()], reverse=True)

    def get_alpha_groups(self, smiles, include_aromatic_substituents=False):
        """
        Get all EWG groups DIRECTLY bonded to the carbanion center.

        This method analyzes what functional groups are directly attached
        to the negatively charged carbon atom.

        Args:
            smiles: SMILES string of the carbanion
            include_aromatic_substituents: If True, also detect EWG substituents
                on directly-bonded aromatic rings (e.g., para-NO2, para-CN).
                These are added with 'Ar-' prefix (e.g., 'Ar-NO2', 'Ar-CN').
                Default: False (Strategy 1: direct bonds only)

        Returns:
            list: List of EWG type strings (e.g., ['Ar', 'CN', 'CO2R'])
                  Returns list (not set) to preserve multiplicity for duplicate groups
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        carbanion_idx = self.find_carbanion_center(mol)
        if carbanion_idx is None:
            return []

        carbanion_atom = mol.GetAtomWithIdx(carbanion_idx)
        neighbors = list(carbanion_atom.GetNeighbors())
        alpha_groups = []

        for n_atom in neighbors:
            n_symbol = n_atom.GetSymbol()
            n_idx = n_atom.GetIdx()
            group_found = False

            # Nitro group directly attached: N+ with oxygens (PRIORITY CHECK)
            if n_symbol == 'N':
                oxygen_count = sum(1 for nn in n_atom.GetNeighbors()
                                 if nn.GetSymbol() == 'O')
                if oxygen_count >= 2:
                    alpha_groups.append('NO2')
                    group_found = True
                elif not group_found:
                    # Imine group: C=N or N=C (only if not already NO2)
                    for bond in n_atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            alpha_groups.append('C=N')
                            group_found = True
                            break

            # Check for carbonyl-based groups (CN, CO2R, COR)
            elif n_symbol == 'C':
                has_aromatic = n_atom.GetIsAromatic()

                if has_aromatic:
                    # Aromatic ring - count as one 'Ar' regardless of substituents
                    alpha_groups.append('Ar')
                    group_found = True

                    # Strategy 2: Check for EWG substituents on aromatic ring
                    if include_aromatic_substituents:
                        aromatic_subs = self._get_aromatic_substituents(mol, n_idx)
                        for sub in aromatic_subs:
                            alpha_groups.append(f'Ar-{sub}')
                else:
                    for nn in n_atom.GetNeighbors():
                        if nn.GetIdx() != carbanion_idx:
                            bond = mol.GetBondBetweenAtoms(n_idx, nn.GetIdx())

                            # Cyano group: C≡N
                            if nn.GetSymbol() == 'N' and bond.GetBondType() == Chem.BondType.TRIPLE:
                                alpha_groups.append('CN')
                                group_found = True

                            # Carbonyl groups: C=O
                            if nn.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE:
                                # Check if ester (has O-R) or ketone (no O-R)
                                has_OR = any(
                                    mol.GetBondBetweenAtoms(n_idx, nnn.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                                    for nnn in n_atom.GetNeighbors()
                                    if nnn.GetIdx() != carbanion_idx and nnn.GetSymbol() == 'O'
                                )
                                if has_OR:
                                    alpha_groups.append('CO2R')
                                else:
                                    alpha_groups.append('COR')
                                group_found = True

                    # If no EWG found on carbon, it's aliphatic
                    if not group_found:
                        alpha_groups.append('Aliphatic')
                        group_found = True

            # Sulfonyl group: SO2R
            elif n_symbol == 'S':
                oxygen_count = sum(1 for nn in n_atom.GetNeighbors() if nn.GetSymbol() == 'O')
                if oxygen_count >= 2:
                    alpha_groups.append('SO2R')
                    group_found = True

            # Halides
            elif n_symbol in ['F', 'Cl', 'Br', 'I']:
                alpha_groups.append('X')
                group_found = True

            # Other substituents (S, P, etc.)
            if not group_found and n_symbol not in ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S']:
                alpha_groups.append('Other')

        return alpha_groups

    def _get_aromatic_substituents(self, mol, aromatic_atom_idx):
        """
        Get EWG substituents on an aromatic ring (Strategy 2 helper).

        Args:
            mol: RDKit molecule object
            aromatic_atom_idx: Index of aromatic atom directly bonded to [C-]

        Returns:
            set: Set of EWG types found on the aromatic ring (e.g., {'NO2', 'CN', 'X'})
        """
        substituents = set()
        aromatic_atom = mol.GetAtomWithIdx(aromatic_atom_idx)

        # Get the aromatic ring this atom belongs to
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        # Find which ring(s) this aromatic atom is in
        for ring in atom_rings:
            if aromatic_atom_idx in ring:
                # Check if all atoms in ring are aromatic
                if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                    # Check substituents on each ring atom
                    for ring_atom_idx in ring:
                        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
                        for neighbor in ring_atom.GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            # Skip if neighbor is in the ring
                            if neighbor_idx in ring:
                                continue

                            n_symbol = neighbor.GetSymbol()

                            # Nitro group: N with 2+ oxygens
                            if n_symbol == 'N':
                                oxygen_count = sum(1 for nn in neighbor.GetNeighbors() if nn.GetSymbol() == 'O')
                                if oxygen_count >= 2:
                                    substituents.add('NO2')

                            # Cyano group attached to ring: C≡N
                            if n_symbol == 'C':
                                for nn in neighbor.GetNeighbors():
                                    if nn.GetIdx() != ring_atom_idx:
                                        bond = mol.GetBondBetweenAtoms(neighbor_idx, nn.GetIdx())
                                        if nn.GetSymbol() == 'N' and bond.GetBondType() == Chem.BondType.TRIPLE:
                                            substituents.add('CN')

                            # Halides on aromatic ring
                            if n_symbol in ['F', 'Cl', 'Br', 'I']:
                                substituents.add('X')

        return substituents

    def get_primary_group(self, smiles, priority=None):
        """
        Get the primary (highest priority) EWG group.

        Args:
            smiles: SMILES string
            priority: List of EWG types in priority order (highest first)
                     Default: ['NO2', 'SO2R', 'CN', 'CO2R', 'COR', 'C=N', 'Ar', 'F', 'Cl', 'Br', 'I']

        Returns:
            str or None: Primary EWG type
        """
        if priority is None:
            priority = ['NO2', 'SO2R', 'CN', 'CO2R', 'COR', 'C=N', 'Ar', 'X']

        groups = self.get_alpha_groups(smiles)

        for ewg in priority:
            if ewg in groups:
                return ewg
        return None

    def get_group_count(self, smiles):
        """
        Get the number of unique EWG alpha-substituent groups.

        Note: Excludes 'Aliphatic' and 'Other' as they are not EWGs.

        Args:
            smiles: SMILES string

        Returns:
            int: Number of unique EWG groups (0 if no EWGs)
        """
        groups = self.get_alpha_groups(smiles)
        ewg_groups = [g for g in groups if g not in ['Aliphatic', 'Other']]
        return len(set(ewg_groups))

    def classify_multiplicity(self, smiles):
        """
        Classify by number of unique EWG group types.

        Args:
            smiles: SMILES string

        Returns:
            str: 'single', 'double', 'triple', or 'none'
        """
        count = self.get_group_count(smiles)
        if count == 0:
            return 'none'
        elif count == 1:
            return 'single'
        elif count == 2:
            return 'double'
        else:
            return 'triple'

    def get_ewg_counts(self, smiles):
        """
        Get count of each EWG type as alpha substituent.

        Args:
            smiles: SMILES string

        Returns:
            dict: Counts for each EWG type (e.g., {'Ar': 1, 'CO2R': 2, 'CN': 0, ...})
        """
        groups = self.get_alpha_groups(smiles)
        group_counts = Counter(groups)

        # Return counts for all actual EWG types (exclude Aliphatic, Other)
        ewg_types = ['Ar', 'CN', 'CO2R', 'COR', 'C=N', 'NO2', 'SO2R', 'X']
        return {ewg: group_counts.get(ewg, 0) for ewg in ewg_types}

    def get_combination_string(self, smiles, separator=' + '):
        """
        Get a string representation of all EWG groups with multiplicity.

        Note: 'Aliphatic' is excluded as it's not an EWG, just the carbon backbone.

        Examples:
            ['CO2R', 'CO2R', 'Aliphatic'] -> 'CO2R, CO2R'
            ['Ar', 'NO2'] -> 'Ar + NO2'
            ['COR'] -> 'COR'
            ['Aliphatic'] -> '' (no EWGs)

        Args:
            smiles: SMILES string
            separator: String to separate different group types (default: ' + ')

        Returns:
            str: Formatted combination string (empty if no EWGs)
        """
        groups = self.get_alpha_groups(smiles)
        if not groups:
            return ''

        # Filter out non-EWG groups (Aliphatic, Other)
        ewg_groups = [g for g in groups if g not in ['Aliphatic', 'Other']]
        if not ewg_groups:
            return ''

        # Count each group type
        group_counts = Counter(ewg_groups)

        # Build sorted list with multiplicities
        result_parts = []
        for group in sorted(group_counts.keys()):
            count = group_counts[group]
            if count == 1:
                result_parts.append(group)
            else:
                # Repeat the group name: "CO2R, CO2R"
                result_parts.append(', '.join([group] * count))

        return separator.join(result_parts)

    def has_group(self, smiles, group):
        """
        Check if a molecule has a specific EWG group as an alpha substituent.

        Args:
            smiles: SMILES string
            group: EWG type string (e.g., 'CN', 'Ar')

        Returns:
            bool: True if group is present as alpha substituent
        """
        return group in self.get_alpha_groups(smiles)

    def has_ewg_in_molecule(self, smiles, ewg_type):
        """
        Check if a molecule contains a specific EWG group anywhere in the structure.

        This is different from has_group() which only checks alpha substituents.
        This method searches the entire molecule for the functional group.

        Args:
            smiles: SMILES string
            ewg_type: EWG type string (e.g., 'CN', 'Ar', 'CO2R', etc.)

        Returns:
            bool: True if the EWG is present anywhere in the molecule
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        if ewg_type == 'Ar':
            # Check for any aromatic atoms
            return any(atom.GetIsAromatic() for atom in mol.GetAtoms())

        elif ewg_type == 'CN':
            # Check for cyano group: C≡N
            pattern = Chem.MolFromSmarts('C#N')
            return mol.HasSubstructMatch(pattern)

        elif ewg_type == 'CO2R':
            # Check for ester group: C(=O)O[C,H]
            pattern = Chem.MolFromSmarts('C(=O)O[C,H]')
            return mol.HasSubstructMatch(pattern)

        elif ewg_type == 'COR':
            # Check for ketone: C(=O)[C,H] but not ester
            # Look for carbonyl with at least one C neighbor and no O-R
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    has_double_o = False
                    has_carbon_neighbor = False
                    has_single_o = False

                    for neighbor in atom.GetNeighbors():
                        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                        if neighbor.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE:
                            has_double_o = True
                        elif neighbor.GetSymbol() == 'C':
                            has_carbon_neighbor = True
                        elif neighbor.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.SINGLE:
                            has_single_o = True

                    # Ketone: has C=O, has C neighbor, no single-bonded O
                    if has_double_o and has_carbon_neighbor and not has_single_o:
                        return True
            return False

        elif ewg_type == 'C=N':
            # Check for imine: C=N or N=C
            pattern = Chem.MolFromSmarts('[C,N]=[N,C]')
            return mol.HasSubstructMatch(pattern)

        elif ewg_type == 'NO2':
            # Check for nitro group: N with 2+ oxygens
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N':
                    oxygen_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O')
                    if oxygen_count >= 2:
                        return True
            return False

        elif ewg_type == 'SO2R':
            # Check for sulfonyl group: S with 2+ oxygens
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'S':
                    oxygen_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O')
                    if oxygen_count >= 2:
                        return True
            return False

        elif ewg_type == 'X':
            # Check for any halide
            pattern = Chem.MolFromSmarts('[F,Cl,Br,I]')
            return mol.HasSubstructMatch(pattern)

        return False

    def classify_dataframe(self, df, smiles_column='smiles', include_aromatic_substituents=False):
        """
        Add classification columns to a DataFrame.

        Args:
            df: pandas DataFrame with SMILES column
            smiles_column: Name of the SMILES column
            include_aromatic_substituents: If True, use Strategy 2 (detect EWGs on aromatic rings)
                Default: False (Strategy 1: direct bonds only)

        Returns:
            DataFrame with added columns:
                - alpha_groups: set of unique EWG group types (as alpha substituents)
                - n_unique_ewg: number of unique EWG group types (as alpha substituents)
                - ewg_multiplicity: 'single'/'double'/'triple' (based on unique alpha EWG types)
                - combination: string representation of alpha substituents
                - substitution_degree: 'primary'/'secondary'/'tertiary' (based on C neighbors)
                - n_carbon_neighbors: number of C atoms on carbanion
                - n_aromatic_rings: number of aromatic rings in molecule
                - has_<EWG>: boolean for each EWG type (anywhere in molecule, not just alpha)
                - If include_aromatic_substituents=True, also adds has_Ar-<EWG> columns for alpha substituents
        """
        result = df.copy()

        # Core classification - unique EWG group types (ALPHA SUBSTITUENTS ONLY)
        result['alpha_groups'] = result[smiles_column].apply(
            lambda x: self.get_alpha_groups(x, include_aromatic_substituents=include_aromatic_substituents)
        )
        # n_unique_ewg: count only actual EWGs (exclude Aliphatic, Other)
        result['n_unique_ewg'] = result[smiles_column].apply(self.get_group_count)
        # ewg_multiplicity: based on EWG count only
        result['ewg_multiplicity'] = result[smiles_column].apply(self.classify_multiplicity)
        # combination: use get_combination_string which filters out non-EWGs
        result['combination'] = result[smiles_column].apply(self.get_combination_string)

        # Substitution degree (primary/secondary/tertiary)
        result['substitution_degree'] = result[smiles_column].apply(self.get_substitution_degree)
        result['n_carbon_neighbors'] = result[smiles_column].apply(self.count_carbon_neighbors)

        # Ring counts
        result['n_aromatic_rings'] = result[smiles_column].apply(self.count_aromatic_rings)

        # Individual EWG count columns (alpha substituents only)
        ewg_types = ['Ar', 'CN', 'CO2R', 'COR', 'C=N', 'NO2', 'SO2R', 'X']
        for ewg in ewg_types:
            result[f'n_{ewg}_alpha'] = result[smiles_column].apply(
                lambda x: self.get_ewg_counts(x).get(ewg, 0)
            )

        # Common pattern boolean flags
        result['is_pure_diester'] = (result['n_CO2R_alpha'] >= 2) & (result['n_unique_ewg'] == 1)
        result['is_mixed_diester'] = (result['n_CO2R_alpha'] >= 2) & (result['n_unique_ewg'] > 1)
        result['is_Ar_CO2R'] = (result['n_Ar_alpha'] >= 1) & (result['n_CO2R_alpha'] >= 1)
        result['is_Ar_CN'] = (result['n_Ar_alpha'] >= 1) & (result['n_CN_alpha'] >= 1)
        result['is_CN_CO2R'] = (result['n_CN_alpha'] >= 1) & (result['n_CO2R_alpha'] >= 1)
        result['is_Ar_NO2'] = (result['n_Ar_alpha'] >= 1) & (result['n_NO2_alpha'] >= 1)

        # Individual group flags - ANYWHERE in molecule (not just alpha substituents)
        for ewg in self.EWG_TYPES:
            result[f'has_{ewg}'] = result[smiles_column].apply(lambda x: self.has_ewg_in_molecule(x, ewg))

        # If using Strategy 2, also add flags for aromatic substituents (alpha only)
        if include_aromatic_substituents:
            for ewg in ['CN', 'NO2', 'X']:
                ar_ewg = f'Ar-{ewg}'
                result[f'has_{ar_ewg}'] = result['alpha_groups'].apply(lambda x: ar_ewg in x)

        return result

    def compute_cooccurrence_matrix(self, groups_series):
        """
        Compute co-occurrence matrix for EWG groups.

        Args:
            groups_series: Series/list of sets containing EWG groups

        Returns:
            numpy array: Co-occurrence matrix (diagonal = total count)
        """
        n = len(self.EWG_TYPES)
        cooccur = np.zeros((n, n))

        for groups in groups_series:
            for i, ewg1 in enumerate(self.EWG_TYPES):
                for j, ewg2 in enumerate(self.EWG_TYPES):
                    if ewg1 in groups and ewg2 in groups:
                        cooccur[i, j] += 1

        return cooccur

    def compute_jaccard_similarity(self, groups_series):
        """
        Compute Jaccard similarity between EWG groups.

        Args:
            groups_series: Series/list of sets containing EWG groups

        Returns:
            numpy array: Jaccard similarity matrix
        """
        n = len(self.EWG_TYPES)
        jaccard = np.zeros((n, n))

        for i, ewg1 in enumerate(self.EWG_TYPES):
            for j, ewg2 in enumerate(self.EWG_TYPES):
                set1 = set(idx for idx, groups in enumerate(groups_series) if ewg1 in groups)
                set2 = set(idx for idx, groups in enumerate(groups_series) if ewg2 in groups)
                if len(set1 | set2) > 0:
                    jaccard[i, j] = len(set1 & set2) / len(set1 | set2)

        return jaccard

    def get_combination_counts(self, groups_series):
        """
        Count occurrences of each unique EWG combination.

        Args:
            groups_series: Series/list of sets containing EWG groups

        Returns:
            Counter: Counts of each combination (as frozensets)
        """
        return Counter(frozenset(g) for g in groups_series)

    def summary_statistics(self, groups_series):
        """
        Generate summary statistics for EWG distribution.

        Args:
            groups_series: Series/list of sets containing EWG groups

        Returns:
            dict: Summary statistics including:
                - total_molecules
                - single/double/triple unique EWG counts and percentages
                - individual EWG counts and percentages
        """
        total = len(groups_series)
        stats = {'total_molecules': total}

        # Unique EWG type multiplicity distribution
        multiplicities = [len(g) for g in groups_series]
        for n, name in [(1, 'single_unique_ewg'), (2, 'double_unique_ewg'), (3, 'triple_unique_ewg')]:
            count = sum(1 for m in multiplicities if m == n)
            stats[f'{name}_count'] = count
            stats[f'{name}_pct'] = 100.0 * count / total if total > 0 else 0

        # Individual EWG distribution
        for ewg in self.EWG_TYPES:
            count = sum(1 for g in groups_series if ewg in g)
            stats[f'{ewg}_count'] = count
            stats[f'{ewg}_pct'] = 100.0 * count / total if total > 0 else 0

        return stats

    def highlight_carbanion(self, smiles, size=(300, 300)):
        """
        Create an image with the carbanion center highlighted.

        Args:
            smiles: SMILES string
            size: Tuple of (width, height)

        Returns:
            RDKit image or None if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        carbanion_idx = self.find_carbanion_center(mol)
        if carbanion_idx is None:
            return Draw.MolToImage(mol, size=size)

        # Highlight carbanion and its neighbors
        carbanion_atom = mol.GetAtomWithIdx(carbanion_idx)
        highlight_atoms = [carbanion_idx] + [n.GetIdx() for n in carbanion_atom.GetNeighbors()]

        return Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)


# Convenience function for quick use
def classify_carbanion(smiles, include_aromatic_substituents=False):
    """
    Quick classification of a single carbanion SMILES.

    Args:
        smiles: SMILES string
        include_aromatic_substituents: If True, use Strategy 2 (detect EWGs on aromatic rings)

    Returns:
        dict: Classification results
    """
    classifier = CarbanionClassifier()
    groups = classifier.get_alpha_groups(smiles, include_aromatic_substituents=include_aromatic_substituents)

    return {
        'smiles': smiles,
        'alpha_groups': groups,
        'n_unique_ewg': len(groups),
        'combination': ' + '.join(sorted(groups)),
        'ewg_multiplicity': classifier.classify_multiplicity(smiles),
        'substitution_degree': classifier.get_substitution_degree(smiles),
        'n_carbon_neighbors': classifier.count_carbon_neighbors(smiles),
        'strategy': 'Strategy 2' if include_aromatic_substituents else 'Strategy 1'
    }


if __name__ == '__main__':
    # Example usage
    print("CarbanionClassifier - Example Usage")
    print("=" * 50)

    # Test molecules
    test_smiles = [
        'c1ccccc1[C-](C#N)C(=O)OC',  # Ar + CN + CO2R
        'CC(=O)[C-](C#N)C(=O)OC',     # COR + CN + CO2R
        'c1ccccc1[C-](S(=O)(=O)C)C',  # Ar + SO2R
        '[O-][N+](=O)[C-](c1ccccc1)c2ccccc2',  # NO2 + Ar
        'F[C-](F)C(=O)OC',            # X + CO2R (difluoro - halides grouped as X)
        'Cl[C-](Br)c1ccccc1',         # X + Ar (mixed halides grouped as X)
    ]

    classifier = CarbanionClassifier()

    for smi in test_smiles:
        result = classify_carbanion(smi)
        print(f"\nSMILES: {smi}")
        print(f"  Unique EWG types: {result['alpha_groups']}")
        print(f"  Combination: {result['combination']}")
        print(f"  EWG multiplicity: {result['ewg_multiplicity']} ({result['n_unique_ewg']} unique types)")
        print(f"  Substitution degree: {result['substitution_degree']} ({result['n_carbon_neighbors']} C neighbors)")

    # Test with DataFrame
    print("\n" + "=" * 50)
    print("DataFrame Example:")
    df = pd.DataFrame({'smiles': test_smiles})
    df_classified = classifier.classify_dataframe(df)
    print(df_classified[['smiles', 'combination', 'n_unique_ewg', 'substitution_degree']].to_string())
