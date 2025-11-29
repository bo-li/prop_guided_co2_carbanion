#!/usr/bin/env python
"""
Finetune hgraph2graph generator for carbanion generation using hybrid approach.
Uses Chemprop with solvent features to predict N and sN, then calculates log_k for CO2 reactivity.
Filters by log_k window AND regex pattern to ensure only true carbanions are generated.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import math, random, sys, re
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

# Add parent directory to path for hgraph imports
sys.path.insert(0, './hgraph2graph')
import hgraph
from hgraph import HierVAE, common_atom_vocab, PairVocab

# Import chemprop
from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader, MoleculeDatapoint
from chemprop.utils import load_checkpoint, load_scalers

# Import solvent features function
from generate_solvent_features import get_solvent_features

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))


def safe_to_numpy(x):
    """Convert tensor/array to numpy safely."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (np.ndarray, np.generic)):
        return np.array(x)
    if isinstance(x, (int, float)):
        return x
    return x


def is_carbanion(smiles):
    """Check if SMILES contains exactly one carbanion pattern [C-], [CH-], or [CH2-]"""
    matches = re.findall(r'\[(?:C|CH|CH2)-\]', smiles)
    return len(matches) == 1  # Only mono-anions


class CO2ChempropPredictor:
    """
    Chemprop predictor for N and sN parameters with configurable solvent.
    Calculates log_k for CO2 reactivity from predictions.
    """

    def __init__(self, checkpoint_dir, solvent='DMSO'):
        """
        Args:
            checkpoint_dir: Path to Chemprop model checkpoints
            solvent: Solvent name (DMSO, MeCN, water, etc.)
        """
        self.solvent = solvent
        self.solvent_features = get_solvent_features(solvent)

        print(f"✅ Target solvent: {solvent}")
        print(f"✅ Solvent features: {len(self.solvent_features)} dimensions")

        # Load model checkpoints
        self.checkpoints, self.scalers, self.features_scalers = [], [], []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    fpath = os.path.join(root, fname)
                    scaler, features_scaler = load_scalers(fpath)
                    self.scalers.append(scaler)
                    self.features_scalers.append(features_scaler)
                    model = load_checkpoint(fpath)
                    self.checkpoints.append(model)

        if len(self.checkpoints) == 0:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

        print(f"✅ Loaded {len(self.checkpoints)} model checkpoints")

    def predict(self, smiles_list, batch_size=500):
        """
        Predict N and sN for a list of SMILES.

        Returns:
            Tuple of (N_predictions, sN_predictions) as numpy arrays
        """
        # Create features matrix: same solvent features for each SMILES
        features_matrix = [self.solvent_features for _ in smiles_list]

        # Create MoleculeDatapoint objects
        test_data = []
        for i, smi in enumerate(smiles_list):
            datapoint = MoleculeDatapoint(
                smiles=[smi],
                targets=None,
                features=features_matrix[i]
            )
            test_data.append(datapoint)

        # Filter valid molecules
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol[0] is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])
        test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=batch_size)

        # Ensemble predictions (returns [N, sN] for each molecule)
        sum_preds = np.zeros((len(test_data), 2))
        for model, scaler, features_scaler in zip(self.checkpoints, self.scalers, self.features_scalers):
            test_data.reset_features_and_targets()
            if features_scaler is not None:
                test_data.normalize_features(features_scaler)

            model_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            sum_preds += np.array(model_preds)

        # Average across ensemble
        avg_preds = sum_preds / len(self.checkpoints)

        # Put zeros for invalid SMILES
        full_preds = np.zeros((len(full_data), 2))
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]

        N_preds = full_preds[:, 0]
        sN_preds = full_preds[:, 1]

        return N_preds, sN_preds

    @staticmethod
    def calculate_logk_CO2(N, sN, E_CO2=-14.6, sE=0.81):
        """Calculate log_k for CO2 reaction using Mayr equation."""
        return sE * sN * (E_CO2 + N)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Training SMILES file (CO2-active seeds)')
    parser.add_argument('--vocab', required=True, help='Motif vocabulary file')
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--save_dir', required=True, help='Directory to save checkpoints and results')
    parser.add_argument('--generative_model', required=True, help='Pretrained generator checkpoint')
    parser.add_argument('--chemprop_model', required=True, help='Chemprop model directory')
    parser.add_argument('--seed', type=int, default=7)

    # Solvent configuration
    parser.add_argument('--solvent', type=str, default='DMSO',
                       help='Target solvent (DMSO, MeCN, water, dichloromethane, THF, MeOH)')

    # CO2 reactivity window
    parser.add_argument('--logk_min', type=float, default=-6.0, help='Minimum log_k for CO2 reactivity')
    parser.add_argument('--logk_max', type=float, default=8.0, help='Maximum log_k for CO2 reactivity')

    # Model architecture
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--inner_epoch', type=int, default=10)
    parser.add_argument('--min_similarity', type=float, default=0.1, help='Minimum Tanimoto similarity to seeds')
    parser.add_argument('--max_similarity', type=float, default=0.5, help='Maximum Tanimoto similarity to seeds')
    parser.add_argument('--nsample', type=int, default=10000, help='Number of molecules to generate per epoch')
    parser.add_argument('--generation_batch_size', type=int, default=5, help='Batch size for generation')

    args = parser.parse_args()
    print("="*80)
    print("MONO-ANION CARBANION GENERATION WITH HYBRID APPROACH")
    print("="*80)
    print(f"Method: log_k predictor + regex filter")
    print(f"Solvent: {args.solvent}")
    print(f"log_k window: [{args.logk_min}, {args.logk_max}]")
    print(f"Carbanion pattern: [C-], [CH-], [CH2-] (exactly 1 per molecule)")
    print(f"Similarity range: [{args.min_similarity}, {args.max_similarity}]")
    print(f"Samples per epoch: {args.nsample}")
    print("="*80)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load training seeds
    with open(args.train) as f:
        train_smiles = [line.strip("\r\n ") for line in f]
    print(f"Loaded {len(train_smiles)} carbanion seed molecules")

    # Sanity check: verify all seeds are connected molecules
    print("Performing sanity check on seed molecules...")
    invalid_seeds = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_seeds.append(smi)
            continue
        frags = Chem.GetMolFrags(mol, asMols=False)
        if len(frags) > 1:
            invalid_seeds.append(smi)

    if invalid_seeds:
        print(f"❌ ERROR: Found {len(invalid_seeds)} disconnected/invalid molecules in seeds!")
        print("Please regenerate seeds with connected molecules only.")
        print(f"First few invalid: {invalid_seeds[:3]}")
        sys.exit(1)
    print("✅ All seed molecules are connected and valid")

    # Load vocabulary
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
    args.vocab = PairVocab(vocab)

    # Initialize Chemprop predictor
    predictor = CO2ChempropPredictor(
        checkpoint_dir=args.chemprop_model,
        solvent=args.solvent
    )

    # Precompute fingerprints for similarity calculation
    good_smiles = train_smiles
    train_mol = [Chem.MolFromSmiles(s) for s in train_smiles]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in train_mol]

    # Initialize generator model
    model = HierVAE(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load pretrained weights
    print(f'Loading pretrained model from {args.generative_model}')
    model_state, optimizer_state, _, beta = torch.load(args.generative_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Finetuning loop
    for epoch in range(args.epoch):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}")
        print(f"{'='*80}")

        # Update training set with unique molecules
        good_smiles = sorted(set(good_smiles))
        random.shuffle(good_smiles)
        dataset = hgraph.MoleculeDataset(good_smiles, args.vocab, args.atom_vocab, args.batch_size)
        print(f"Training set size: {len(good_smiles)} molecules")

        # Training phase
        print(f'\nTraining generator...')
        for inner in range(args.inner_epoch):
            meters = np.zeros(6)
            dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x:x[0], shuffle=True, num_workers=16)
            for batch in tqdm(dataloader, desc=f'Inner epoch {inner+1}/{args.inner_epoch}'):
                model.zero_grad()
                loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                meters = meters + np.array([kl_div, loss.item(),
                        safe_to_numpy(wacc) * 100, safe_to_numpy(iacc) * 100,
                        safe_to_numpy(tacc) * 100, safe_to_numpy(sacc) * 100])

            meters /= len(dataset)
            print(f"  Beta: {beta:.3f}, KL: {meters[0]:.2f}, Loss: {meters[1]:.3f}, "
                  f"Word: {meters[2]:.2f}/{meters[3]:.2f}, Topo: {meters[4]:.2f}, "
                  f"Assm: {meters[5]:.2f}, PNorm: {param_norm(model):.2f}, GNorm: {grad_norm(model):.2f}")

        # Save checkpoint
        ckpt = (model.state_dict(), optimizer.state_dict(), epoch, beta)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{epoch}"))
        print(f"✅ Saved checkpoint: model.ckpt.{epoch}")

        # Generation phase
        print(f'\nGenerating {args.nsample} molecules...')
        decoded_smiles = []
        with torch.no_grad():
            safe_batch_size = min(args.generation_batch_size, args.batch_size)
            total_batches = args.nsample // safe_batch_size

            for batch_idx in tqdm(range(total_batches), desc='Generating'):
                try:
                    outputs = model.sample(safe_batch_size, greedy=True)
                    decoded_smiles.extend(outputs)
                except (RuntimeError, KeyError, ValueError) as e:
                    # Handle vocabulary errors, tensor errors, etc.
                    error_type = type(e).__name__
                    if batch_idx % 100 == 0:  # Only print occasionally to avoid spam
                        print(f'\n{error_type} in batch {batch_idx}, falling back to batch size 1')

                    # Try generating one-by-one
                    for _ in range(safe_batch_size):
                        try:
                            outputs = model.sample(1, greedy=True)
                            decoded_smiles.extend(outputs)
                        except (RuntimeError, KeyError, ValueError):
                            # Skip this single sample if it still fails
                            continue
                except Exception as e:
                    # Catch-all for unexpected errors
                    print(f'\nUnexpected error in batch {batch_idx}: {type(e).__name__}: {str(e)[:100]}')
                    continue

        print(f"Generated {len(decoded_smiles)} molecules")

        # Prediction phase
        print(f'\nPredicting N and sN with Chemprop...')
        N_preds, sN_preds = predictor.predict(decoded_smiles)

        # Calculate log_k for CO2
        logk_values = np.array([
            predictor.calculate_logk_CO2(N, sN)
            for N, sN in zip(N_preds, sN_preds)
        ])

        # Filter by log_k window
        active_mask = (logk_values >= args.logk_min) & (logk_values <= args.logk_max)
        outputs = [(s, N, sN, logk)
                   for s, N, sN, logk, is_active in zip(decoded_smiles, N_preds, sN_preds, logk_values, active_mask)
                   if is_active]
        print(f'✅ Discovered {len(outputs)} molecules with log_k ∈ [{args.logk_min}, {args.logk_max}]')

        # Apply mono-anion regex filter (HYBRID APPROACH)
        outputs_carbanion = [(s, N, sN, logk) for s, N, sN, logk in outputs if is_carbanion(s)]
        carb_pct = 100 * len(outputs_carbanion) / len(outputs) if len(outputs) > 0 else 0
        print(f'✅ Mono-anion filter: {len(outputs_carbanion)}/{len(outputs)} ({carb_pct:.1f}%) are mono-anion carbanions')
        outputs = outputs_carbanion

        # Filter by similarity (and sanity check for connected molecules)
        novel_entries = []
        good_entries = []
        disconnected_count = 0
        for s, N, sN, logk in outputs:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue

            # Sanity check: skip disconnected molecules
            frags = Chem.GetMolFrags(mol, asMols=False)
            if len(frags) > 1:
                disconnected_count += 1
                continue

            fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps, train_fps))
            max_sim = sims.max()

            good_entries.append((s, N, sN, logk, max_sim))

            if args.min_similarity <= max_sim <= args.max_similarity:
                novel_entries.append((s, N, sN, logk, max_sim))
                good_smiles.append(s)

        if disconnected_count > 0:
            print(f'⚠️  Filtered out {disconnected_count} disconnected molecules from generated set')

        print(f'✅ Discovered {len(novel_entries)} NOVEL mono-anion carbanion molecules')
        print(f'   (Similarity range: [{args.min_similarity}, {args.max_similarity}])')

        # Mono-anion statistics
        carb_good = sum(1 for s, _, _, _, _ in good_entries if is_carbanion(s))
        carb_novel = sum(1 for s, _, _, _, _ in novel_entries if is_carbanion(s))
        print(f'   Mono-anions in all active: {carb_good}/{len(good_entries)} ({100*carb_good/len(good_entries) if len(good_entries) > 0 else 0:.1f}%)')
        print(f'   Mono-anions in novel: {carb_novel}/{len(novel_entries)} ({100*carb_novel/len(novel_entries) if len(novel_entries) > 0 else 0:.1f}%)')

        # Save novel molecules
        with open(os.path.join(args.save_dir, f"new_molecules.{epoch}"), 'w') as f:
            f.write("SMILES\tN\tsN\tlogk_CO2\tmax_similarity\n")
            for s, N, sN, logk, sim in novel_entries:
                f.write(f"{s}\t{N:.4f}\t{sN:.4f}\t{logk:.4f}\t{sim:.4f}\n")

        # Save all active molecules
        with open(os.path.join(args.save_dir, f"good_molecules.{epoch}"), 'w') as f:
            f.write("SMILES\tN\tsN\tlogk_CO2\tmax_similarity\n")
            for s, N, sN, logk, sim in good_entries:
                f.write(f"{s}\t{N:.4f}\t{sN:.4f}\t{logk:.4f}\t{sim:.4f}\n")

        # Statistics
        if len(novel_entries) > 0:
            logk_novel = [logk for _, _, _, logk, _ in novel_entries]
            print(f"\n📊 Novel molecules statistics:")
            print(f"   log_k range: [{min(logk_novel):.2f}, {max(logk_novel):.2f}]")
            print(f"   log_k mean: {np.mean(logk_novel):.2f} ± {np.std(logk_novel):.2f}")

    print(f"\n{'='*80}")
    print("FINETUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved in: {args.save_dir}")
