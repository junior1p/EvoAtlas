"""EvoAtlas: Cross-Scale Evolutionary Pressure Landscape Reconstruction Engine."""
__version__ = "0.1.0"

from .data import download_demo_spike_sequences, align_sequences
from .phylogeny import compute_distance_matrix, neighbor_joining, hky85_Q_matrix
from .dnds import (
    compute_site_omega_fast,
    compute_nucleotide_diversity,
    compute_tajimas_d,
    compute_fu_li_F,
)
from .epistasis import compute_mutual_information_matrix, walsh_hadamard_epistasis
from .viz import create_evoatlas_landscape

import os, json, numpy as np, pandas as pd
from collections import Counter


def run_evoatlas(
    input_source=None,
    n_seqs: int = 80,
    out_dir: str = "evoatlas_output",
    window_size: int = 30,
    mi_max_sites: int = 120,
    fast_mode: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(" EvoAtlas v" + __version__ + " — Evolutionary Pressure Landscape")
    print(" CPU-only | HKY85 | Felsenstein | MI | WHT")
    print("=" * 60)

    # Step 1: data
    if input_source is None:
        records = download_demo_spike_sequences(n=n_seqs, out_dir=f"{out_dir}/data")
    else:
        records = input_source[:n_seqs]

    # Step 2: alignment
    msa_matrix, seq_ids = align_sequences(records)
    n_seqs_actual, align_len = msa_matrix.shape

    # Step 3: omega
    omega = compute_site_omega_fast(msa_matrix, seq_ids, None)
    print(f"\n[Layer 2] ω computed for {align_len} positions")

    # Step 4: population genetics
    print("\n[Layer 3] Tajima's D...")
    tajd_pos, tajd_vals = compute_tajimas_d(msa_matrix, window=window_size,
                                            step=max(1, window_size // 3))
    print("\n[Layer 3] Fu & Li F*...")
    fu_li_F = compute_fu_li_F(msa_matrix)

    # Step 5: MI
    print("\n[Layer 4] Mutual information...")
    mi_sites, MI_matrix = compute_mutual_information_matrix(
        msa_matrix, max_sites=mi_max_sites)

    # Step 6: WHT
    print("\n[Layer 4] WHT epistasis decomposition...")
    if len(mi_sites) >= 4:
        consensus = [Counter(msa_matrix[:, col]).most_common(1)[0][0]
                     for col in mi_sites[:10]]
        n_wht = min(10, len(mi_sites))
        freq_profile = np.zeros(2**n_wht)
        for row in range(n_seqs_actual):
            idx = 0
            for bit, col in enumerate(mi_sites[:n_wht]):
                if msa_matrix[row, col] != consensus[bit]:
                    idx |= (1 << bit)
            freq_profile[idx] += 1
        freq_profile /= (freq_profile.sum() + 1e-10)
        wht_result = walsh_hadamard_epistasis(freq_profile)
    else:
        wht_result = {"additive_fraction": 1.0, "pairwise_fraction": 0.0,
                       "higher_fraction": 0.0, "n_sites": 0}

    print(f"  WHT: additive={wht_result['additive_fraction']*100:.1f}%  "
          f"pairwise={wht_result['pairwise_fraction']*100:.1f}%  "
          f"higher={wht_result['higher_fraction']*100:.1f}%")

    # Step 7: visualization
    print("\n[Output] Building Plotly landscape...")
    create_evoatlas_landscape(
        omega=omega,
        tajd_positions=tajd_pos,
        tajd_values=tajd_vals,
        mi_sites=mi_sites,
        MI_matrix=MI_matrix,
        fu_li_F=fu_li_F,
        wht_result=wht_result,
        n_seqs=n_seqs_actual,
        align_len=align_len,
        output_path=f"{out_dir}/evoatlas_landscape.html",
    )

    # Step 8: structured outputs
    _save_outputs(omega, tajd_pos, tajd_vals, mi_sites, MI_matrix,
                   fu_li_F, wht_result, seq_ids, out_dir, align_len)

    print(f"\n{'='*60}")
    print(" EvoAtlas Summary")
    print(f"{'='*60}")
    print(f" {'Sequences':<40} {n_seqs_actual}")
    print(f" {'Alignment length':<40} {align_len}")
    print(f" {'High-variability sites (ω>0.6)':<40} "
          f"{int((np.clip(omega,0,5)>0.6).sum())}")
    print(f" {'Mean ω':<40} {float(np.mean(omega)):.4f}")
    print(f" {'Tajima D mean':<40} {float(np.mean(tajd_vals)):.4f}")
    print(f" {'Fu & Li F*':<40} {fu_li_F:.4f}")
    print(f" {'WHT additive':<40} {wht_result['additive_fraction']*100:.1f}%")
    print(f" {'WHT pairwise':<40} {wht_result['pairwise_fraction']*100:.1f}%")
    print(f"{'='*60}")
    return {"omega": omega, "tajd": (tajd_pos, tajd_vals),
            "MI": MI_matrix, "wht": wht_result}


def _save_outputs(omega, tajd_pos, tajd_vals, mi_sites, MI_matrix,
                  fu_li_F, wht_result, seq_ids, out_dir, align_len):
    n_pos = len(omega)
    df = pd.DataFrame({
        "position":      np.arange(1, n_pos + 1),
        "omega_proxy":   omega,
        "selection":     ["high_var" if w > 0.6 else "conserved" if w < 0.3
                          else "moderate" for w in np.clip(omega, 0, 5)],
    })
    if len(tajd_pos) > 0:
        tajd_interp = np.interp(np.arange(n_pos), tajd_pos, tajd_vals)
        df["tajimas_D"] = tajd_interp
    df.to_csv(f"{out_dir}/per_site_stats.csv", index=False)

    if len(mi_sites) > 1:
        mi_df = pd.DataFrame(MI_matrix, index=mi_sites, columns=mi_sites)
        mi_df.to_csv(f"{out_dir}/mutual_information_matrix.csv")

    summary = {
        "n_sequences":               len(seq_ids),
        "alignment_length":          n_pos,
        "mean_omega":                float(np.nanmean(omega)),
        "high_variability_sites":    int((np.clip(omega, 0, 5) > 0.6).sum()),
        "fu_li_F":                   float(fu_li_F),
        "tajimas_D_mean":            float(np.mean(tajd_vals)) if len(tajd_vals) > 0 else 0.0,
        "wht_additive_fraction":     float(wht_result.get("additive_fraction", 0)),
        "wht_pairwise_fraction":     float(wht_result.get("pairwise_fraction", 0)),
        "wht_higher_fraction":      float(wht_result.get("higher_fraction", 0)),
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs saved to: {out_dir}/")
