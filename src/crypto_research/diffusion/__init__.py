"""
src/crypto_research/diffusion/__init__.py

Package initialization for MLMC Diffusion Architecture Research Suite.
"""

from .matrix_validation import (
    GF16,
    gf16,
    matmul_gf2,
    matvec_gf2,
    rank_gf2,
    is_invertible_gf2,
    inverse_gf2,
    matmul_gf16,
    det_gf16,
    inverse_gf16,
    gf16_matrix_to_binary_matrix,
    is_involutory_gf2,
    is_orthogonal_gf2,
)
from .branch_number import compute_exact_branch_numbers
from .xor_cost import (
    compute_direct_xor_cost,
    compute_paar_optimized_xor_cost,
    estimate_hardware_ge,
)
from .inverse_generator import generate_and_verify_inverse
from .matrix_generator import (
    DiffusionCandidate,
    get_all_candidates,
    generate_baseline_mlmc_g16,
    generate_baseline_present_16,
    generate_baseline_gift_16,
    generate_family_a_sylvester_hadamard,
    generate_family_b_cauchy_mds,
    generate_family_c_involutory_hadamard_opt,
    generate_family_c_involutory_hadamard_maxbit,
    generate_family_d_circulant_mds,
    generate_family_e_recursive_companion_mds,
    generate_family_f_hybrid_permutation_xor,
    generate_family_f_oin_mds,
)
from .diffusion_analysis import (
    compute_diffusion_rounds_index,
    evaluate_candidate_avalanche,
)
from .differential_analysis import (
    compute_sbox_ddt,
    evaluate_active_sboxes_bounds,
)
from .linear_analysis import (
    compute_sbox_lat,
    evaluate_linear_trails_bounds,
)
from .benchmark import (
    benchmark_software_throughput,
    run_comprehensive_diffusion_benchmark,
)

__all__ = [
    "GF16",
    "gf16",
    "matmul_gf2",
    "matvec_gf2",
    "rank_gf2",
    "is_invertible_gf2",
    "inverse_gf2",
    "matmul_gf16",
    "det_gf16",
    "inverse_gf16",
    "gf16_matrix_to_binary_matrix",
    "is_involutory_gf2",
    "is_orthogonal_gf2",
    "compute_exact_branch_numbers",
    "compute_direct_xor_cost",
    "compute_paar_optimized_xor_cost",
    "estimate_hardware_ge",
    "generate_and_verify_inverse",
    "DiffusionCandidate",
    "get_all_candidates",
    "generate_baseline_mlmc_g16",
    "generate_baseline_present_16",
    "generate_baseline_gift_16",
    "generate_family_a_sylvester_hadamard",
    "generate_family_b_cauchy_mds",
    "generate_family_c_involutory_hadamard_opt",
    "generate_family_c_involutory_hadamard_maxbit",
    "generate_family_d_circulant_mds",
    "generate_family_e_recursive_companion_mds",
    "generate_family_f_hybrid_permutation_xor",
    "generate_family_f_oin_mds",
    "compute_diffusion_rounds_index",
    "evaluate_candidate_avalanche",
    "compute_sbox_ddt",
    "evaluate_active_sboxes_bounds",
    "compute_sbox_lat",
    "evaluate_linear_trails_bounds",
    "benchmark_software_throughput",
    "run_comprehensive_diffusion_benchmark",
]

