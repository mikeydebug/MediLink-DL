# MLMC Diffusion Layer Research Suite

This directory contains the formal research documentation, theoretical derivations, prior-art literature review, experimental benchmarks, and novelty assessments for the diffusion architecture design of the **MediLink Mini Cipher (MLMC-G16)** educational research candidate.

## Documentation Index

1. [**Research Objectives & Methodology**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/research_objectives.md)
   - Problem statement for MLMC-G16 educational cipher
   - Target metrics: branch numbers ($\mathcal{B}_{\text{bit}}$, $\mathcal{B}_{\text{nibble}}$), Paar CSE XOR complexity, hardware Gate Equivalents (GE), diffusion index ($R_{\text{diff}}$)
   - Scope and non-production research boundaries

2. [**Literature Review & Prior Art**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/literature_review.md)
   - Exhaustive survey of lightweight block cipher diffusion layers: AES, PRESENT, GIFT, RECTANGLE, Ascon, PHOTON/LED, and Involutory Hadamards (Sajadieh et al., Khoo et al.)
   - Authentic academic bibliography and taxonomy of techniques

3. [**Mathematical Foundations**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/mathematical_foundations.md)
   - Finite field $\mathbb{F}_{2^4}$ arithmetic modulo $P(x) = x^4 + x + 1$
   - Homomorphism mapping $\phi: \mathcal{M}_{4 \times 4}(\mathbb{F}_{2^4}) \to \mathcal{M}_{16 \times 16}(\mathbb{F}_2)$
   - Coding theory bounds: Singleton bound ($\mathcal{B}_{\text{nibble}} \le 5$) and Delsarte binary linear programming bound ($\mathcal{B}_{\text{bit}} \le 8$)
   - Wide-Trail Strategy (Daemen & Rijmen) and Matsui Piling-up Lemma

4. [**Candidate Architectural Designs**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/candidate_designs.md)
   - Formal mathematical definitions and matrices for all 11 evaluated candidates across Families A through F and baselines
   - Detailed specification of the primary candidate: **Orthogonal Involutory Nibble-MDS (`PROPOSED-OIN-MDS`)**

5. [**Experimental Results & Quantitative Evaluation**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/experimental_results.md)
   - Empirical and theoretical benchmark results table matching `experiments/diffusion/results/diffusion_benchmark_results.csv`
   - Detailed analysis: branch numbers, hardware Gate Equivalents, Paar CSE XOR reductions, multi-round diffusion speed, active S-boxes, and software latencies
   - Multi-objective Pareto frontier analysis

6. [**Novelty Assessment & Attribution Matrix**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/novelty_assessment.md)
   - 5-tier intellectual honesty taxonomy
   - Clear attribution: inherited classical techniques vs. proposed domain-specific co-optimizations
   - Unverified research hypotheses and empirical limitations

7. [**Known Limitations & Threat Model**](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/diffusion/limitations.md)
   - Critical medical notice: strictly non-production educational status
   - Theoretical bounds of 16-bit blocks and 16-bit keys
   - Side-channel vulnerabilities: software cache-timing and hardware DPA/CPA
   - Side-by-side comparison with production standards (AES-128-GCM, NIST SP 800-232 Ascon-128a)

## Source Code & Automated Test Suite

- **Implementation**: [`src/crypto_research/diffusion/`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/crypto_research/diffusion/)
- **Automated Verification**: [`tests/test_invertibility.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_invertibility.py), [`tests/test_inverse.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_inverse.py), [`tests/test_branch_number.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_branch_number.py), [`tests/test_properties.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_properties.py)
- **Raw Benchmark Artifacts**: [`experiments/diffusion/results/diffusion_benchmark_results.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/experiments/diffusion/results/diffusion_benchmark_results.csv)

