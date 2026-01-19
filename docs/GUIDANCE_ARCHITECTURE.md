# BoltzGen Property Steering Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BOLTZGEN DIFFUSION                                  │
│                                                                                 │
│   User Request: "Design a protein with HIGH hydrophobicity"                     │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  DIFFUSION LOOP (50 steps: noisy → clean structure)                      │  │
│   │                                                                          │  │
│   │    Step 0      Step 1      Step 2           ...        Step 49           │  │
│   │   ┌─────┐     ┌─────┐     ┌─────┐                     ┌─────┐            │  │
│   │   │noise│ ──► │     │ ──► │     │ ──► ─── ─── ─── ──► │clean│            │  │
│   │   └─────┘     └─────┘     └─────┘                     └─────┘            │  │
│   │       │           │           │                           │              │  │
│   │       ▼           ▼           ▼                           ▼              │  │
│   │   ┌───────────────────────────────────────────────────────────┐          │  │
│   │   │            GUIDANCE APPLIED AT EACH STEP                  │          │  │
│   │   │   Nudge coordinates toward higher hydrophobicity          │          │  │
│   │   └───────────────────────────────────────────────────────────┘          │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   Output: Protein structure with hydrophobic sequence                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## The Key Innovation: Geometric Residue Encoding

BoltzGen encodes amino acid identity **geometrically** using virtual atoms. Each residue has 14 atom positions, and sidechain atoms "collapse" onto backbone atoms in patterns unique to each amino acid:

```
                    GEOMETRIC ENCODING OF AMINO ACIDS
    ═══════════════════════════════════════════════════════════════

    Each residue = 14 atoms: [N, CA, C, O] backbone + 10 sidechain atoms
    
    Sidechain atoms COLLAPSE to nearest backbone atom:
    
    ┌────────────────────────────────────────────────────────────────┐
    │   GLYCINE (GLY)              LEUCINE (LEU)                     │
    │   Small - all collapse       Large - sidechains spread out     │
    │                                                                │
    │      N ●──────● CA              N ●──────● CA                  │
    │        \      /                   \      / \                   │
    │         \    /                     \    /   ● ← sidechain      │
    │          \  /                       \  /    │                  │
    │     O ●───● C                  O ●───● C    ● ← sidechain      │
    │                                             │                  │
    │   Count: [~10, ~0, ~0, ~0]                  ● ← sidechain      │
    │   (all sidechains on N)                                        │
    │                              Count: [2, 3, 2, 3]               │
    │                              (spread across backbone)          │
    └────────────────────────────────────────────────────────────────┘
    
    The COUNT PATTERN uniquely identifies each of the 20 amino acids!
```

## Guidance Mechanism: Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ONE DIFFUSION STEP WITH GUIDANCE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: Noisy atom coordinates x_t                                              │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: NETWORK FORWARD PASS                                             │  │
│  │                                                                           │  │
│  │     x_t (noisy)  ──►  [ Boltz Network ]  ──►  x_0 (denoised prediction)   │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: GEOMETRIC DECODING (differentiable)                              │  │
│  │                                                                           │  │
│  │  For each designed residue:                                               │  │
│  │                                                                           │  │
│  │    14 atom coords  ──►  Compute distances  ──►  Count pattern  ──►  AA    │  │
│  │    [N,CA,C,O,...]       sidechain → backbone    [n1,n2,n3,n4]     probs   │  │
│  │                                                                           │  │
│  │                         ┌─────────────────┐                               │  │
│  │    Distances:           │ Match to known  │     Soft AA probabilities:    │  │
│  │    d(SC_i, BB_j)  ──►   │ AA patterns     │ ──►  P(ALA), P(LEU), ...     │  │
│  │                         │ via softmax     │                               │  │
│  │                         └─────────────────┘                               │  │
│  │                                                                           │  │
│  │    Uses Straight-Through Estimator (STE) for gradient flow               │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: PROPERTY SCORING                                                 │  │
│  │                                                                           │  │
│  │    AA probs  ──►  [ Hydrophobicity Scale ]  ──►  Score                    │  │
│  │                                                                           │  │
│  │    score = Σ P(aa_i) × hydrophobicity(aa_i)                              │  │
│  │                                                                           │  │
│  │    Kyte-Doolittle scale:                                                  │  │
│  │      ILE: +4.5  VAL: +4.2  LEU: +3.8  (hydrophobic)                      │  │
│  │      ARG: -4.5  LYS: -3.9  ASP: -3.5  (hydrophilic)                      │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: BACKPROPAGATION                                                  │  │
│  │                                                                           │  │
│  │    loss = -score  (we want to MAXIMIZE hydrophobicity)                    │  │
│  │                                                                           │  │
│  │    ∂loss/∂x_0 = gradient telling us how to move atoms                     │  │
│  │                 to increase hydrophobicity                                │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: COORDINATE UPDATE                                                │  │
│  │                                                                           │  │
│  │    x_{t-1} = normal_diffusion_update(x_t, x_0)                            │  │
│  │                                                                           │  │
│  │            - scale × schedule_weight × σ_t × ∂loss/∂x_0                   │  │
│  │              ▲                                  ▲                          │  │
│  │              │                                  │                          │  │
│  │         guidance                           gradient                        │  │
│  │         strength                        (from backprop)                    │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  OUTPUT: x_{t-1} (slightly less noisy, nudged toward desired property)          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Code Integration Points

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CODE ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   USER CODE                                                                     │
│   ─────────                                                                     │
│   guidance = create_geometric_guidance(                                         │
│       property_type="hydrophobicity",                                           │
│       guidance_scale=50.0,                                                      │
│   )                                                                             │
│   out = model.forward(feats=feats, guidance=guidance, ...)                      │
│                       │                                                         │
│                       ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  boltz.py: Boltz.forward()                                              │   │
│   │                                                                         │   │
│   │    Passes guidance to structure_module.sample()                         │   │
│   └───────────────────────────────┬─────────────────────────────────────────┘   │
│                                   │                                             │
│                                   ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  diffusion.py: AtomDiffusion.sample()                                   │   │
│   │                                                                         │   │
│   │    for step in diffusion_steps:                                         │   │
│   │        if guidance.enabled:                                             │   │
│   │            # Forward pass with gradients                                │   │
│   │            score = guidance.compute_geometric_score(coords, feats)      │   │
│   │            loss = -score                                                │   │
│   │            loss.backward()                                              │   │
│   │            coord_grad = coords.grad                                     │   │
│   │            coords -= scale * coord_grad  # Apply guidance               │   │
│   │                          │                                              │   │
│   └──────────────────────────┼──────────────────────────────────────────────┘   │
│                              │                                                  │
│                              ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  guidance.py: GeometricGuidance.compute_geometric_score()               │   │
│   │                                                                         │   │
│   │    1. Extract 14 atoms per residue                                      │   │
│   │    2. Compute sidechain→backbone distances                              │   │
│   │    3. Count atoms closest to each backbone atom                         │   │
│   │    4. Match count pattern to AA type (with STE)                         │   │
│   │    5. Compute weighted property score                                   │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Straight-Through Estimator (STE)

The key challenge: amino acid selection is **discrete** (hard argmin), but we need **gradients** to flow.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STRAIGHT-THROUGH ESTIMATOR                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   FORWARD PASS: Use hard (discrete) decisions                                   │
│   ─────────────                                                                 │
│                                                                                 │
│   distances = [0.3, 0.8, 0.1, 0.5]  ──►  argmin = 2  ──►  one_hot = [0,0,1,0]  │
│                                                                                 │
│   BACKWARD PASS: Pretend we used soft (continuous) decisions                    │
│   ──────────────                                                                │
│                                                                                 │
│   soft_assignment = softmax(-distances/τ) = [0.25, 0.05, 0.65, 0.05]           │
│                                                                                 │
│   Gradients flow through the soft version!                                      │
│                                                                                 │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │  IMPLEMENTATION:                                                       │     │
│   │                                                                       │     │
│   │  hard = one_hot(argmin(distances))           # No gradients           │     │
│   │  soft = softmax(-distances / temperature)    # Has gradients          │     │
│   │                                                                       │     │
│   │  output = hard + (soft - soft.detach())      # STE trick!             │     │
│   │           ▲                                                           │     │
│   │           │                                                           │     │
│   │    Forward: hard    Backward: soft                                    │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Guidance Schedule

Guidance strength varies over diffusion progress:

```
   GUIDANCE WEIGHT OVER DIFFUSION
   
   weight │
     1.0  │                          ┌──────────────── constant
          │                         ╱
     0.8  │                        ╱      ╭──────── cosine  
          │                       ╱      ╱
     0.6  │                      ╱      ╱
          │                     ╱      ╱
     0.4  │                    ╱    ╱
          │                   ╱   ╱
     0.2  │                  ╱  ╱
          │                 ╱ ╱
     0.0  │────────────────╱╱
          └─────────────────────────────────────────────────────
          0.0    0.2    0.4    0.6    0.8    1.0
                         diffusion progress ──►
          
          early steps         │        later steps
          (noisy, guidance    │        (clean, guidance
           has little effect) │         steers effectively)
```

## Summary: What Changed in Boltz

| File | Change |
|------|--------|
| `boltz.py` | Added `guidance` parameter to `forward()` |
| `diffusion.py` | Added guidance gradient computation in sampling loop |
| `guidance/guidance.py` | **NEW** - GeometricGuidance class with differentiable AA decoding |
| `guidance/__init__.py` | **NEW** - Package exports |

The beauty of this approach: **no retraining required**. We leverage the existing geometric encoding to decode amino acid identity, compute property scores, and steer via classifier-free guidance.
