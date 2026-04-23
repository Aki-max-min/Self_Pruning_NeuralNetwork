# Self-Pruning Neural Network on CIFAR-10

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/self-pruning-neural-network-cifar10/blob/main/self_pruning_cifar10.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## Overview

This project implements a **self-pruning neural network** for CIFAR-10 classification — built as part of the Tredence AI Engineering Internship case study. Rather than applying pruning as a separate post-training step, the model learns *during training* which connections are worth keeping and which can be discarded.

The core idea is simple but powerful: pair every weight in the dense layers with a **learnable sigmoid gate**. The network is then trained to solve the classification task *and* minimize the number of active connections simultaneously, producing a sparse model without any manual pruning intervention.

This is a controlled sparsity experiment. The focus is not just on getting a high accuracy number — it is on understanding how different levels of sparsity pressure affect the trade-off between classification performance and network compactness.

---

## The Core Idea

Traditional dense layers compute:

$$z = Wx + b$$

A prunable layer replaces this with:

$$z = (W \odot \sigma(G))x + b$$

where:

- $W$ is the weight matrix
- $G$ is a learnable gate score matrix of the same shape
- $\sigma(G)$ maps gate scores to values in $(0, 1)$
- $W' = W \odot \sigma(G)$ is the effective gated weight

When a gate value approaches zero, that connection is effectively removed from the network. Crucially, the gating is end-to-end differentiable — gradients flow through the sigmoid, and the network learns to prune itself through standard backpropagation.

---

## Training Objective

The total loss combines standard cross-entropy with a sparsity regularization term:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{sparse}$$

where the sparsity penalty is:

$$\mathcal{L}_{sparse} = \sum_l \sum_{i,j} \sigma(G_{ij}^{(l)})$$

Because gate values live in $(0, 1)$, summing them penalizes the model for keeping too many connections active. As $\lambda$ increases, the sparsity pressure grows stronger, and the model learns to solve CIFAR-10 with fewer active weights. This creates a clean **accuracy–sparsity trade-off** that is the central subject of this study.

---

## Architecture

The model uses a compact CNN backbone for feature extraction, followed by a prunable dense classifier head. The pruning mechanism is deliberately placed in the dense layers, where it is easy to inspect, measure, and analyze.

```
Input: CIFAR-10 RGB image (3 × 32 × 32)
│
├── Conv(3→64) + BatchNorm + ReLU
├── Conv(64→64) + BatchNorm + ReLU
├── MaxPool (→ 16×16)
│
├── Conv(64→128) + BatchNorm + ReLU
├── Conv(128→128) + BatchNorm + ReLU
├── MaxPool (→ 8×8)
│
├── Conv(128→256) + BatchNorm + ReLU
├── AdaptiveAvgPool (→ 1×1 = 256-dim vector)
│
├── PrunableLinear(256 → 256) + ReLU + Dropout
└── PrunableLinear(256 → 10)
```

A raw MLP over flattened pixels is a common baseline, but it ignores spatial structure and tends to produce noisy, hard-to-interpret sparsity patterns. The CNN backbone handles the spatial reasoning; the prunable head keeps the self-pruning mechanism clean and well-defined.

---

## Engineering Decisions

**Sigmoid gates instead of binary masks.** Hard binary masks are not naturally differentiable, which makes them difficult to learn through backpropagation. Sigmoid gates allow smooth gradient flow throughout training, and can be thresholded to binary at evaluation time.

**Positive gate initialization.** Gate scores are initialized to a moderately positive value (e.g., `+1.0`), which places starting gate values around 0.73. This ensures connections begin mostly active and pruning happens gradually, rather than collapsing the network too early in training.

**Lambda warm-up.** The sparsity penalty coefficient is linearly increased over the first few epochs. This gives the model time to establish a reasonable classification solution before sparsity pressure begins pushing gates toward zero.

**Separate optimizer groups.** The weight parameters use weight decay; the gate score parameters do not. This separation keeps gate learning clean and prevents weight decay from interfering with the sparsity signal.

**Hard-pruned evaluation.** After training, gates below a threshold of 0.5 are zeroed out, and the sparse model is evaluated again. This approximates real deployment behavior where low-weight connections would be removed entirely.

---

## Experiments

The case study asks for results across at least three values of $\lambda$. The recommended sweep is:

| λ | Sparsity Pressure |
|---|---|
| `1e-4` | Low |
| `5e-4` | Moderate |
| `2e-3` | High |

These values were chosen to produce a useful spread across the accuracy–sparsity spectrum without pushing the high-lambda model into unrecoverable accuracy collapse.

### Metrics reported per run

- Test accuracy (soft-gated model)
- Sparsity level (% of gates below threshold)
- Hard-pruned test accuracy
- Hard-pruned sparsity

### Example results table

| Lambda | Test Accuracy (%) | Sparsity (%) | Hard-Pruned Accuracy (%) | Hard-Pruned Sparsity (%) |
|--------|-------------------|--------------|--------------------------|--------------------------|
| 1e-4   | —                 | —            | —                        | —                        |
| 5e-4   | —                 | —            | —                        | —                        |
| 2e-3   | —                 | —            | —                        | —                        |

*Run the notebook to populate these values.*

---

## Visualizations

The notebook generates eight plots that together tell the full story of the pruning behavior:

1. **Lambda vs Test Accuracy** — how classification performance degrades as sparsity pressure increases
2. **Lambda vs Sparsity** — how many connections each lambda value actually removes
3. **Accuracy vs Sparsity Trade-off** — the core scatter plot showing the tension between the two objectives
4. **Test Accuracy Over Training** — learning curves for all lambda values on a single axis
5. **Mean Gate Value Across Training** — shows when and how quickly gates collapse during training
6. **Soft vs Hard-Pruned Accuracy** — compares the sigmoid-gated model against the thresholded sparse model
7. **Per-Layer Sparsity** — separates fc1 and fc2 sparsity to show which layer prunes more aggressively
8. **Confusion Matrix for Best Model** — class-level performance breakdown for the best-performing run

---

## What Makes This Stronger Than a Baseline Submission

Many approaches to this problem stop at: replace `nn.Linear` with a gated variant, add an L1 penalty on the gates, train once, and report a single number.

This project goes further in several meaningful ways:

- The `PrunableLinear` layer is implemented entirely from scratch, with clear separation of weights and gate scores
- The lambda sweep treats the experiment as a mini model-compression study, not a one-shot classifier
- Lambda warm-up prevents gate collapse in early training, which is a common failure mode that baselines often ignore
- Separate optimizer groups prevent weight decay from corrupting the sparsity signal
- Hard-pruned evaluation closes the gap between training-time behavior and what would actually happen at deployment
- Eight complementary visualizations show that the network is genuinely learning to remove connections, not just memorizing the task with small weights

---

## GAN Extension Idea

As an exploratory future direction, a GAN-inspired pruning controller was considered during the design of this project. While it was not used as the primary solution, it is worth outlining as a potential research direction.

### Concept

Rather than learning gate scores directly inside each layer, a GAN-based setup would separate the pruning decision from the classification weights entirely:

- A **generator** (controller network) would propose sparse gate masks or gate logit patterns, given the current model state as input
- A **discriminator or critic** would score whether a proposed mask preserves useful task behavior under a given sparsity budget

In this framing, the generator learns to produce masks that are both compact and classification-preserving, while the critic learns to distinguish genuinely useful sparse structures from random or degenerate ones.

### Why it was not used as the main solution

A GAN-based pruning setup is harder to stabilize than a direct regularized objective. GAN training introduces adversarial dynamics between two networks, which adds hyperparameter complexity and can lead to mode collapse or oscillating masks. More importantly, the case study explicitly asks for a differentiable gate-based pruning mechanism *inside* the model itself — a controller network operating externally is architecturally further from that specification.

For a focused, interpretable case study submission, direct sigmoid gating with a sparsity penalty is the cleaner and more aligned choice.

### How it could be explored later

A future version of this project could combine both ideas:

- Keep the CNN backbone and classifier weights trainable as usual
- Replace the per-layer gate parameters with a small **controller network** that generates gate logits or masks conditioned on the layer's weight statistics or activation patterns
- Train a **critic** to reward compact sparse structures that maintain classification accuracy, effectively providing a learned sparsity budget signal

This would reframe pruning as an **adversarial mask-learning problem** rather than a direct regularization problem. The controller would not just learn a fixed set of gate values — it would learn a general policy for proposing useful masks, potentially generalizing across model checkpoints or even different architectures.

Whether the added complexity is worth the gain depends on the application. For interpretable, reproducible research, the direct approach in this project is preferable. For large-scale compression pipelines where the sparsity budget needs to be dynamic or data-dependent, a learned controller becomes more attractive.

---

## Repository Structure

```
self-pruning-neural-network-cifar10/
├── README.md
├── requirements.txt
└── self_pruning_cifar10.ipynb
```

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Upload this repository to GitHub
2. Replace `YOUR_USERNAME` in the badge link at the top of this file
3. Click the **Open in Colab** badge
4. Run all cells from top to bottom (GPU runtime recommended)

### Option 2 — Local

```bash
git clone https://github.com/YOUR_USERNAME/self-pruning-neural-network-cifar10.git
cd self-pruning-neural-network-cifar10
pip install -r requirements.txt
jupyter notebook self_pruning_cifar10.ipynb
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
scikit-learn>=1.2.0
```


## Acknowledgements

Built as part of the **Tredence AI Engineering Internship** case study. The self-pruning mechanism draws on ideas from structured pruning, magnitude-based gate learning, and differentiable neural architecture search literature.
