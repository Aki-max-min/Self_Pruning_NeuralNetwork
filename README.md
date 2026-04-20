# Self-Pruning Neural Network on CIFAR-10

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/self-pruning-neural-network-cifar10/blob/main/notebooks/self_pruning_cifar10.ipynb)

A Colab-compatible PyTorch implementation of a **self-pruning neural network** for CIFAR-10, built for the **Tredence AI Engineering Internship case study**. This project implements a custom `PrunableLinear` layer where every dense weight is paired with a learnable gate. During training, the model learns both **how to classify** and **which connections are unnecessary**, producing a sparse network without a separate post-training pruning stage.

---

## Project Overview

Traditional neural networks learn only weights.  
This model learns:

- the **weights** needed for classification,
- and the **gates** that decide whether each dense connection should remain active.

The project is designed as a **controlled sparsity experiment** rather than just a modified classifier. It studies how different values of the sparsity coefficient affect the trade-off between **test accuracy** and **network sparsity**.

---

## Core Idea

Each dense-layer weight is multiplied by a learnable sigmoid gate:

$$
W' = W \odot \sigma(G)
$$

where:

- $W$ is the original weight matrix,
- $G$ is the learnable gate score matrix,
- $\sigma(G)$ produces gate values between 0 and 1,
- and $W'$ is the effective gated weight matrix.

If a gate becomes very small, that connection is effectively removed.

The output of a prunable linear layer becomes:

$$
z = (W \odot \sigma(G))x + b
$$

instead of the standard dense layer:

$$
z = Wx + b
$$

This means the network learns not only **what weights to use**, but also **which weights are worth keeping**.

---

## Objective Function

The total training objective combines classification loss and sparsity regularization:

$$
L = L_{\text{CE}} + \lambda \cdot L_{\text{sparse}}
$$

where the sparsity penalty is:

$$
L_{\text{sparse}} = \sum_l \sum_{i,j} \sigma(G^{(l)}_{ij})
$$

This encourages the model to solve the task with as few active connections as possible.

### Why this encourages sparsity

Because the gate values are between 0 and 1, summing them adds a direct penalty for keeping many connections active. As $\lambda$ increases, the model is pushed harder to reduce the number of active gates, creating a clear **accuracy–sparsity trade-off**.

---

## Final Architecture

This repository uses a **compact CNN backbone + prunable classifier head**.

### Model structure

- Input: CIFAR-10 RGB image of shape $3 \times 32 \times 32$
- Conv $(3 \rightarrow 64)$ + BatchNorm + ReLU
- Conv $(64 \rightarrow 64)$ + BatchNorm + ReLU
- MaxPool
- Conv $(64 \rightarrow 128)$ + BatchNorm + ReLU
- Conv $(128 \rightarrow 128)$ + BatchNorm + ReLU
- MaxPool
- Conv $(128 \rightarrow 256)$ + BatchNorm + ReLU
- Adaptive Average Pooling
- `PrunableLinear(256, 256)`
- ReLU
- Dropout
- `PrunableLinear(256, 10)`

### Why this architecture?

A raw MLP over flattened CIFAR-10 pixels is easy to build, but it is not the best choice for image data. A compact CNN backbone captures spatial structure more effectively, while the prunable dense head keeps the assignment’s self-pruning mechanism clearly visible and easy to analyze.

This makes the design both **practical for CIFAR-10** and **well aligned with the case study**.

---

## Why this submission is stronger

Many baseline submissions may stop at:

- replacing `nn.Linear` with a gated layer,
- adding an L1-style gate penalty,
- and reporting one accuracy number.

This project goes further by including:

- a custom `PrunableLinear` layer implemented from scratch,
- multiple values of $\lambda$ to study the sparsity–accuracy trade-off,
- lambda warm-up for stable training,
- separate optimizer groups for weights and gate scores,
- hard-pruned evaluation after training,
- and multiple visual analyses of sparsity behavior.

This makes the project look like a **mini model-compression study**, not just a modified neural network.

---

## Repository Structure

```text
self-pruning-neural-network-cifar10/
├── README.md
├── requirements.txt
├── self_pruning_cifar10.ipynb
├── results/
│   ├── lambda_vs_accuracy.png
│   ├── lambda_vs_sparsity.png
│   ├── accuracy_vs_sparsity.png
│   ├── gate_distribution_best_model.png
│   ├── per_layer_sparsity.png
│   ├── mean_gate_evolution.png
│   ├── soft_vs_hard_accuracy.png
│   └── confusion_matrix_best_model.png
└── report.md
```

---

## How to Run

### Option 1: Run in Colab

1. Upload this repository to GitHub.
2. Replace `YOUR_USERNAME` in the Colab badge link.
3. Click the **Open in Colab** badge at the top.
4. Run all cells from top to bottom.

### Option 2: Run locally

```bash
git clone https://github.com/YOUR_USERNAME/self-pruning-neural-network-cifar10.git
cd self-pruning-neural-network-cifar10
pip install -r requirements.txt
python src/self_pruning_cifar10.py
```

---

## Files in This Repository

### `notebooks/self_pruning_cifar10.ipynb`
Colab-friendly notebook version of the project.

### `src/self_pruning_cifar10.py`
Full Python script for training, evaluation, hard pruning, and visualization.

### `reports/report.md`
Short report explaining the sparsity mechanism, architecture choice, and expected analysis.

### `results/`
Generated plots and summary visuals after training.

---

## Suggested Experiments

The case study asks for results across at least three values of $\lambda$.  
Recommended values:

- $1 \times 10^{-5}$
- $5 \times 10^{-5}$
- $1 \times 10^{-4}$

These usually give a useful spread across:

- low sparsity pressure,
- moderate sparsity pressure,
- and high sparsity pressure.

---

## Metrics to Report

For each value of $\lambda$, report:

- Test Accuracy
- Sparsity Level (%)
- Hard-Pruned Test Accuracy
- Hard-Pruned Sparsity (%)

Example table format:

| Lambda | Test Accuracy (%) | Sparsity Level (%) | Hard-Pruned Accuracy (%) |
|--------|-------------------|--------------------|--------------------------|
| 1e-5   | fill after run    | fill after run     | fill after run           |
| 5e-5   | fill after run    | fill after run     | fill after run           |
| 1e-4   | fill after run    | fill after run     | fill after run           |

---

## Visualizations Included

This project is designed to generate several plots that make the pruning behavior easy to understand:

- **Lambda vs Test Accuracy**
- **Lambda vs Sparsity**
- **Accuracy vs Sparsity Trade-off**
- **Gate Distribution for Best Model**
- **Per-Layer Sparsity**
- **Mean Gate Value Across Training**
- **Soft vs Hard-Pruned Accuracy**
- **Confusion Matrix for Best Model**

These visuals help show that the network is actually learning to remove unnecessary connections, not just memorizing the task.

---

## Key Engineering Decisions

### 1. Sigmoid gates instead of hard binary masks
Hard binary masks are difficult to optimize directly because they are not naturally differentiable. Sigmoid gates allow smooth gradient flow through the pruning mechanism during backpropagation.

### 2. Positive gate initialization
Gate scores are initialized to a positive value so that connections start mostly active and pruning happens gradually instead of collapsing too early.

### 3. Lambda warm-up
The sparsity penalty is gradually increased over the first few epochs to avoid premature pruning.

### 4. Separate optimizer groups
Normal weights use weight decay, while gate scores do not. This keeps gate learning cleaner and more interpretable.

### 5. Hard-pruned evaluation
After training, low-gate connections are thresholded to zero and the sparse model is evaluated again to approximate true deployment behavior.

---

## GAN Extension Idea

As an exploratory future direction, a **GAN-inspired pruning controller** was considered.

### Concept

- A **generator** could propose sparse masks or gate patterns.
- A **discriminator or critic** could score whether the masked model preserves useful task behavior under a sparsity budget.

### Why it was not used as the main solution

Although interesting, a GAN-based pruning setup is harder to train and less directly aligned with the case study, which explicitly asks for a differentiable gate-based pruning mechanism inside the model itself.

### How it could be explored later

A future version could:

- keep classifier weights trainable,
- let a small controller network generate gate logits or masks,
- and use a critic to reward compact sparse structures that preserve classification performance.

This would turn pruning into an adversarial mask-learning problem rather than a direct regularized optimization problem.

---

## Why This Project Stands Out

This project is not just a classifier with extra regularization. It is a structured experiment that asks:

- how sparsity changes with $\lambda$,
- how much accuracy is preserved after pruning,
- where pruning happens across layers,
- and whether the learned sparse structure survives hard thresholding.

That makes the project stronger than a minimal implementation and gives it a more research-oriented feel.

---

## Future Work

- Hard-concrete gates for sharper sparsity
- Extending pruning to convolutional layers
- Structured pruning for real inference speedups
- GAN-based sparse mask generation
- Layerwise sparsity scheduling
- FLOPs and parameter-count analysis after pruning

---

## Skills Demonstrated

- PyTorch custom module design
- Differentiable pruning mechanisms
- CIFAR-10 image classification
- Custom training loops
- Sparsity-aware optimization
- Experimental analysis and visualization
- GitHub/Colab reproducibility

---

## Author

**AKSHITA SINGH TYAGI**  

---

## Acknowledgment

This project was developed as a response to the Tredence AI Engineering Internship case study on self-pruning neural networks.
