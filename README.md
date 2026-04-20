# Self_Pruning_NeuralNetwork
# Self-Pruning Neural Network on CIFAR-10


A Colab-compatible PyTorch implementation of a **self-pruning neural network** for CIFAR-10, built for the **Tredence AI Engineering Internship case study**.  
This project implements a custom `PrunableLinear` layer where every weight is paired with a learnable gate. During training, the model learns both **how to classify** and **which connections are unnecessary**, producing a sparse network without a separate post-training pruning stage.

---

## Project Idea

Traditional neural networks learn only weights.  
This model learns:

- the **weights** needed for classification,
- and the **gates** that determine whether each connection should remain active.

Each linear weight is multiplied by a sigmoid gate:

\[
W' = W \odot \sigma(G)
\]

where:
- \(W\) is the weight matrix,
- \(G\) is the learnable gate score matrix,
- \(\sigma(G)\) produces gate values between 0 and 1.

If a gate becomes very small, that connection is effectively removed.

---

## Objective Function

The training objective combines classification accuracy and sparsity:

\[
L = L_{\text{CE}} + \lambda \cdot L_{\text{sparse}}
\]

where:

\[
L_{\text{sparse}} = \sum_{l} \sum_{i,j} \sigma(G^{(l)}_{ij})
\]

This encourages the model to solve the task with as few active connections as possible.

---

## Architecture

This repository uses a **compact CNN backbone + prunable classifier head**.

### Final model
- Convolution block: Conv → BatchNorm → ReLU
- Convolution block: Conv → BatchNorm → ReLU → MaxPool
- Convolution block: Conv → BatchNorm → ReLU
- Convolution block: Conv → BatchNorm → ReLU → MaxPool
- Convolution block: Conv → BatchNorm → ReLU
- Adaptive Average Pooling
- `PrunableLinear(256, 256)`
- ReLU
- Dropout
- `PrunableLinear(256, 10)`

### Why this architecture?
A raw MLP on flattened CIFAR-10 pixels is easy to build but is not ideal for image data.  
A small CNN backbone captures spatial structure, while the prunable dense head clearly demonstrates the required self-pruning mechanism.

---

## Why this submission is different

Many baseline submissions may stop at:
- replacing `nn.Linear` with a gated layer,
- adding L1 loss,
- and reporting a single accuracy number.

This project goes further by including:

- a **custom prunable layer** implemented from scratch,
- **multiple lambda values** to study the sparsity-accuracy trade-off,
- **lambda warm-up** for stable training,
- **separate optimizer groups** for weights and gate scores,
- **hard-pruned evaluation** after training,
- and **gate distribution analysis** for the best model.

This makes the project a controlled sparsity experiment rather than just a modified classifier.

---

## Files

```text
self-pruning-neural-network-cifar10/
├── README.md
├── requirements.txt
├── notebooks/
│   └── self_pruning_cifar10.ipynb
├── src/
│   └── self_pruning_cifar10.py
├── results/
│   ├── gate_distribution.png
│   └── results_table.png
└── reports/
    └── report.md
```

---

## How to Run

### Option 1: Open in Colab
Click the **Open in Colab** badge at the top and run all cells.

### Option 2: Run locally
```bash
git clone https://github.com/YOUR_USERNAME/self-pruning-neural-network-cifar10.git
cd self-pruning-neural-network-cifar10
pip install -r requirements.txt
python src/self_pruning_cifar10.py
```

---

## Results to Report

For each \(\lambda\), report:

- Test Accuracy
- Sparsity Level (%)
- Hard-pruned Test Accuracy
- Gate distribution for the best model

Example result table format:

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|--------------------|
| 1e-5   | xx.xx             | xx.xx              |
| 5e-5   | xx.xx             | xx.xx              |
| 1e-4   | xx.xx             | xx.xx              |

---

## Key Engineering Decisions

### 1. Sigmoid gates instead of hard binary masks
Hard masks are difficult to optimize directly because they are not naturally differentiable.  
Sigmoid gates allow smooth gradient flow through the pruning mechanism during training.

### 2. Positive gate initialization
Gate scores are initialized to a positive value so that connections start mostly active and pruning happens gradually instead of destroying learning at the beginning.

### 3. Lambda warm-up
The sparsity penalty is gradually increased during early epochs to avoid premature pruning.

### 4. Hard-pruned evaluation
After training, low-gate connections are thresholded to zero and the sparse model is evaluated again to measure real pruning quality.

---

## GAN Extension Idea

As an exploratory future direction, I considered a **GAN-inspired pruning controller**.

### Concept
- A **generator** could propose sparse masks or gate patterns.
- A **discriminator/critic** could evaluate whether the masked network preserves useful task behavior under a sparsity budget.

### Why it was not used as the main solution
Although interesting, a GAN-based pruning setup is more complex and harder to stabilize.  
For this case study, the required differentiable gate-based pruning mechanism is more aligned with the problem statement, easier to analyze, and more reliable to train.

### How it could be explored later
A future version could:
- keep the main classifier weights trainable,
- let a mask-generator network produce gate logits,
- and train a critic to score task-preserving compactness.

This could turn pruning into an adversarial architecture search problem.

---

## Skills Demonstrated

- PyTorch custom module design
- Differentiable sparsity mechanisms
- Custom training loops
- CIFAR-10 image classification
- Model compression reasoning
- Experimental analysis and reporting

---

## Future Work

- Hard-concrete gates for sharper sparsity
- Pruning in convolutional layers
- Structured pruning for actual inference speedup
- GAN-based sparse mask generation
- Layerwise sparsity scheduling

---

## Author

**AKSHITA TYAGI**  

## Acknowledgment

This project was developed as part of the Tredence AI Engineering Internship case study on self-pruning neural networks.
