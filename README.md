# Tri-Forensic Hypergraph Discrepancy Network for Generalized Image Forgery Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Official Implementation of FIDF (Fake Image Detection Framework)** — Tri-Forensic Hypergraph Discrepancy Network

---

## 🔥 Key Contributions

- **Tri-Forensic Stream Architecture**: Three complementary forensic streams  
  - RGB Stream (EfficientNet-B0) — semantic & structural anomalies  
  - DCT Stream (Learnable DCT basis) — compression inconsistencies  
  - Noise Stream (SRM-initialized) — local noise residual deviations

- **Discrepancy-Gated Cross-Attention (DGCA)**: Novel fusion mechanism that amplifies forensic clues where streams disagree

- **k-NN Hypergraph Convolution**: Captures higher-order relationships among feature tokens

- Comprehensive benchmarking on 5 datasets with identical splits

- Strong cross-dataset generalization evaluation

---

## 🏗️ Architecture

![FIDF Architecture](assets/architecture.png)

### FIDF Pipeline Overview

```mermaid
flowchart TD
    A[Input Image RGB] --> B1[RGB Trunk<br>EffNet-B0]
    A --> B2[DCT Stream<br>Learnable DCT]
    A --> B3[SRM Conv<br>Noise Residual]
    
    B1 --> C1[Token Embedding]
    B2 --> C2[Token Embedding]
    B3 --> C3[Token Embedding]
    
    C1 & C2 --> D[Discrepancy-Gated<br>Cross-Attention DGCA]
    C2 & C3 --> D
    C1 & C3 --> D
    
    D --> E[Token Fusion<br>Weighted Mean]
    E --> F[k-NN Hypergraph<br>Convolution 2 layers]
    F --> G[Classification Head]
    G --> H[Real / Fake]

Core Components
Component,Description,Key Innovation
Learnable DCT,8×8 block DCT with learnable basis,Adapts to dataset-specific compression while preserving orthogonality
SRM Conv,Noise residual extractor,Learnable adaptation of SRM steganalysis filters
DGCA,Discrepancy-gated cross-attention,Gate = σ(MLP(‖Q−K_other‖)) — amplifies disagreements
HyperGraph Conv,k-NN incidence matrix + spectral convolution,Models higher-order token relationships
