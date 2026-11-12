# Deep Learning and Multi-Objective Optimization Master Guide
## Mathematical Foundations, Layer Mechanics, Differentiable Grad-CAM, Kendall Uncertainty, and the AdamW Optimization Engine in MediLink-DL

**Authors & Engineering Team**:
* **Mayank Soni** (Roll No: CS24B1031) — *System Architecture, Multi-Objective Optimization & Cryptographic Research*
* **Nandan** (Roll No: CS24B1034) — *Deep Learning Localization, Differentiable Grad-CAM & Loss Formulations*
* **Sumedha** (Roll No: AD24B1061) — *Digital Image Processing, Dual-Stream Compression & Priority ARQ Transport*

**Faculty Mentors**:
* **Dr. Kaushiki Roy** — *Faculty Advisor, Digital Image Processing (DIP)*
* **Dr. Mehbub Alam** — *Faculty Advisor, Computer Networks (CN)*
* **Arnav Ghosh** — *Faculty Advisor, Multi-Objective Optimization & Applied Mathematics*

**Institution**: Indian Institute of Information Technology Raichur (IIIT Raichur)  
**Date**: November 2026 &nbsp;|&nbsp; **Technical Reference Manual**

---

## Table of Contents
1. [Introduction: Why Optimization Is the Core of MediLink-DL](#1-introduction-why-optimization-is-the-core-of-medilink-dl)
2. [Ground Zero: Deep Learning Fundamentals on Grayscale Radiographs](#2-ground-zero-deep-learning-fundamentals-on-grayscale-radiographs)
3. [MedicalResNet18 Architecture: Layer-by-Layer Tensor Mechanics](#3-medicalresnet18-architecture-layer-by-layer-tensor-mechanics)
4. [Saliency Drift and Differentiable Grad-CAM](#4-saliency-drift-and-differentiable-grad-cam)
5. [Multi-Task Loss Formulation (Classification + Spatial Localization)](#5-multi-task-loss-formulation-classification--spatial-localization)
6. [Kendall & Gal Homoscedastic Uncertainty Optimization](#6-kendall--gal-homoscedastic-uncertainty-optimization)
7. [The Complete Step-by-Step Propagation Walkthrough (With Real Numbers)](#7-the-complete-step-by-step-propagation-walkthrough-with-real-numbers)
8. [The AdamW Optimization Engine: Momentum, Velocity, and Bias Correction](#8-the-adamw-optimization-engine-momentum-velocity-and-bias-correction)
9. [How Model Weights ($W$) and Uncertainty Knobs ($s$) Converge Together](#9-how-model-weights-w-and-uncertainty-knobs-s-converge-together)
10. [System-Level Multi-Objective Pareto Optimization (108 Configurations)](#10-system-level-multi-objective-pareto-optimization-108-configurations)
11. [Viva Defense Master Cheat Sheet (Tough Questions & Exact Answers)](#11-viva-defense-master-cheat-sheet-tough-questions--exact-answers)

---

## 1. Introduction: Why Optimization Is the Core of MediLink-DL

In everyday language, **Optimization** means finding the absolute best possible settings for a system.
* If a metric represents an **error or cost** (such as misdiagnosis, file size, or transmission delay), optimization means **MINIMIZING** it to the lowest possible value.
* If a metric represents a **benefit or quality** (such as diagnostic accuracy, Pointing Game Hit Rate, or structural similarity SSIM), optimization means **MAXIMIZING** it.

In MediLink-DL, optimization operates at **two distinct levels**:

```
+-------------------------------------------------------------------------------------------------+
|                                 TWO LEVELS OF OPTIMIZATION                                      |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   LEVEL 1: Micro-Level (Deep Learning Model Training)                                           |
|   - Goal: Train MedicalResNet18 to classify COVID-19 AND locate lung lesions simultaneously.    |
|   - Algorithm: Second-order backpropagation with Kendall & Gal uncertainty loss + AdamW.       |
|   - Variables being optimized: ~11.17 million convolutional weights W + 3 uncertainty knobs s.  |
|                                                                                                 |
|   LEVEL 2: Macro-Level (System Pipeline Engineering)                                            |
|   - Goal: Find the best trade-off between Image Quality, File Size, Speed, and CPU Battery.     |
|   - Algorithm: Multi-Objective Grid Search (108 configurations) + Pareto Dominance Filtering.   |
|   - Output: 26 Non-Dominated Pareto-optimal operating points (pareto_frontier.csv).             |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Ground Zero: Deep Learning Fundamentals on Grayscale Radiographs

To understand how our model learns, we must first understand what an image and a neural network look like to a computer.

### 2.1 What Is a Digital Radiograph to a Computer?
A digital chest X-ray is not a visual photograph. To a microprocessor, it is a two-dimensional matrix (tensor) of numbers:
$$X \in \mathbb{R}^{1 \times 512 \times 512}$$
* `1`: Single channel (Grayscale intensity, unlike 3-channel RGB consumer photos).
* `512 x 512`: Spatial dimensions (262,144 individual pixel locations).
* Each pixel value represents radiological radiodensity:
  * $0.0$ (Black): Air / Lucent regions (lungs filled with air).
  * $1.0$ (White): Dense tissue (bones, ribs, metal implants, or pathological liquid consolidations/pneumonia).

```
Pixel Grid Representation:
(0,0)  [ 0.05,  0.08,  0.12, ... ,  0.02 ]
(0,1)  [ 0.10,  0.89,  0.92, ... ,  0.04 ]  <-- Dense rib bone (high value)
 ...   [ ... ,  ... ,  ... , ... ,  ...  ]
(511)  [ 0.01,  0.02,  0.01, ... ,  0.00 ]  <-- Empty background air
```

### 2.2 What Is a Convolutional Kernel (Weight $W$)?
A convolutional kernel is a small matrix of numbers (e.g., $3 \times 3$ or $7 \times 7$) that slides across the input image. At every position, it performs an element-wise multiplication and sums the result (a Dot Product):

```
Input Image Patch (3x3):         Convolutional Kernel W (3x3):
   [ 0.2   0.8   0.9 ]                [ +1.0   0.0  -1.0 ]
   [ 0.1   0.7   0.8 ]        *       [ +2.0   0.0  -2.0 ]
   [ 0.3   0.9   0.8 ]                [ +1.0   0.0  -1.0 ]

Output Value = (0.2*1.0) + (0.8*0.0) + (0.9*-1.0) + ... = Filter Response
```

* If the kernel contains vertical edge-detecting numbers, it outputs a high number when it slides over a rib or pleural border.
* If the kernel contains diffuse patch-detecting numbers, it outputs a high number when it slides over a hazy ground-glass consolidation.
* **The numbers inside the kernel are the "Weights" ($W$) of the model.** Initially, these weights are random garbage numbers. Training is the process of adjusting these weights until the kernels become expert feature detectors.

---

## 3. MedicalResNet18 Architecture: Layer-by-Layer Tensor Mechanics

Standard torchvision ResNet-18 is designed for ImageNet (3-channel RGB consumer photos of cats, dogs, cars). Feeding a grayscale X-ray into standard ResNet-18 requires artificially triplicating the grayscale channel into RGB ($3 \times 512 \times 512$), which wastes 3× memory bandwidth and introduces unnecessary parameter redundancy.

We engineered `MedicalResNet18` (`src/roi_localization.py`), modifying the first convolutional layer directly for single-channel medical radiographs.

```
+----------------------------------------------------------------------------------------------------+
|                               MEDICALRESNET18 TENSOR FLOW PIPELINE                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ Input CXR ]           1 x 512 x 512 single-channel grayscale raster                             |
|        |                                                                                           |
|        v                                                                                           |
|  [ conv1 ]               7x7 Conv, Stride=2, Padding=3, 64 Filters -> Output: 64 x 256 x 256       |
|        |                                                                                           |
|        v                                                                                           |
|  [ bn1 + relu ]          Batch Normalization + Rectified Linear Unit                               |
|        |                                                                                           |
|        v                                                                                           |
|  [ maxpool ]             3x3 Max Pooling, Stride=2, Padding=1 -> Output: 64 x 128 x 128             |
|        |                                                                                           |
|        v                                                                                           |
|  [ layer1 ]              2 Residual Blocks (64 channels)      -> Output: 64 x 128 x 128            |
|        |                                                                                           |
|        v                                                                                           |
|  [ layer2 ]              2 Residual Blocks (128 channels, s=2) -> Output: 128 x 64 x 64            |
|        |                                                                                           |
|        v                                                                                           |
|  [ layer3 ]              2 Residual Blocks (256 channels, s=2) -> Output: 256 x 32 x 32            |
|        |                                                                                           |
|        v                                                                                           |
|  [ layer4 ]              2 Residual Blocks (512 channels, s=2) -> Output: 512 x 16 x 16 (A^k maps) |
|        |                                                         (Target Layer for Grad-CAM)       |
|        +----------------------------------------+                                                  |
|        |                                        |                                                  |
|        v (Branch A: Classification)             v (Branch B: Differentiable Grad-CAM)              |
|  [ Global Avg Pool ]                            [ Backprop Gradients d y^c / d A^k ]               |
|  Outputs: 512 x 1 x 1                                   |                                          |
|        |                                                v                                          |
|        v                                        [ Neuron Weights alpha_k^c ]                       |
|  [ Linear FC Head ]                                     |                                          |
|  Outputs: 2 Logits (Normal, COVID)                      v                                          |
|        |                                        [ ReLU Aggregation + Bilinear Upsample ]           |
|        v                                                |                                          |
|  [ Logit y^c ] ---------------------------------------->+                                          |
|                                                         v                                          |
|                                                 [ Heatmap M_CAM in [0, 1]^(512x512) ]              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 The Residual BasicBlock Equation
Inside each layer, input $x$ passes through two $3 \times 3$ convolutions and adds a shortcut connection (skip connection):
$$F(x) = \text{BN}_2\big(\text{Conv}_2\big(\text{ReLU}\big(\text{BN}_1(\text{Conv}_1(x))\big)\big)\big)$$
$$\text{Output} = \text{ReLU}\big(F(x) + x\big)$$

The shortcut connection $+ x$ allows gradients during backpropagation to flow directly backward without vanishing, enabling deep networks to train smoothly.

---

## 4. Saliency Drift and Differentiable Grad-CAM

### 4.1 What Is Saliency Drift?
If a neural network is trained *only* on classification labels (COVID vs Normal), it frequently memorizes superficial shortcuts. For example:
* In emergency rooms, COVID-19 patients were often imaged with portable anteroposterior (AP) scanners bearing a distinct metallic "L" or "PORTABLE" marker in the upper-left corner.
* Normal control patients were imaged with standard posteroanterior (PA) machines without that marker.

An unconstrained network achieves 90% classification accuracy simply by detecting the white letters in the corner, completely ignoring the lungs! When this heatmap is used to compress the image, the compressor preserves the corner letters at high quality and blurs out the actual pneumonia in the lungs. This catastrophic failure is called **Saliency Drift**.

### 4.2 Standard Grad-CAM vs Differentiable Grad-CAM
* **Standard Grad-CAM (Selvaraju et al., ICCV 2017)**: 
  A post-hoc visualization tool. It evaluates activations $A^k$ and gradients $\frac{\partial y^c}{\partial A^k}$ with gradient tracking disabled (`torch.no_grad()` or detached graphs). It generates a pretty picture for humans to inspect, but **it cannot modify the model weights**.
* **Differentiable Grad-CAM (Our Formulation in `src/localization_loss.py`)**:
  We execute the Grad-CAM backward pass with `create_graph=True`. This preserves the computational graph of the gradients themselves, enabling **second-order backpropagation**:
  $$\frac{\partial \mathcal{L}_{\text{loc}}}{\partial W} = \frac{\partial \mathcal{L}_{\text{loc}}}{\partial M_{\text{CAM}}} \cdot \frac{\partial M_{\text{CAM}}}{\partial A^k} \cdot \frac{\partial A^k}{\partial W}$$
  The spatial error between the heatmap and the true lung lesion physically pulls the convolutional weights $W$ toward learning lung pathology textures rather than corner text artifacts.

### 4.3 Mathematical Formulation of Differentiable Grad-CAM
1. **Neuron Importance Weights ($\alpha_k^c$)**:
   Let $y^c$ be the predicted classification score for class $c$ (COVID positive). Let $A^k \in \mathbb{R}^{H' \times W'}$ be the $k$-th feature map of `layer4[-1]` ($k \in [1, 512]$). The importance of feature map $k$ is its spatially pooled gradient:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A_{i,j}^k}, \quad Z = H' \times W' = 16 \times 16 = 256$$

2. **Continuous Heatmap Assembly**:
   We compute the weighted combination of all 512 feature maps and pass through ReLU to retain only features that positively support class $c$:
   $$H_{\text{CAM}}^c = \operatorname{ReLU}\left( \sum_{k=1}^{512} \alpha_k^c A^k \right)$$

3. **Bilinear Upsampling and Normalization**:
   $H_{\text{CAM}}^c$ is bilinearly interpolated from $16 \times 16$ to the native image resolution $512 \times 512$ and normalized to $[0.0, 1.0]$:
   $$M_{\text{CAM}} = \frac{H_{\text{CAM}}^c - \min(H_{\text{CAM}}^c)}{\max(H_{\text{CAM}}^c) - \min(H_{\text{CAM}}^c) + \epsilon}$$

---

## 5. Multi-Task Loss Formulation (Classification + Spatial Localization)

Training `MedicalResNet18` requires evaluating three distinct mathematical penalties on every forward pass:

```
                                [ Total Training Loss ]
                                           |
             +-----------------------------+-----------------------------+
             |                                                           |
             v                                                           v
   [ Task 1: Classification ]                                  [ Task 2: Spatial Localization ]
   Cross-Entropy Loss (L_cls)                                     Localization Loss (L_loc)
   "Did you name the disease?"                                    "Did you look at the lungs?"
                                                                         |
                                                   +---------------------+---------------------+
                                                   |                                           |
                                                   v                                           v
                                        [ Cosine Alignment ]                            [ Dice Overlap ]
                                            Loss (L_align)                                Loss (L_dice)
                                      "Is attention vector aligned?"                "Do masks overlap in 2D?"
```

### 5.1 Classification Loss ($\mathcal{L}_{\text{cls}}$)
Evaluated using standard categorical cross-entropy between ground-truth label $y \in \{0, 1\}$ and predicted softmax probability $\hat{p}$:
$$\mathcal{L}_{\text{cls}} = -\Big[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \Big]$$

### 5.2 Cosine Vector Alignment Loss ($\mathcal{L}_{\text{align}}$)
Treats the continuous Grad-CAM heatmap $M_{\text{CAM}}$ and the radiologist's ground truth binary mask $M_{\text{GT}}$ as flattened 1D vectors in $\mathbb{R}^{262,144}$ and measures directional alignment:
$$\mathcal{L}_{\text{align}} = 1 - \frac{\langle \operatorname{vec}(M_{\text{CAM}}), \operatorname{vec}(M_{\text{GT}}) \rangle}{\|\operatorname{vec}(M_{\text{CAM}})\|_2 \|\operatorname{vec}(M_{\text{GT}})\|_2 + \epsilon}$$
* If $M_{\text{CAM}}$ points in the exact same spatial direction as the lung mask, the cosine similarity is $1.0$, and $\mathcal{L}_{\text{align}} = 0.0$.
* If $M_{\text{CAM}}$ focuses on the upper clavicle bone while the lesion is in the lower lobe, cosine similarity drops to $0.1$, generating a heavy penalty of $\mathcal{L}_{\text{align}} = 0.9$.

### 5.3 Continuous Spatial Dice Loss ($\mathcal{L}_{\text{dice}}$)
Measures the region overlap and penalizes both false positives (highlighting healthy tissue) and false negatives (missing lesion margins):
$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{i,j} M_{\text{CAM}}(i,j) M_{\text{GT}}(i,j) + \epsilon}{\sum_{i,j} M_{\text{CAM}}(i,j)^2 + \sum_{i,j} M_{\text{GT}}(i,j)^2 + \epsilon}$$

### 5.4 Composite Localization Loss ($\mathcal{L}_{\text{loc}}$)
$$\mathcal{L}_{\text{loc}} = (1 - \beta) \mathcal{L}_{\text{align}} + \beta \mathcal{L}_{\text{dice}}$$

---

## 6. Kendall & Gal Homoscedastic Uncertainty Optimization

### 6.1 The Fundamental Flaw of Fixed Weighting
In conventional multi-task learning, researchers combine losses using fixed scalar hyperparameters:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{loc}}$$

This approach fails in clinical practice:
* Classification loss ($\mathcal{L}_{\text{cls}}$) typically starts around $0.7$ and decreases to $0.2$.
* Localization loss ($\mathcal{L}_{\text{loc}}$) operates on high-dimensional pixel matrices and can range from $1.5$ down to $0.4$.
* If $\lambda$ is set too large (e.g., $2.0$), the gradients of localization overwhelm the network. The model learns to draw perfect circles on the lungs but fails to diagnose whether the patient actually has COVID-19.
* If $\lambda$ is set too small (e.g., $0.1$), the model classifies accurately but exhibits saliency drift.
* Sweeping $\lambda$ over a grid ($0.1, 0.2, \dots, 5.0$) requires hundreds of hours of GPU training and does not adapt dynamically as the model learns over epochs.

### 6.2 The Probabilistic Formulation
Kendall, Gal, and Cipolla (CVPR 2018) reformulated multi-task loss weighting as **Maximum Likelihood Estimation under Homoscedastic Task Uncertainty**.

Assume each task output is corrupted by Gaussian observation noise with variance $\sigma_i^2$. The joint Gaussian log-likelihood for classification and localization is:
$$p(y, M \mid f(x)) = \mathcal{N}\big(y; f_{\text{cls}}(x), \sigma_1^2\big) \times \mathcal{N}\big(M; f_{\text{loc}}(x), \sigma_2^2\big)$$

Taking the negative log-likelihood ($-\log p$):
$$\mathcal{L}(w, \sigma_1, \sigma_2) = \frac{1}{2\sigma_1^2}\mathcal{L}_{\text{cls}}(w) + \frac{1}{2\sigma_2^2}\mathcal{L}_{\text{loc}}(w) + \log(\sigma_1) + \log(\sigma_2)$$

* $\frac{1}{2\sigma_i^2}\mathcal{L}_i$: Down-weights tasks with high observation noise / uncertainty.
* $+\log(\sigma_i)$: A mathematical **regularizer (penalty)** that prevents the model from trivially setting $\sigma_i \to \infty$ to drive the loss to zero.

### 6.3 Log-Variance Parameterization ($s_i = \log \sigma_i^2$)
To prevent division-by-zero crashes when $\sigma_i \to 0$ and ensure $\sigma_i > 0$ strictly holds without constrained optimization, we substitute:
$$s_i = \log(\sigma_i^2) \implies \sigma_i^2 = \exp(s_i) \implies \frac{1}{\sigma_i^2} = \exp(-s_i)$$
$$\log(\sigma_i) = \frac{1}{2}\log(\sigma_i^2) = \frac{1}{2}s_i$$

Substituting these yields the final, numerically stable **Kendall & Gal Optimization Function**:
$$\mathcal{L}_{\text{total}}(w, s_1, s_2) = \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}}(w) + \frac{1}{2}\exp(-s_2)\mathcal{L}_{\text{loc}}(w) + \frac{1}{2}s_1 + \frac{1}{2}s_2$$

In MediLink-DL (`src/localization_loss.py`), we implement this across three distinct sub-tasks:
$$\mathcal{L}_{\text{total}} = \frac{1}{2}e^{-s_{\text{cls}}}\mathcal{L}_{\text{cls}} + \frac{1}{2}e^{-s_{\text{align}}}\mathcal{L}_{\text{align}} + \frac{1}{2}e^{-s_{\text{dice}}}\mathcal{L}_{\text{dice}} + \frac{1}{2}\big(s_{\text{cls}} + s_{\text{align}} + s_{\text{dice}}\big)$$

### 6.4 The Mathematical Proof: Why the Minimum Converges to $\sigma_i^2 = \mathcal{L}_i$
Let us analytically differentiate $\mathcal{L}_{\text{total}}$ with respect to the uncertainty parameter $s_1$:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial s_1} = \frac{\partial}{\partial s_1}\left[ \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}} + \frac{1}{2}s_1 \right]$$

Applying basic calculus:
$$\frac{d}{ds_1}\big[\exp(-s_1)\big] = -\exp(-s_1), \quad \frac{d}{ds_1}\left[\frac{1}{2}s_1\right] = \frac{1}{2}$$
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial s_1} = -\frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}} + \frac{1}{2} = \frac{1}{2}\Big( 1 - \exp(-s_1)\mathcal{L}_{\text{cls}} \Big)$$

To find the minimum of the loss surface, set the derivative to zero:
$$\frac{1}{2}\Big( 1 - \exp(-s_1)\mathcal{L}_{\text{cls}} \Big) = 0$$
$$1 - \exp(-s_1)\mathcal{L}_{\text{cls}} = 0 \implies \exp(-s_1)\mathcal{L}_{\text{cls}} = 1$$
$$\frac{\mathcal{L}_{\text{cls}}}{\exp(s_1)} = 1 \implies \frac{\mathcal{L}_{\text{cls}}}{\sigma_1^2} = 1 \implies \mathbf{\sigma_1^2 = \mathcal{L}_{\text{cls}}}$$

> **Key Mathematical Insight**: At the optimal stationary point, the learned variance $\sigma_i^2$ automatically equals the current loss of that task ($\mathcal{L}_i$). When a task is difficult and its loss is high, the model automatically increases $\sigma_i$, preventing the difficult task from destabilizing training. As the model masters the task and its loss shrinks, $\sigma_i$ shrinks, increasing the task weight!

### 6.5 Derivation of Effective $\lambda$ and $\beta$
How does this relate to standard hyperparameters?
The standard composite loss is:
$$\mathcal{L} \propto \mathcal{L}_{\text{cls}} + \lambda_{\text{effective}} \cdot \mathcal{L}_{\text{align}} + (\lambda_{\text{effective}} \cdot \beta_{\text{effective}}) \cdot \mathcal{L}_{\text{dice}}$$

Dividing our Kendall total loss by the classification weight $w_{\text{cls}} = \exp(-s_{\text{cls}})$:
$$\lambda_{\text{effective}} = \frac{w_{\text{align}}}{w_{\text{cls}}} = \frac{\exp(-s_{\text{align}})}{\exp(-s_{\text{cls}})} = \exp\big(s_{\text{cls}} - s_{\text{align}}\big)$$
$$\beta_{\text{effective}} = \frac{w_{\text{dice}}}{w_{\text{align}}} = \frac{\exp(-s_{\text{dice}})}{\exp(-s_{\text{align}})} = \exp\big(s_{\text{align}} - s_{\text{dice}}\big)$$

In `src/localization_loss.py`, `get_effective_weights()` reports these dynamically learned multipliers at every training epoch.

---

## 7. The Complete Step-by-Step Propagation Walkthrough (With Real Numbers)

To eliminate any ambiguity, let us trace one single training step ($t=1$) through the entire system using concrete hypothetical numbers.

```text
====================================================================================================
                        STEP-BY-STEP NUMERICAL TRACE OF ONE ITERATION
====================================================================================================
```

### [Phase 1: Forward Pass]
* **Input**: Single patient radiograph $X \in \mathbb{R}^{1 \times 512 \times 512}$.
* **Ground Truth**: Label $y = 1$ (COVID positive); Mask $M_{\text{GT}}$ has lesion area $= 1.0$.
* **Initial State**:
  * Convolutional kernel weight: $W_0 = 0.500$.
  * Kendall uncertainty parameters: $s_{\text{cls}} = 0.0$, $s_{\text{loc}} = 0.0 \implies \exp(-s) = 1.0$.

1. **Feature Extraction**:
   $$A = \operatorname{ReLU}(W \cdot x) = \operatorname{ReLU}(0.500 \cdot 1.200) = 0.600$$
2. **Classification Output**:
   $$\text{Logit } z = 1.0986 \implies \hat{p} = \frac{e^{1.0986}}{e^{1.0986} + 1} = \frac{3.0}{4.0} = 0.750 \quad (75\% \text{ confidence})$$
3. **Grad-CAM Generation**:
   $$\alpha = \frac{\partial z}{\partial A} = 0.667 \implies H_{\text{CAM}} = \operatorname{ReLU}(0.667 \cdot 0.600) = 0.400$$
   Upsampled normalized heatmap value over lesion site: $M_{\text{CAM}} = 0.400$.

### [Phase 2: Loss Computation]
1. **Classification Loss**:
   $$\mathcal{L}_{\text{cls}} = -\log(0.750) = 0.2877$$
2. **Localization Loss (Dice)**:
   $$\mathcal{L}_{\text{loc}} = 1 - \frac{2 \cdot (0.400) \cdot (1.000)}{(0.400)^2 + (1.000)^2} = 1 - \frac{0.800}{0.160 + 1.000} = 1 - \frac{0.800}{1.160} = 1 - 0.6897 = 0.3103$$
3. **Kendall Total Loss**:
   $$\mathcal{L}_{\text{total}} = \frac{1}{2}(1.0)(0.2877) + \frac{1}{2}(1.0)(0.3103) + \frac{1}{2}(0.0) + \frac{1}{2}(0.0) = 0.1438 + 0.1552 = \mathbf{0.2990}$$

### [Phase 3: Backward Pass (`loss.backward()`)]
PyTorch executes automatic differentiation to compute the gradients:
1. **Gradient of Kendall Knob $s_{\text{cls}}$**:
   $$g_{s} = \frac{\partial \mathcal{L}_{\text{total}}}{\partial s_{\text{cls}}} = \frac{1}{2} - \frac{1}{2}e^{-s_{\text{cls}}}\mathcal{L}_{\text{cls}} = 0.500 - 0.500(1.0)(0.2877) = 0.500 - 0.1438 = \mathbf{+0.3562}$$
2. **Gradient of Model Weight $W$**:
   Via chain rule, both tasks send back gradient forces:
   $$\frac{\partial \mathcal{L}_{\text{cls}}}{\partial W} = -0.4000 \quad (\text{push to increase } \hat{p})$$
   $$\frac{\partial \mathcal{L}_{\text{loc}}}{\partial W} = -0.6000 \quad (\text{push to increase heatmap overlap})$$
   $$g_W = \frac{\partial \mathcal{L}_{\text{total}}}{\partial W} = \frac{1}{2}(1.0)(-0.4000) + \frac{1}{2}(1.0)(-0.6000) = -0.2000 - 0.3000 = \mathbf{-0.5000}$$

---

## 8. The AdamW Optimization Engine: Momentum, Velocity, and Bias Correction

Once gradients $g_W = -0.5000$ and $g_s = +0.3562$ are calculated, `optimizer.step()` updates the parameters using the **AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)** algorithm.

### 8.1 The Core Equations of AdamW
For any trainable parameter $\theta$ (whether a CNN weight $W$ or a Kendall parameter $s$):
1. **Gradient Evaluation**: $g_t = \nabla_\theta \mathcal{L}_t$
2. **First Moment Vector (Momentum / Moving Average of Gradients)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   *(Tracks direction and builds momentum along consistent slopes, typically $\beta_1 = 0.9$)*
3. **Second Moment Vector (Velocity / Uncentered Variance of Gradients)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   *(Tracks volatility and gradient magnitude, typically $\beta_2 = 0.999$)*
4. **Bias Correction**:
   Because $m_0 = 0$ and $v_0 = 0$, both vectors are biased toward zero at early time steps ($t$). Adam corrects this by dividing by $(1 - \beta^t)$:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
5. **Decoupled Weight Decay and Parameter Update**:
   $$\theta_t = \theta_{t-1} - \underbrace{\alpha \lambda_{\text{wd}} \theta_{t-1}}_{\text{Weight Decay}} - \underbrace{\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t}_{\text{Adaptive Gradient Step}}$$
   * $\alpha$: Learning rate (e.g., $10^{-3} = 0.001$).
   * $\lambda_{\text{wd}}$: Weight decay factor ($10^{-4} = 0.0001$).
   * $\epsilon$: Numerical stabilization constant ($10^{-8}$).

---

### 8.2 Detailed Step-by-Step Numerical Walkthrough of AdamW Updates

Let us calculate the exact numbers for Step 1 ($t=1$) and Step 2 ($t=2$):

#### A) Step $t=1$ for Convolutional Weight $W$ (Initial $W_0 = 0.5000, g_1 = -0.5000$):
1. **Momentum $m_1$**:
   $$m_1 = 0.9 \cdot (0) + (1 - 0.9) \cdot (-0.5000) = 0.1 \cdot (-0.5000) = \mathbf{-0.0500}$$
2. **Velocity $v_1$**:
   $$v_1 = 0.999 \cdot (0) + (1 - 0.999) \cdot (-0.5000)^2 = 0.001 \cdot (0.2500) = \mathbf{0.000250}$$
3. **Bias Correction $\hat{m}_1, \hat{v}_1$**:
   $$\hat{m}_1 = \frac{-0.0500}{1 - 0.9^1} = \frac{-0.0500}{0.100} = \mathbf{-0.5000}$$
   $$\hat{v}_1 = \frac{0.000250}{1 - 0.999^1} = \frac{0.000250}{0.0010} = \mathbf{0.2500}$$
4. **Adaptive Step Size**:
   $$\sqrt{\hat{v}_1} = \sqrt{0.2500} = 0.5000$$
   $$\text{Step} = \frac{\alpha}{\sqrt{\hat{v}_1} + \epsilon} \hat{m}_1 = \frac{0.001}{0.5000 + 10^{-8}} \cdot (-0.5000) = 0.002 \cdot (-0.5000) = \mathbf{-0.0010}$$
5. **Decoupled Weight Decay Penalty**:
   $$\alpha \lambda_{\text{wd}} W_0 = 0.001 \cdot 0.0001 \cdot 0.5000 = 0.00000005 \approx 0.0000$$
6. **New Weight $W_1$**:
   $$W_1 = W_0 - \text{Step} = 0.5000 - (-0.0010) = \mathbf{0.5010}$$

> **Conclusion**: Because both classification and localization indicated that increasing $W$ reduces loss, AdamW nudged $W$ from $0.5000$ to **$0.5010$**.

---

#### B) Step $t=1$ for Kendall Uncertainty Knob $s_{\text{cls}}$ (Initial $s_0 = 0.0000, g_1 = +0.3562$):
*(Note: Weight decay is NOT applied to uncertainty scalars)*
1. **Momentum $m_1$**:
   $$m_1 = 0.1 \cdot (+0.3562) = +0.03562 \implies \hat{m}_1 = \frac{0.03562}{0.1} = \mathbf{+0.3562}$$
2. **Velocity $v_1$**:
   $$v_1 = 0.001 \cdot (+0.3562)^2 = 0.001 \cdot 0.12688 = 0.0001269 \implies \hat{v}_1 = \frac{0.0001269}{0.001} = \mathbf{0.1269}$$
3. **Adaptive Step Size**:
   $$\sqrt{\hat{v}_1} = \sqrt{0.1269} \approx 0.3562$$
   $$\text{Step} = \frac{0.001}{0.3562} \cdot (+0.3562) = \mathbf{+0.0010}$$
4. **New Knob Value $s_1$**:
   $$s_1 = s_0 - \text{Step} = 0.0000 - 0.0010 = \mathbf{-0.0010}$$
5. **Resulting Effective Weight**:
   $$w_{\text{cls}} = \exp(-s_1) = \exp(-(-0.0010)) = \exp(0.0010) = \mathbf{1.0010}$$

> **Conclusion**: Because classification loss was low ($0.2877 < 1.0$), the positive gradient nudged $s$ downward, slightly increasing the priority multiplier $w_{\text{cls}}$ from $1.0000$ to $1.0010$.

---

#### C) Step $t=2$: Demonstrating the Power of Momentum
Suppose on the second batch, the gradient for $W$ is once again $g_2 = -0.5000$ (the slope continues in the same direction). Look at how momentum builds up:
$$m_2 = \beta_1 m_1 + (1 - \beta_1) g_2 = 0.9 \cdot (-0.0500) + 0.1 \cdot (-0.5000) = -0.0450 - 0.0500 = \mathbf{-0.0950}$$
Corrected momentum:
$$\hat{m}_2 = \frac{m_2}{1 - 0.9^2} = \frac{-0.0950}{1 - 0.81} = \frac{-0.0950}{0.19} = \mathbf{-0.5000}$$

If the gradient had suddenly reversed (e.g., noisy batch with $g_2 = +0.2000$), the momentum vector would have dampened the oscillation:
$$m_2 = 0.9 \cdot (-0.0500) + 0.1 \cdot (+0.2000) = -0.0450 + 0.0200 = \mathbf{-0.0250}$$
The previous momentum prevents the optimizer from violently whipping backward on noisy batches!

---

## 9. How Model Weights ($W$) and Uncertainty Knobs ($s$) Converge Together

Because both parameter sets are trained concurrently in a single computational graph, a harmonious co-adaptation occurs:

```
                                  CONVERGENCE DYNAMICS
                                  
    Early Epochs (Epoch 1 - 5)                  Late Epochs (Epoch 20 - 30)
    -------------------------                  ---------------------------
    - Model predictions are random             - Model classifies accurately (Acc > 70%)
    - Heatmaps drift to ribs & text            - Heatmaps locked to lung lesions
    - Gradients g_W are large & erratic        - Gradients g_W shrink (fine-tuning)
    - Uncertainties s_i adjust rapidly         - Uncertainties s_i reach steady-state:
                                                 sigma_i^2 ≈ L_i
    
    Resulting Saliency Energy: 49.21% (Drift)   Resulting Saliency Energy: 80.32% (Locked)
```

### 9.1 Empirical Verification: The Four Training Regimes
We trained `MedicalResNet18` across four distinct objective formulations on the real COVID-19 radiograph cohort (`results/real_dataset_training_results.csv`):

| Training Regime | Objective Formulation | Test Accuracy | Test F1 | Test Recall | Test ROI Dice | Pointing Game Hit Rate | Energy in ROI | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A** | Pure Classification ($\mathcal{L}_{\text{cls}}$ only) | 0.6897 | 0.8163 | 1.0000 | 0.4764 | 65.52% | 49.21% | 27.05 s |
| **Regime B** | **Combined Adaptive Loss (Proposed)** | **0.6897** | **0.8163** | **1.0000** | **0.6856** | **93.10%** | **80.32%** | **28.03 s** |
| **Regime C** | Alignment-Only ($\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{align}}$) | 0.6897 | 0.8163 | 1.0000 | 0.5799 | **96.55%** | **82.22%** | 27.24 s |
| **Regime D** | Dice-Only ($\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{dice}}$) | **0.7586** | **0.8511** | **1.0000** | **0.7107** | 82.76% | 69.36% | 26.88 s |

* **Why Regime B is the Champion**: While Regime D scored higher continuous Dice (0.7107), its energy inside the actual lesion dropped to 69.36% and Pointing Game fell to 82.76%. Regime B balances continuous Dice with tight peak energy (80.32%) and achieves a **93.10% Pointing Game accuracy** while maintaining **100% Sensitivity (Recall)** on viral lesions.

---

## 10. System-Level Multi-Objective Pareto Optimization (108 Configurations)

Once the deep learning model produces candidate bounding boxes, we encounter a second optimization challenge: **How should the entire telemedicine transmission pipeline be configured?**

### 10.1 The Four Competing Engineering Goals
1. **Maximize Diagnostic Quality**: $\text{SSIM}_{\text{roi}} \to \max$ (Radiologist needs pristine pathology).
2. **Minimize Wire Footprint**: $\text{Bytes} \to \min$ (Must fit through 48–64 kbps links).
3. **Minimize Triage Delay**: $T_{\text{first\_roi}} \to \min$ (Doctor needs critical data in $<1$ second).
4. **Minimize Cryptographic CPU Cost**: $\text{CPU}_{\text{crypto}} \to \min$ (Conserves edge battery).

### 10.2 Mathematical Definition of Pareto Dominance
Let candidate configuration $A$ and configuration $B$ have objective evaluation vectors $\mathbf{f}(A)$ and $\mathbf{f}(B)$.
Configuration $A$ strictly dominates configuration $B$ ($A \succ B$) if and only if:
$$\forall i \in \{1, 2, 3, 4\}, \quad f_i(A) \ge f_i(B) \quad \text{and} \quad \exists j \in \{1, 2, 3, 4\}, \quad f_j(A) > f_j(B)$$

If $A$ is better than or equal to $B$ in every metric, and strictly better in at least one metric, $B$ is **dominated** and discarded. The set of all configurations that are not dominated by any other configuration forms the **Pareto Frontier**.

```
    High Quality ^               [A] Pareto Optimal (High SSIM, Moderate Speed)
     (SSIM_roi)  |              / 
                 |            /
                 |          [C] Pareto Optimal (Moderate SSIM, Fast Speed)
                 |         / 
                 |       /     [B] Dominated (Worse SSIM & Slower than C!)
                 |     /
                 +--------------------------------------------->
                                               Fast Triage Speed (1 / T_roi)
```

### 10.3 The Grid Search Space (108 Configurations)
Implemented in [`src/optimization.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/optimization.py):
* Grad-CAM Cutoff Thresholds: $\tau \in \{0.25, 0.35, 0.50\}$ (3 values)
* Diagnostic ROI Quality: $q_{\text{roi}} \in \{85, 90, 95\}$ (3 values)
* Background Context Quality: $q_{\text{bg}} \in \{15, 25, 40\}$ (3 values)
* Authenticated Ciphers: AES-256-GCM vs Ascon-AEAD128 (2 options)
* Transport Schedulers: Priority Min-Heap vs FIFO (2 options)
$$\text{Total Tested Combinations} = 3 \times 3 \times 3 \times 2 \times 2 = 108 \text{ configurations}$$

### 10.4 The Result: 26 Non-Dominated Operating Points
Out of 108 configurations, 82 were mathematically dominated. Exactly **26 non-dominated configurations** remained (`results/pareto_frontier.csv`), establishing the global operating frontier:
* **The Clinical Champion**: $\tau=0.35, q_{\text{roi}}=90, q_{\text{bg}}=25$, AES-256-GCM, Priority Scheduling.
  * ROI PSNR: **48.31 dB**, ROI SSIM: **0.9891**.
  * Transmitted Bytes: **26,515 bytes** (89.89% data reduction).
  * Time-to-First-ROI over 64 kbps link: **0.391 seconds** (**6.21× faster than FIFO**).

---

## 11. Viva Defense Master Cheat Sheet (Tough Questions & Exact Answers)

Use this quick-reference sheet when defending the project before faculty advisors or external examiners:

### Q1: "Why didn't you just use standard Grad-CAM to find the ROI?"
> **Answer**: *"Standard Grad-CAM is detached post-hoc visualization; its computational graph is discarded, so it cannot influence training. If the model memorizes spurious shortcuts—like hospital corner labels or clavicle density (Saliency Drift)—standard Grad-CAM cannot fix it. By using Differentiable Grad-CAM with `create_graph=True`, second-order gradients $\frac{\partial \mathcal{L}_{\text{loc}}}{\partial W}$ flow back into the convolutional kernels, actively forcing the filters to detect lung consolidation textures."*

### Q2: "How did you balance classification loss and localization loss without manual tuning?"
> **Answer**: *"We implemented Kendall & Gal (CVPR 2018) homoscedastic uncertainty optimization. We treat multi-task weights as learnable observation noise parameters $s_i = \log \sigma_i^2$. By taking the derivative $\frac{\partial \mathcal{L}}{\partial s_i} = 0$, we proved mathematically that the minimum loss occurs when $\sigma_i^2 = \mathcal{L}_i$. The AdamW optimizer automatically adjusts these parameters during training, eliminating manual hyperparameter guesswork."*

### Q3: "What are the effective $\lambda$ and $\beta$ in your model?"
> **Answer**: *"They are the dynamic ratios of learned task weights: $\lambda_{\text{effective}} = \exp(s_{\text{cls}} - s_{\text{align}})$ and $\beta_{\text{effective}} = \exp(s_{\text{align}} - s_{\text{dice}})$. In `src/localization_loss.py`, `get_effective_weights()` monitors their convergence over training epochs."*

### Q4: "How does the AdamW optimizer update model weights and uncertainty knobs simultaneously?"
> **Answer**: *"In `src/train_real_models.py`, we pass `list(model.parameters()) + list(criterion.parameters())` into a single `optim.AdamW` instance. When `loss.backward()` executes, the autograd engine computes gradients $g_W$ for all 11.17M convolutional weights and $g_s$ for the uncertainty knobs. During `optimizer.step()`, AdamW updates each parameter using separate first-moment (momentum $m_t$) and second-moment (velocity $v_t$) vectors with adaptive learning rate scaling."*

### Q5: "Why did you choose Regime B over Regime D when Regime D had higher Dice?"
> **Answer**: *"Regime D achieved higher continuous Dice (0.7107) but its energy inside the radiologist ROI dropped to 69.36% and Pointing Game dropped to 82.76%, meaning its attention leaked outward into healthy ribs. Regime B achieved 93.10% Pointing Game Hit Rate and 80.32% Energy Confinement with 100% Sensitivity (Zero False Negatives) on viral lesions, providing superior clinical safety for cropping."*

### Q6: "What is Pareto Dominance in your networking module?"
> **Answer**: *"Configuration $A$ dominates configuration $B$ ($A \succ B$) if $A$ is at least as good as $B$ across all four objectives (SSIM, wire bytes, triage delay, crypto CPU) and strictly better in at least one. We evaluated 108 configurations via discrete-event simulation and extracted 26 non-dominated Pareto configurations, demonstrating that Priority Min-Heap scheduling achieves 0.391s triage without transmitting any extra bytes."*

---

*End of Technical Master Guide. Maintained under the MediLink-DL Project Repository at `docs/DEEP_LEARNING_AND_OPTIMIZATION_MASTER_GUIDE.md`.*

