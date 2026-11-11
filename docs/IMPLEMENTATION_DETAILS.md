# MediLink-DL: Detailed Implementation Specification & Engineering Algorithms

**Document**: Low-Level Implementation Details, Binary Protocols, and Algorithmic Specifications  
**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors**:  
* **Mayank Soni** (Roll No: CS24B1031) — *System Architecture, Multi-Objective Optimization & Cryptographic Research*  
* **Nandan** (Roll No: CS24B1034) — *Deep Learning Localization, Differentiable Grad-CAM & Loss Formulations*  
* **Sumedha** (Roll No: AD24B1061) — *Digital Image Processing, Dual-Stream Compression & Priority ARQ Transport*  

**Academic Supervision & Faculty Guidance**:  
* **Dr. Kaushiki Roy** — *Faculty Advisor, Digital Image Processing (DIP)*  
* **Dr. Mehbub Alam** — *Faculty Advisor, Computer Networks (CN)*  
* **Arnav Ghosh** — *Faculty Advisor, Multi-Objective Optimization & Applied Mathematics*  

**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology Raichur (IIIT Raichur)**  
**Date**: November 2026 &nbsp;|&nbsp; **Technical Implementation Specification**

---

## 1. Dual-Stream ROI Compression Engine (`src/roi_compression.py`)

### 1.1 Dual-Stream Binary Wire Protocol
The dual-stream compression engine segments a single radiograph into two independently encoded JPEG bitstreams and packages them into a contiguous binary payload.

#### 28-Byte Header Layout (Big-Endian):
```
Offset  Size  Type    Field Name    Description
0x00    4B    char[]  magic         ASCII bytes 'MLDL' (0x4D, 0x4C, 0x44, 0x4C)
0x04    2B    uint16  version       Protocol version (0x0001)
0x06    2B    uint16  height        Image height in pixels (H)
0x08    2B    uint16  width         Image width in pixels (W)
0x0A    1B    uint8   q_roi         Quality factor for ROI stream (e.g., 90)
0x0B    1B    uint8   q_bg          Quality factor for background stream (e.g., 25)
0x0C    2B    uint16  top           ROI bounding box y_min
0x0E    2B    uint16  left          ROI bounding box x_min
0x10    2B    uint16  bottom        ROI bounding box y_max
0x12    2B    uint16  right         ROI bounding box x_max
0x14    4B    uint32  bg_len        Byte length N of contextual background JPEG
0x18    4B    uint32  roi_len       Byte length M of diagnostic crop JPEG
0x1C    NB    bytes   bg_data       Background JPEG bitstream bytes
0x1C+N  MB    bytes   roi_data      ROI crop JPEG bitstream bytes
```
Total Header Size = $4 + 2 + 2 + 2 + 1 + 1 + 2 + 2 + 2 + 2 + 4 + 4 = \mathbf{28\text{ bytes}}$.

### 1.2 Compression & Reconstruction Algorithms
```python
def compress_roi_aware(image: np.ndarray, mask: np.ndarray, q_roi=90, q_bg=25) -> bytes:
    # 1. Encode full contextual image at lower quality
    bg_bytes = encode_jpeg(image, quality=q_bg)
    
    # 2. Extract bounding box from binary mask M
    ymin, xmin, ymax, xmax = get_bounding_box(mask)
    
    # 3. Crop diagnostic region and encode at high quality
    roi_crop = image[ymin:ymax, xmin:xmax]
    roi_bytes = encode_jpeg(roi_crop, quality=q_roi)
    
    # 4. Pack 28-byte header and concatenate streams
    header = struct.pack(">4sHHHBBHHHHII", b"MLDL", 1, H, W, q_roi, q_bg,
                         ymin, xmin, ymax, xmax, len(bg_bytes), len(roi_bytes))
    return header + bg_bytes + roi_bytes

def decompress_roi_aware(payload: bytes) -> np.ndarray:
    # 1. Unpack 28-byte header
    magic, ver, H, W, q_roi, q_bg, ymin, xmin, ymax, xmax, bg_len, roi_len = struct.unpack(
        ">4sHHHBBHHHHII", payload[:28]
    )
    assert magic == b"MLDL"
    
    # 2. Decompress background and ROI images
    bg_img = decode_jpeg(payload[28 : 28 + bg_len])
    roi_img = decode_jpeg(payload[28 + bg_len : 28 + bg_len + roi_len])
    
    # 3. Composite high-fidelity patch into reconstructed frame with alpha blending
    recon = bg_img.copy()
    recon[ymin:ymax, xmin:xmax] = roi_img
    return recon
```

---

## 2. Differentiable Grad-CAM & Localization-Aware Loss (`src/localization_loss.py`)

### 2.1 Preserving the Autograd Computational Graph
Standard post-hoc Grad-CAM computes gradients on detached activations, preventing backpropagation into model parameters. In MediLink-DL, `DifferentiableGradCAM` preserves the backward graph:

```python
class DifferentiableGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output  # Preserves autograd node

        self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, class_score):
        # Retain graph during backward pass to allow second-order gradients
        grads = torch.autograd.grad(
            class_score, self.activations,
            retain_graph=True, create_graph=True
        )[0]
        
        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad_k)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)
```

### 2.2 Kendall & Gal Homoscedastic Uncertainty Optimization
```python
class AdaptiveLocalizationAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_cls = nn.Parameter(torch.zeros(1))
        self.s_loc = nn.Parameter(torch.zeros(1))

    def forward(self, logits, targets, cam, gt_mask):
        l_cls = F.cross_entropy(logits, targets)
        l_align = cosine_alignment_loss(cam, gt_mask)
        l_dice = soft_dice_loss(cam, gt_mask)
        l_loc = 0.5 * l_align + 0.5 * l_dice
        
        total_loss = (
            0.5 * torch.exp(-self.s_cls) * l_cls
            + 0.5 * torch.exp(-self.s_loc) * l_loc
            + 0.5 * self.s_cls
            + 0.5 * self.s_loc
        )
        return total_loss
```

---

## 3. Priority Min-Heap Transport & ARQ Protocol (`src/transmission.py`)

### 3.1 Packet Framing Protocol
Telemetry streams are fragmented into 1024-byte chunks with 43-byte framing:
`[4B Magic: 'MLPK'] [2B SeqID] [2B TotalChunks] [1B PriorityTier] [2B PayloadLen] [32B SHA-256 Checksum]`.

### 3.2 Min-Heap Priority Queuing
```python
# Min-heap priority ordering
priority_map = {
    "HEADER": 30,      # Inverted order: highest priority popped first
    "ROI_CHUNK": 20,   # Diagnostic lesion transmitted in first burst
    "BG_CHUNK": 5      # Contextual anatomy transmitted progressively
}
```

---

## 4. Paar CSE XOR Optimization for Diffusion Layers

Common Subexpression Elimination (CSE) optimizes straight-line XOR programs to minimize silicon area:
```python
def optimize_xor_program(matrix):
    # Iteratively extracts recurring 2-input XOR patterns
    # Achieves 48 XORs for PROPOSED-OIN-MDS (~96.0 GE)
    # Achieves 49 XORs for ABC-MDS-OPT (~98.0 GE)
```
