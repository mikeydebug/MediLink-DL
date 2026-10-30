# MediLink-DL: Detailed Implementation Specification & Engineering Algorithms

**Document**: Low-Level Implementation Details, Binary Protocols, and Algorithmic Specifications  
**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors**: Mayank Soni (CS24B1031), Nandan (CS24B1034), Sumedha (AD24B1061)  
**Affiliation**: Department of Computer Science & Engineering, Indian Institute of Information Technology Raichur  
**Date**: October 31, 2026  

---

## 1. Dual-Stream ROI Compression Engine (`src/roi_compression.py`)

### 1.1 Dual-Stream Binary Wire Protocol
The dual-stream compression engine segments a single radiograph into two independently encoded JPEG bitstreams and packages them into a contiguous binary payload.

#### 26-Byte Header Layout (Big-Endian):
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
    
    # 4. Pack 26-byte header and concatenate streams
    header = struct.pack(">4sHHHBBHHHHII", b"MLDL", 1, H, W, q_roi, q_bg,
                         ymin, xmin, ymax, xmax, len(bg_bytes), len(roi_bytes))
    return header + bg_bytes + roi_bytes

def decompress_roi_aware(payload: bytes) -> np.ndarray:
    # 1. Unpack header
    magic, ver, H, W, q_roi, q_bg, ymin, xmin, ymax, xmax, bg_len, roi_len = struct.unpack(
        ">4sHHHBBHHHHII", payload[:26]
    )
    assert magic == b"MLDL"
    
    # 2. Decompress background and ROI images
    bg_img = decode_jpeg(payload[26 : 26 + bg_len])
    roi_img = decode_jpeg(payload[26 + bg_len : 26 + bg_len + roi_len])
    
    # 3. Composite high-fidelity patch into reconstructed frame
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
        
        # Weighted linear combination followed by ReLU
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Smooth spatial normalization
        cam = F.interpolate(cam, size=(512, 512), mode='bilinear', align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)
```

### 2.2 Continuous Alignment & Dice Loss
The localization loss guides the network’s visual attention directly toward ground-truth thoracic pathology:
$$\mathcal{L}_{\text{align}} = 1 - \frac{\sum_{i,j} M_{\text{CAM}}(i,j) \cdot M_{\text{GT}}(i,j)}{\sum_{i,j} M_{\text{CAM}}(i,j) + \sum_{i,j} M_{\text{GT}}(i,j) - \sum_{i,j} M_{\text{CAM}}(i,j) M_{\text{GT}}(i,j)}$$
$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{i,j} M_{\text{CAM}}(i,j) \cdot M_{\text{GT}}(i,j) + \epsilon}{\sum_{i,j} M_{\text{CAM}}(i,j)^2 + \sum_{i,j} M_{\text{GT}}(i,j)^2 + \epsilon}$$

---

## 3. Priority Network Transmission & Framing Protocol (`src/transmission.py`)

### 3.1 Packet Header Structure
Payloads are split into MTU-sized packets (default: 1024 bytes). Each packet contains an explicit wire frame:
```
Offset  Size  Type    Field Name      Description
0x00    2B    uint16  seq_num         Packet sequence index (0, 1, 2, ...)
0x02    2B    uint16  total_chunks    Total count of packets in transmission
0x04    1B    uint8   priority_type   Priority tier: 30 (Header), 20 (ROI), 5 (Background)
0x05    2B    uint16  payload_len     Byte length of chunk data segment
0x07    32B   bytes   sha256_chk      SHA-256 integrity hash of chunk payload
0x27    LB    bytes   payload_data    Chunk binary data (<= 1024 bytes)
```

### 3.2 Min-Heap Priority Scheduling
```python
class PriorityChunkScheduler:
    def __init__(self):
        self.queue = []  # Python heapq: entries (priority_key, seq_num, packet)

    def enqueue(self, packet):
        # Invert priority so that highest priority (e.g., 30) pops first (min-heap)
        priority_key = -packet.priority_type
        heapq.heappush(self.queue, (priority_key, packet.seq_num, packet))

    def dequeue_next(self):
        if self.queue:
            return heapq.heappop(self.queue)[2]
        return None
```

### 3.3 Channel Simulation & ARQ Retransmission
- **Serialization Delay**: $T_{\text{ser}} = \frac{(\text{len}(\text{packet}) + 64) \times 8}{R_{\text{bps}}}$.
- **One-Way Propagation Delay**: $T_{\text{prop}} = \frac{\text{RTT}}{2}$.
- **Packet Loss Simulation**: For each packet transmission:
  ```python
  if random.random() < p_loss:
      # Packet dropped: trigger timeout after 2 * RTT
      current_simulation_time += 2 * rtt
      # Reschedule packet for retransmission (ARQ)
      scheduler.retransmit(packet)
  else:
      current_simulation_time += (t_ser + t_prop)
  ```

---

## 4. Authenticated Encryption Implementations (`src/security.py`)

### 4.1 AES-256-GCM Envelope
- **Implementation**: Cryptography library (`cryptography.hazmat.primitives.ciphers.aead.AESGCM`).
- **Nonce Generation**: 96-bit CSPRNG (`os.urandom(12)`).
- **AAD Binding**: Additional Authenticated Data string contains study metadata:
  $$\text{AAD} = \text{SHA256}(\text{PatientID} \,\|\, \text{StudyUID} \,\|\, \text{Timestamp})$$
- **Wire Envelope**: $\text{Nonce}_{12\text{B}} \,\|\, \text{Ciphertext}_{N\text{B}} \,\|\, \text{Tag}_{16\text{B}}$.

### 4.2 NIST SP 800-232 Ascon-AEAD128
- **Pure Python Reference Implementation**: Permutation $p^{12}$ (initialization/finalization) and $p^6$ (intermediate processing) over a 320-bit state (five 64-bit words: $x_0, x_1, x_2, x_3, x_4$).
- **State Initialization**: Constant $IV = \text{0x80400c0600000000}$, key $K \in \{0, 1\}^{128}$, nonce $N \in \{0, 1\}^{128}$.
- **Permutation Operations**: Round constants addition ($c_r$), 5-bit S-box layer, linear diffusion layer with circular 64-bit rotations:
  $$\Sigma_0(x_0) = x_0 \oplus (x_0 \ggg 19) \oplus (x_0 \ggg 28)$$
  $$\Sigma_1(x_1) = x_1 \oplus (x_1 \ggg 61) \oplus (x_1 \ggg 39)$$
  $$\Sigma_2(x_2) = x_2 \oplus (x_2 \ggg 1) \oplus (x_2 \ggg 6)$$
  $$\Sigma_3(x_3) = x_3 \oplus (x_3 \ggg 10) \oplus (x_3 \ggg 17)$$
  $$\Sigma_4(x_4) = x_4 \oplus (x_4 \ggg 7) \oplus (x_4 \ggg 41)$$

---

## 5. Non-Hadamard Diffusion Engine (`src/crypto_research/diffusion/non_hadamard.py`)

### 5.1 Finite Field $\mathbb{F}_{2^4}$ Arithmetic
- **Irreducible Polynomial**: $P(x) = x^4 + x + 1$ (integer 0x13).
- **Multiplication**: Bitwise polynomial multiplication followed by reduction modulo 0x13:
```python
def gf_mult(a: int, b: int) -> int:
    res = 0
    for i in range(4):
        if (b >> i) & 1:
            res ^= (a << i)
    for i in range(6, 3, -1):
        if (res >> i) & 1:
            res ^= (0x13 << (i - 4))
    return res & 0x0F
```

### 5.2 Paar Heuristic-1 Common Subexpression Elimination (CSE)
The Paar-1 algorithm reduces the straight-line program XOR count for linear transformations $y = M_{\text{bin}} x$:
1. Construct the base dependency matrix where each row $i$ represents an output bit as a set of active input variables.
2. At each iteration, count the frequencies of all unordered pairs of active variables $(u, v)$ across all equations.
3. Select the pair $(u^*, v^*)$ with maximal frequency. If maximal frequency $\le 1$, terminate.
4. Introduce a new intermediate variable $w_k = u^* \oplus v^*$, substituting $w_k$ into all equations containing $\{u^*, v^*\}$.
5. Repeat until no subexpressions of frequency $\ge 2$ remain.
- **Execution on `ABC-MDS-OPT`**: Reduces direct XOR count from **64 to 49** (a **23.4% reduction**).
- **Execution on `ABC-MDS-MAXBIT`**: Reduces direct XOR count from **94 to 58** (a **38.3% reduction**).
- **Execution on `TBOI-16`**: Reduces direct XOR count from **128 to 60** (a **53.1% reduction**).

