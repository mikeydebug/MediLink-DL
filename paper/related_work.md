# Related Work

## 1. Medical Image Compression: Uniform vs. Region-of-Interest (ROI) Coding
Medical image compression is traditionally dichotomized into strictly lossless compression (e.g. JPEG-LS [ISO/IEC 14495-1], PNG, FLAC) and lossy compression (JPEG [ISO/IEC 10918-1], JPEG 2000 [ISO/IEC 15444-1], HEVC/H.265 intra-coding). While lossless techniques guarantee exact pixel preservation, their achievable compression ratios on 12-bit/16-bit digital radiography rarely exceed 2.5:1 to 3.5:1, which remains insufficient for severely constrained rural uplinks (e.g., 32–64 kbps).

To reconcile high compression efficiency with diagnostic preservation, Region-of-Interest (ROI) coding was standardized in JPEG 2000 Part 1 and Part 2 (Christopoulos et al., 2000). The MAXSHIFT method (Christopoulos et al., 2000; Anastassopoulos & Skodras) scales wavelet transform coefficients corresponding to the ROI above the background coefficients prior to bitplane entropy coding. While mathematically elegant, hardware and software adoption of JPEG 2000 in rural clinical nodes has been limited due to computational complexity, memory footprints, and proprietary codec licensing compared to universal baseline JPEG/WebP support.

## 2. Deep Learning and Weakly Supervised Saliency in Chest Radiography
Computer-aided triage in chest radiography expanded significantly with large-scale open datasets, such as NIH ChestX-ray14 (Wang et al., 2017) and Stanford CheXpert (Irvin et al., 2019). Dense convolutional networks (DenseNet-121) achieved notable benchmark classification performance on thoracic pathologies (Rajpurkar et al., 2017).

To understand model predictions without bounding-box annotations, visual explanation methods have been applied. Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) uses the gradient of the target class score flowing into the final convolutional layer to generate a coarse 2D localization map highlighting discriminative regions. 

*Critical Scholarly Distinction*:
In published literature, Grad-CAM is primarily used as an interpretability or post-hoc auditing mechanism. In our proposed framework, model-generated or radiologist-defined heatmaps are repurposed downstream as spatial priority weight masks to dynamically govern regional compression quantization budgets and transport scheduling. We emphasize that Grad-CAM identifies feature correlations learned by the neural network; it does not replace expert radiological ground truth.

## 3. Telemedicine Transport and Cryptographic Security
Transmitting medical imagery across intermittent, bandwidth-limited rural channels requires robust application-layer transport. Standard HTTP/TCP connections degrade severely under high packet loss and latency due to TCP congestion collapse and Head-of-Line (HoL) blocking.

Furthermore, medical privacy regulations (HIPAA Security Rule 45 CFR § 164.312; EU GDPR Art. 32) mandate end-to-end data confidentiality and integrity. The National Institute of Standards and Technology (NIST SP 800-38D; Dworkin, 2007) recommends Galois/Counter Mode (AES-GCM), an authenticated encryption with associated data (AEAD) construction providing both privacy and tamper detection.

## 4. Summary of Research Gap
While the constituent components—deep learning saliency, lossy compression, and AES encryption—are established in their respective domains, existing telemedicine literature lacks a unified, reproducible framework that systematically couples deep-learning saliency masks to dual-stream differential compression and priority-based network retransmission.

---

## References

1. **Christopoulos, C., Skodras, A., & Ebrahimi, T.** (2000). The JPEG2000 still image coding system: an overview. *IEEE Transactions on Consumer Electronics*, 46(4), 1103–1127. DOI: [10.1109/30.920468](https://doi.org/10.1109/30.920468).
2. **Dworkin, M.** (2007). Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC. *NIST Special Publication 800-38D*, National Institute of Standards and Technology.
3. **Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., ... & Ng, A. Y.** (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Human Computation and Crowdsourcing*, 33(01), 590–597.
4. **Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y.** (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv preprint arXiv:1711.05225*.
5. **Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626.
6. **Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M.** (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2097–2106.
7. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612. DOI: [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).
