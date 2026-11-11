"""generate_pdf.py: Compile professional academic research paper PDF using ReportLab Platypus.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Updated to reflect real-world clinical dataset (Cohen et al., 2020),
Kendall & Gal (CVPR 2018) Homoscedastic Uncertainty Weighting,
NIST SP 800-232 Cryptographic Lab, and Faculty Supervision:
- Dr. Kaushiki Roy (Digital Image Processing)
- Dr. Mehbub Alam (Computer Networks)
- Arnav Ghosh (Multi-Objective Optimization)
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total page count and add running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Don't draw header on first page
        if self._pageNumber > 1:
            self.drawString(54, 750, "MediLink-DL: Secure & Bandwidth-Adaptive Telemedicine Transmission")
            self.drawRightString(612 - 54, 750, "IIIT Raichur Academic Research Paper")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)
            
        # Footer on all pages
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — IIIT RAICHUR RESEARCH CAPSTONE")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 612 - 54, 46)
        self.restoreState()


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "AcademicTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15.5,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8
    )
    
    author_style = ParagraphStyle(
        "AcademicAuthor",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3
    )
    
    supervision_style = ParagraphStyle(
        "AcademicSupervision",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=3
    )
    
    context_style = ParagraphStyle(
        "AcademicContext",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=8
    )
    
    abstract_body = ParagraphStyle(
        "AbstractBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b")
    )
    
    h1_style = ParagraphStyle(
        "AcademicH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "AcademicBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5
    )
    
    caption_style = ParagraphStyle(
        "AcademicCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceBefore=3,
        spaceAfter=6
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor("#0f172a")
    )
    
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []
    
    # Title & Authors & Supervision Header
    story.append(Paragraph("A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine", title_style))
    story.append(Paragraph("<b>Mayank Soni</b> (CS24B1031) &nbsp;&bull;&nbsp; <b>Nandan</b> (CS24B1034) &nbsp;&bull;&nbsp; <b>Sumedha</b> (AD24B1061)", author_style))
    story.append(Paragraph("<b>Faculty Supervisors</b>: <b>Dr. Kaushiki Roy</b> (DIP) &nbsp;|&nbsp; <b>Dr. Mehbub Alam</b> (CN) &nbsp;|&nbsp; <b>Arnav Ghosh</b> (Optimization)", supervision_style))
    story.append(Paragraph("Department of Computer Science & Engineering &nbsp;|&nbsp; Indian Institute of Information Technology Raichur &nbsp;|&nbsp; November 2026", context_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#94a3b8"), spaceAfter=8))
    
    # Abstract Box
    abstract_text = (
        "<b>Abstract</b>—Rural primary healthcare centers in resource-constrained geographies lack on-site specialist radiologists "
        "and rely on fragile, bandwidth-throttled communication uplinks (2G/EDGE, weak 3G, or satellite links at 48–128 kbps). "
        "Transmitting uncompressed high-resolution digital radiographs (10–35 MB) is impractical over these links, while standard "
        "uniform lossy compression degrades subtle diagnostic features. This paper introduces <b>MediLink-DL</b>, an integrated, "
        "bandwidth-adaptive, and cryptographically secure framework combining: (i) clinical data curation on the Cohen et al. (2020) "
        "cohort (166 radiographs, 110 patients) with strict patient-level non-overlapping stratification (P_train &cap; P_val = &empty;, P_train &cap; P_test = &empty;); "
        "(ii) a single-channel <b>MedicalResNet18</b> with <b>Differentiable Grad-CAM</b> (create_graph=True) and Kendall &amp; Gal (CVPR 2018) "
        "homoscedastic uncertainty optimization, achieving a <b>93.10% Pointing Game Hit Rate</b>, <b>80.32% Energy in ROI</b>, and <b>100% Sensitivity</b>; "
        "(iii) a dual-stream Region-of-Interest (ROI) compression pipeline achieving a <b>9.89x compression ratio</b> (89.89% data reduction) "
        "with diagnostic <b>ROI PSNR &gt; 48.3 dB</b> and <b>ROI SSIM &gt; 0.989</b> (+10.13 dB gain over matched uniform JPEG); "
        "(iv) standardized authenticated encryption supporting AES-256-GCM, NIST SP 800-232 Ascon-AEAD128, and MLMC-128 research candidate with 100% bit-flip tamper rejection; "
        "(v) application-layer priority scheduling delivering diagnostic ROIs in <b>0.391 seconds</b> (&gt;6.2x faster than FIFO); and "
        "(vi) cross-layer multi-objective Pareto optimization identifying 26 non-dominated configurations across 108 evaluated parameter sets."
    )
    keywords_text = "<b>Keywords</b>: Telemedicine, Digital Image Processing, Differentiable Grad-CAM, Kendall Uncertainty, Patient-Level Splitting, Authenticated Encryption, NIST SP 800-232, Priority Scheduling, Pareto Optimization."
    
    abstract_table = Table(
        [[Paragraph(abstract_text, abstract_body)],
         [Paragraph(keywords_text, abstract_body)]],
        colWidths=[504]
    )
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (0,0), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 6))
    
    # 1. Introduction
    story.append(Paragraph("1. Introduction & Clinical Motivation", h1_style))
    story.append(Paragraph(
        "Digital radiography (CR/DR) is the primary diagnostic frontline for detecting acute pulmonary consolidations, "
        "pneumothoraces, and respiratory infections [1]. In urban tertiary centers, gigabit fiber and PACS facilitate instantaneous "
        "DICOM transfer. In rural clinics, however, transmission is throttled to 48–128 kbps wireless uplinks with high packet drop rates (2–10%) "
        "and multi-hop latencies. Transmitting raw uncompressed studies (10–35 MB) incurs multi-minute delays, while standard uniform lossy "
        "compression introduces blocking artifacts uniformly across the visual field. Consumer tools such as WhatsApp cannot be used due "
        "to non-compliance with HIPAA Security Rules, unverified quantization loss, and high 2G cellular latency. MediLink-DL resolves "
        "this by uniting deep learning localization, dual-stream compression, authenticated encryption, and priority network scheduling.",
        body_style
    ))
    
    # 2. Real Medical Dataset
    story.append(Paragraph("2. Clinical Medical Dataset & Leakage-Free Stratification", h1_style))
    story.append(Paragraph(
        "MediLink-DL is evaluated on genuine clinical radiographs from the open-access <b>COVID-19 Chest X-ray Dataset Initiative</b> "
        "(Cohen et al., 2020) [2] with ground-truth bilateral lung bounding boxes annotated by <code>imageannotation.ai</code>. "
        "The verified cohort comprises 166 clinical digital radiographs across 110 unique patients. To eliminate data leakage "
        "(where images from the same individual appear in both training and test subsets, artificially inflating generalization performance), "
        "we enforce strict <b>patient-level stratification</b>: 77 training patients (111 CXRs), 16 validation patients (26 CXRs), "
        "and 17 held-out test patients (29 CXRs). Verification confirmed exactly zero patient overlap across partitions "
        "(P_train &cap; P_val = &empty;, P_train &cap; P_test = &empty;, P_val &cap; P_test = &empty;).",
        body_style
    ))
    
    # 3. Deep Learning Architecture & Differentiable Grad-CAM
    story.append(Paragraph("3. Deep Learning Localization & Kendall Uncertainty Weighting", h1_style))
    story.append(Paragraph(
        "Standard Grad-CAM is a post-hoc visualization technique where activations and gradients are detached. In MediLink-DL, "
        "<b>DifferentiableGradCAM</b> preserves the backward autograd computational graph (<code>create_graph=True</code>), allowing "
        "second-order gradients to backpropagate into convolutional weights. We implement <b>MedicalResNet18</b>, adapting conv1 for "
        "1-channel grayscale chest radiographs. To automatically balance classification loss and localization penalties without manual "
        "hyperparameter heuristics, we formulate the Kendall &amp; Gal (CVPR 2018) [4] homoscedastic uncertainty loss:",
        body_style
    ))
    story.append(Paragraph(
        "<i>L_total(w, s_1, s_2) = 0.5 &middot; exp(-s_1) &middot; L_cls(w) + 0.5 &middot; exp(-s_2) &middot; L_loc(w) + 0.5 &middot; s_1 + 0.5 &middot; s_2</i>",
        ParagraphStyle("Eq", parent=body_style, fontName="Courier-Oblique", fontSize=7.5, alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        "where s_1 = log(&sigma;_1^2) and s_2 = log(&sigma;_2^2) are learnable log-variance parameters optimized via autograd AdamW. "
        "Four experimental regimes were trained on Apple Silicon MPS and evaluated on held-out test CXRs:",
        body_style
    ))
    
    # Table 1: Deep Learning Regimes
    story.append(Paragraph("<b>Table 1: Four-Regime Deep Learning Localization Comparison (results/real_dataset_training_results.csv)</b>", caption_style))
    t1_data = [
        [Paragraph("Regime / Objective", table_header), Paragraph("Test Acc", table_header), Paragraph("Test F1", table_header), Paragraph("Test Recall", table_header), Paragraph("ROI Dice", table_header), Paragraph("ROI IoU", table_header), Paragraph("Pointing Hit Rate", table_header), Paragraph("Energy in ROI", table_header)],
        [Paragraph("Regime A: Cls-Only (L_cls)", table_cell_bold), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("1.0000", table_cell), Paragraph("0.4764", table_cell), Paragraph("0.3571", table_cell), Paragraph("65.52%", table_cell), Paragraph("49.21%", table_cell)],
        [Paragraph("<b>Regime B: Combined (Proposed)</b>", table_cell_bold), Paragraph("<b>0.6897</b>", table_cell_bold), Paragraph("<b>0.8163</b>", table_cell_bold), Paragraph("<b>1.0000</b>", table_cell_bold), Paragraph("<b>0.6856</b>", table_cell_bold), Paragraph("<b>0.5395</b>", table_cell_bold), Paragraph("<b>93.10%</b>", table_cell_bold), Paragraph("<b>80.32%</b>", table_cell_bold)],
        [Paragraph("Regime C: Align-Only (L_align)", table_cell_bold), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("1.0000", table_cell), Paragraph("0.5799", table_cell), Paragraph("0.4243", table_cell), Paragraph("<b>96.55%</b>", table_cell_bold), Paragraph("<b>82.22%</b>", table_cell_bold)],
        [Paragraph("Regime D: Dice-Only", table_cell_bold), Paragraph("<b>0.7586</b>", table_cell_bold), Paragraph("<b>0.8511</b>", table_cell_bold), Paragraph("1.0000", table_cell), Paragraph("<b>0.7107</b>", table_cell_bold), Paragraph("<b>0.5704</b>", table_cell_bold), Paragraph("82.76%", table_cell), Paragraph("69.36%", table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[120, 42, 42, 48, 62, 62, 72, 56])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))
    
    # Embed DL Comparison Figure
    dl_fig_path = Path("results/real_cxr_gradcam_comparison.png")
    if dl_fig_path.exists():
        story.append(RLImage(str(dl_fig_path), width=504, height=125))
        story.append(Paragraph("<b>Figure 1</b>: Qualitative Localization on Real Test CXRs: (Col 1) Original Radiograph, (Col 2) Ground-Truth Lung Mask, (Col 3) Baseline Classifier Grad-CAM exhibiting saliency drift, and (Col 4) Proposed Localization-Aware Grad-CAM tightly locked to thoracic consolidations.", caption_style))
        story.append(Spacer(1, 4))
        
    # 4. Dual-Stream ROI Compression
    story.append(Paragraph("4. Dual-Stream ROI Compression on Real CXRs", h1_style))
    story.append(Paragraph(
        "A genuine human chest radiograph (512x512, 262,144 bytes uncompressed) is decomposed via dual-stream coding: the background context "
        "is compressed at q_bg=25, while the diagnostic lung ROI is preserved at q_roi=90. Streams are serialized into a 28-byte binary container "
        "header ('MLDL') and reassembled at the receiver using distance-transform alpha blending (&alpha;=0.45) across a 4-pixel border:",
        body_style
    ))
    
    # Table 2: Compression Metrics
    t2_data = [
        [Paragraph("Coding Scheme", table_header), Paragraph("Quality Setting", table_header), Paragraph("Transmitted Bytes", table_header), Paragraph("Compression Ratio", table_header), Paragraph("ROI PSNR", table_header), Paragraph("ROI SSIM", table_header), Paragraph("Global PSNR", table_header)],
        [Paragraph("<b>ROI-Aware Dual-Stream</b>", table_cell_bold), Paragraph("q_roi=90, q_bg=25", table_cell), Paragraph("<b>26,515 B</b>", table_cell_bold), Paragraph("<b>9.89x (89.9% drop)</b>", table_cell_bold), Paragraph("<b>48.31 dB</b>", table_cell_bold), Paragraph("<b>0.9891</b>", table_cell_bold), Paragraph("42.10 dB", table_cell)],
        [Paragraph("Rate-Matched Baseline", table_cell_bold), Paragraph("q=32 (Matched Bytes)", table_cell), Paragraph("26,035 B", table_cell), Paragraph("10.07x", table_cell), Paragraph("38.18 dB", table_cell), Paragraph("0.9312", table_cell), Paragraph("39.45 dB", table_cell)],
        [Paragraph("High-Quality Uniform JPEG", table_cell_bold), Paragraph("q=90", table_cell), Paragraph("98,412 B", table_cell), Paragraph("2.66x", table_cell), Paragraph("48.31 dB", table_cell), Paragraph("0.9897", table_cell), Paragraph("49.11 dB", table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[110, 80, 75, 85, 54, 50, 50])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))
    
    # 5. Security & Computer Networks Transport
    story.append(Paragraph("5. Authenticated Encryption & Priority Network Transport", h1_style))
    story.append(Paragraph(
        "To protect patient confidentiality and integrity, payloads are encrypted with fresh nonces using either <b>AES-256-GCM</b> "
        "(NIST SP 800-38D, GHASH tag over GF(2^128)) or lightweight <b>Ascon-AEAD128</b> (NIST SP 800-232, 320-bit permutation). "
        "Hospital metadata is bound via Associated Data (AAD = StudyUID || PatientID || Timestamp) enforcing <b>Zero-Plaintext Release</b>. "
        "In our isolated cryptographic laboratory, we also benchmarked <b>MLMC-128</b>, an educational 128-bit block cipher utilizing PRESENT "
        "S-boxes and circulant MDS diffusion (SAC=49.92%, B=5). Over a 64 kbps cellular link, priority min-heap scheduling delivers diagnostic "
        "ROI chunks in <b>0.391 seconds</b> vs <b>2.430 seconds</b> for FIFO transport (&gt;6.2x faster triage):",
        body_style
    ))
    
    # Table 3: Crypto Benchmarks
    t3_data = [
        [Paragraph("Cipher / Protocol", table_header), Paragraph("Architecture / Standard", table_header), Paragraph("Enc Time (1KB)", table_header), Paragraph("Dec Time (1KB)", table_header), Paragraph("Throughput", table_header), Paragraph("Bit-Flip Tamper Detection", table_header)],
        [Paragraph("<b>AES-256-GCM</b>", table_cell_bold), Paragraph("NIST SP 800-38D (ARMv8 Crypto)", table_cell), Paragraph("<b>2.2 &mu;s</b>", table_cell_bold), Paragraph("<b>2.4 &mu;s</b>", table_cell_bold), Paragraph("<b>465.4 MB/s</b>", table_cell_bold), Paragraph("<b>100.0% Detected</b>", table_cell_bold)],
        [Paragraph("<b>Ascon-AEAD128</b>", table_cell_bold), Paragraph("NIST SP 800-232 (IoT Standard)", table_cell), Paragraph("412.0 &mu;s", table_cell), Paragraph("418.0 &mu;s", table_cell), Paragraph("2.48 MB/s", table_cell), Paragraph("<b>100.0% Detected</b>", table_cell_bold)],
        [Paragraph("<b>MLMC-128 (Raw Block)</b>", table_cell_bold), Paragraph("Research Prototype (Python)", table_cell), Paragraph("2,140.0 &mu;s", table_cell), Paragraph("2,190.0 &mu;s", table_cell), Paragraph("0.47 MB/s", table_cell), Paragraph("N/A (Block Only)", table_cell)],
        [Paragraph("<b>MLMC-128 (AEAD Exp)</b>", table_cell_bold), Paragraph("Research Candidate (CTR+GHASH)", table_cell), Paragraph("7,889.0 &mu;s", table_cell), Paragraph("7,920.0 &mu;s", table_cell), Paragraph("0.13 MB/s", table_cell), Paragraph("<b>100.0% Detected</b>", table_cell_bold)],
    ]
    t3 = Table(t3_data, colWidths=[105, 115, 65, 65, 65, 89])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    # 6. Cryptographic Diffusion Research
    story.append(Paragraph("6. Cryptographic Diffusion Architecture & Involutory MDS Matrices", h1_style))
    story.append(Paragraph(
        "In our cryptographic testbed, we investigated linear diffusion layers for lightweight block ciphers over GF(2^4)/(x^4+x+1). "
        "Beyond the optimal Hadamard instantiation <b>PROPOSED-OIN-MDS</b> = Had(1,8,13,5) (branch=5, bit-branch=8, 48 Paar XORs), "
        "we constructed novel Alternating Block-Circulant matrices <b>ABC-MDS-OPT</b> (branch=5, 49 Paar XORs) and <b>ABC-MDS-MAXBIT</b> "
        "(branch=5, bit-branch=7, 58 Paar XORs), as well as a direct binary involution <b>TBOI-16</b> (K4 (x) K4, branch=4 Near-MDS, 60 Paar XORs). "
        "Formal impossibility theorems were verified: Theorem 1 (Hankel non-existence over GF(2^m)), Proposition 2 (Cauchy field collapse), "
        "Theorem 3 (Galois matrix symmetry obstruction), and Theorem 4 (Tensor branch upper bound B <= 4).",
        body_style
    ))
    
    # 7. Cross-Layer Optimization
    story.append(Paragraph("7. Cross-Layer Multi-Objective Optimization & Pareto Frontier", h1_style))
    story.append(Paragraph(
        "A full-pipeline grid search evaluated 108 end-to-end parameter configurations across DL, DIP, CN, and Cryptography on genuine radiographs. "
        "We extracted <b>26 non-dominated Pareto configurations</b> balancing ROI SSIM, wire bytes, triage latency, and cryptographic CPU cost.",
        body_style
    ))
    
    pareto_fig_path = Path("results/pareto_frontier.png")
    if pareto_fig_path.exists():
        story.append(RLImage(str(pareto_fig_path), width=504, height=125))
        story.append(Paragraph("<b>Figure 2</b>: Empirical Pareto Frontier (108 configurations): Demonstrates dominant trade-offs between ROI delivery latency, compression quality, and wire bandwidth across priority and FIFO regimes.", caption_style))
        story.append(Spacer(1, 4))
        
    # 8. Acknowledgements & Faculty Supervision
    story.append(Paragraph("8. Acknowledgements & Faculty Supervision", h1_style))
    story.append(Paragraph(
        "The authors express sincere gratitude to our faculty mentors at the <b>Indian Institute of Information Technology Raichur (IIIT Raichur)</b> "
        "for their dedicated guidance, technical oversight, and rigorous evaluation throughout this capstone research: "
        "<b>Dr. Kaushiki Roy</b> (Assistant Professor, CSE) for domain supervision in Digital Image Processing, differential JPEG compression, and structural distortion metrics; "
        "<b>Dr. Mehbub Alam</b> (Assistant Professor, CSE) for domain supervision in Computer Networks, priority packet scheduling, and lossy wireless channel modeling; and "
        "<b>Arnav Ghosh</b> for mathematical supervision in multi-objective Pareto optimization, loss formulation autograd dynamics, and algebraic diffusion structures.",
        body_style
    ))
    
    # 9. Conclusion & Limitations
    story.append(Paragraph("9. Discussion, Limitations & Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Limitations</b>: (1) Current evaluation spans 166 radiographs across 110 patients; regulatory clinical clearance requires multi-center "
        "trials across 20,000+ studies. (2) The framework currently targets 2D projection radiographs rather than 3D volumetric CT/MRI studies. "
        "(3) Standard-cell GE figures are pre-synthesis analytical estimates (2.0 GE/XOR). (4) MediLink-DL is an academic research prototype, not FDA-cleared software.<br/>"
        "<b>Conclusion</b>: MediLink-DL successfully demonstrates an end-to-end, reproducible framework uniting differentiable deep learning localization, "
        "dual-stream compression, authenticated AEAD ciphers, priority transport, and verified involutory diffusion architectures, providing a robust template for rural telemedicine.",
        body_style
    ))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] World Health Organization, <i>Telemedicine: Opportunities and Developments in Member States</i>, Geneva: WHO Press, 2010.",
        "[2] J. P. Cohen, P. Morrison, and L. Dao, \"COVID-19 Image Data Collection,\" <i>arXiv:2003.11597</i>, 2020.",
        "[3] R. R. Selvaraju, et al., \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in <i>Proc. IEEE ICCV</i>, 2017.",
        "[4] A. Kendall, Y. Gal, and R. Cipolla, \"Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,\" in <i>Proc. IEEE CVPR</i>, 2018.",
        "[5] M. Dworkin, \"Galois/Counter Mode (GCM) and GMAC,\" <i>NIST SP 800-38D</i>, 2007.",
        "[6] C. Dobraunig, M. Eichlseder, et al., \"Ascon v1.2: Lightweight Authenticated Encryption,\" <i>NIST SP 800-232</i>, 2024.",
        "[7] Z. Wang, A. C. Bovik, et al., \"Image quality assessment: from error visibility to structural similarity,\" <i>IEEE TIP</i>, vol. 13, no. 4, 2004.",
        "[8] P. Rajpurkar, et al., \"CheXNet: Radiologist-level pneumonia detection on chest X-rays,\" <i>arXiv:1711.05225</i>, 2017.",
        "[9] M. Sajadieh, M. Wei, and E. Pasalic, \"On construction of involutory MDS matrices,\" <i>IEEE TIT</i>, vol. 58, no. 9, 2012.",
        "[10] C. Beierle, A. Canteaut, and G. Leander, \"Lightweight Invertible Diffusion Layers,\" <i>IACR ToSC</i>, vol. 2018, no. 3, 2018.",
        "[11] V. Cauchois and P. Loidreau, \"On the number of involutory MDS matrices over finite fields,\" <i>Des. Codes Cryptogr.</i>, 2019."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", parent=body_style, fontSize=6.5, leading=8.5, spaceAfter=2)))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {output_path.resolve()}")


if __name__ == "__main__":
    out_file = Path("paper/MediLink_DL_Research_Paper.pdf")
    build_pdf(out_file)
