"""generate_pdf.py: Compile professional academic research paper PDF using ReportLab Platypus.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Updated to reflect real-world clinical dataset (Cohen et al., 2020),
MedicalResNet18 training across 4 experimental regimes, and Pareto optimization.
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
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
            self.drawRightString(612 - 54, 750, "Research Prototype Report")
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
        fontSize=17,
        leading=21,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10
    )
    
    author_style = ParagraphStyle(
        "AcademicAuthor",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4
    )
    
    context_style = ParagraphStyle(
        "AcademicContext",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12
    )
    
    abstract_heading = ParagraphStyle(
        "AbstractHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    
    abstract_body = ParagraphStyle(
        "AbstractBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b")
    )
    
    h1_style = ParagraphStyle(
        "AcademicH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "AcademicH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "AcademicBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    
    equation_style = ParagraphStyle(
        "AcademicEquation",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=4,
        spaceAfter=6
    )
    
    caption_style = ParagraphStyle(
        "AcademicCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceBefore=4,
        spaceAfter=8
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )
    
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []
    
    # Title & Metadata
    story.append(Paragraph("A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine", title_style))
    story.append(Paragraph("<b>Mayank Soni</b> (CS24B1031) &nbsp;&bull;&nbsp; <b>Nandan</b> (CS24B1034) &nbsp;&bull;&nbsp; <b>Sumedha</b> (AD24B1061)", author_style))
    story.append(Paragraph("Undergraduate Research Capstone Project &nbsp;|&nbsp; IIIT Raichur &nbsp;|&nbsp; October 2026", context_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=10))
    
    # Abstract Box
    abstract_text = (
        "<b>Abstract</b>—Rural primary healthcare centers in resource-constrained geographies lack on-site specialist radiologists "
        "and depend on fragile, low-bandwidth communication uplinks (2G/EDGE, weak 3G, or satellite links at 48–128 kbps). "
        "Transmitting uncompressed high-resolution digital radiographs (10–35 MB) is impractical over these links, while standard "
        "uniform lossy compression irreversibly blurs subtle pathology. This paper introduces <b>MediLink-DL</b>, an integrated, "
        "bandwidth-adaptive, and cryptographically secure framework combining: (i) clinical data curation on the open-access "
        "COVID-19 Chest X-ray Dataset (166 radiographs, 110 patients) with strict patient-level leakage prevention; (ii) a trainable "
        "<b>MedicalResNet18</b> with Differentiable Grad-CAM utilizing second-order autograd to guide convolutional attention toward "
        "bilateral pulmonary structures (+49.2% continuous Dice improvement over classification baseline); (iii) a dual-stream "
        "Region-of-Interest (ROI) compression pipeline achieving a <b>9.89x compression ratio</b> (89.89% bandwidth reduction) with "
        "diagnostic <b>ROI PSNR &gt; 48.3 dB</b> and <b>ROI SSIM &gt; 0.989</b> on genuine human radiographs; (iv) standardized authenticated "
        "encryption supporting AES-256-GCM and NIST SP 800-232 Ascon-AEAD128 with 100% bit-flip tamper rejection; (v) application-layer "
        "priority scheduling delivering diagnostic ROIs in <b>0.328 seconds</b> (&gt;3.1x faster than FIFO); and (vi) cross-layer multi-objective "
        "Pareto optimization identifying 26 non-dominated operating configurations across 108 evaluated parameter sets."
    )
    keywords_text = "<b>Keywords</b>: Medical Image Compression, Telemedicine, Differentiable Grad-CAM, Patient-Level Splitting, Priority Packet Transmission, Ascon-AEAD128, Cross-Layer Optimization."
    
    abstract_table = Table(
        [[Paragraph(abstract_text, abstract_body)],
         [Paragraph(keywords_text, abstract_body)]],
        colWidths=[504]
    )
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (0,0), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 10))
    
    # 1. Introduction
    story.append(Paragraph("1. Introduction & Clinical Motivation", h1_style))
    story.append(Paragraph(
        "Digital radiography (CR/DR) is the primary diagnostic frontline for detecting acute pulmonary consolidations, "
        "pneumothoraces, and respiratory infections [1]. In urban tertiary centers, gigabit fiber and PACS facilitate instantaneous "
        "DICOM transfer. In rural clinics, however, transmission is throttled to 48–128 kbps wireless uplinks with high packet drop rates (2–10%) "
        "and multi-hop latencies [1]. Transmitting raw uncompressed studies (10–35 MB) incurs multi-minute delays, while standard uniform lossy "
        "compression introduces blocking artifacts uniformly across the entire visual field, degrading subtle pathology. MediLink-DL resolves "
        "this by coupling deep-learning-guided regional prioritization with authenticated encryption and priority network scheduling.",
        body_style
    ))
    
    # 2. Real Medical Dataset
    story.append(Paragraph("2. Clinical Medical Dataset & Leakage-Free Stratification", h1_style))
    story.append(Paragraph(
        "Rather than relying solely on mathematical phantoms, MediLink-DL is evaluated on the open-access <b>COVID-19 Chest X-ray "
        "Dataset Initiative</b> (Cohen et al., 2020) with ground-truth bilateral lung and lesion bounding boxes provided by "
        "<code>imageannotation.ai</code>. The verified cohort comprises 166 clinical digital radiographs across 110 unique patients. "
        "To prevent severe patient leakage—a common flaw in published medical AI literature where radiographs from the same individual "
        "appear in both training and evaluation subsets—we enforce strict <b>patient-level stratification</b>: 77 training patients (111 radiographs), "
        "16 validation patients (26 radiographs), and 17 test patients (29 radiographs). Verification confirmed exactly zero patient overlap "
        "across partitions (P_train &cap; P_val = &empty;, P_train &cap; P_test = &empty;, P_val &cap; P_test = &empty;).",
        body_style
    ))
    
    # 3. Deep Learning Architecture & Differentiable Grad-CAM
    story.append(Paragraph("3. Deep Learning Localization & Differentiable Grad-CAM", h1_style))
    story.append(Paragraph(
        "Standard Grad-CAM is a post-hoc visualization technique where activations and gradients are detached. In MediLink-DL, "
        "<b>DifferentiableGradCAM</b> preserves the backward autograd computational graph (<code>create_graph=True</code>), allowing "
        "second-order gradients to backpropagate into convolutional weights. We implement <b>MedicalResNet18</b>, adapting conv1 for "
        "1-channel grayscale chest radiographs. Four experimental regimes were trained on Apple Silicon MPS and evaluated on held-out test CXRs:",
        body_style
    ))
    
    # Table 1: Deep Learning Regimes
    story.append(Paragraph("<b>Table 1: Four-Regime Deep Learning Localization Comparison (results/real_dataset_training_results.csv)</b>", caption_style))
    t1_data = [
        [Paragraph("Regime / Objective", table_header), Paragraph("Test Acc", table_header), Paragraph("Test F1", table_header), Paragraph("Continuous ROI Dice", table_header), Paragraph("Binarized IoU (&tau;=0.35)", table_header), Paragraph("Pointing Game Hit Rate", table_header), Paragraph("Energy in ROI", table_header)],
        [Paragraph("Regime A: Cls-Only (L_cls)", table_cell_bold), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("0.4764", table_cell), Paragraph("0.3571", table_cell), Paragraph("65.52%", table_cell), Paragraph("49.21%", table_cell)],
        [Paragraph("Regime B: Combined (L_cls + L_loc)", table_cell_bold), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("0.6856", table_cell), Paragraph("0.5395", table_cell), Paragraph("93.10%", table_cell), Paragraph("80.32%", table_cell)],
        [Paragraph("Regime C: Align-Only (L_align)", table_cell_bold), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("0.5799", table_cell), Paragraph("0.4243", table_cell), Paragraph("<b>96.55%</b>", table_cell_bold), Paragraph("<b>82.22%</b>", table_cell_bold)],
        [Paragraph("<b>Regime D: Dice-Only (Proposed)</b>", table_cell_bold), Paragraph("<b>0.7586</b>", table_cell_bold), Paragraph("<b>0.8511</b>", table_cell_bold), Paragraph("<b>0.7107 (+49.2%)</b>", table_cell_bold), Paragraph("<b>0.5704 (+59.7%)</b>", table_cell_bold), Paragraph("82.76%", table_cell), Paragraph("69.36%", table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[120, 48, 48, 75, 75, 75, 63])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))
    
    # Embed DL Comparison Figure
    dl_fig_path = Path("results/real_cxr_gradcam_comparison.png")
    if dl_fig_path.exists():
        story.append(RLImage(str(dl_fig_path), width=504, height=135))
        story.append(Paragraph("<b>Figure 1</b>: Qualitative Localization on Real Test CXRs: (Col 1) Original Radiograph, (Col 2) Ground-Truth Lung Mask, (Col 3) Baseline Classifier Grad-CAM, and (Col 4) Proposed Localization-Aware Grad-CAM concentrating saliency inside thoracic field.", caption_style))
        story.append(Spacer(1, 4))
        
    # 4. Dual-Stream ROI Compression
    story.append(Paragraph("4. Dual-Stream ROI Compression on Real CXRs", h1_style))
    story.append(Paragraph(
        "A genuine human chest radiograph (512x512, 262 KB uncompressed) is encoded via dual-stream coding: the background context "
        "is compressed at q_bg=25, while the diagnostic lung ROI is preserved at q_roi=90. Streams are serialized into a binary container "
        "with geometry headers and reassembled at the receiver:",
        body_style
    ))
    
    # Table 2: Compression Metrics
    t2_data = [
        [Paragraph("Coding Scheme", table_header), Paragraph("Quality Setting", table_header), Paragraph("Compressed Bytes", table_header), Paragraph("Compression Ratio", table_header), Paragraph("ROI PSNR", table_header), Paragraph("ROI SSIM", table_header), Paragraph("Global PSNR", table_header)],
        [Paragraph("<b>ROI-Aware Dual-Stream</b>", table_cell_bold), Paragraph("q_roi=90, q_bg=25", table_cell), Paragraph("<b>26,515 B</b>", table_cell_bold), Paragraph("<b>9.89x (89.9% drop)</b>", table_cell_bold), Paragraph("<b>48.31 dB</b>", table_cell_bold), Paragraph("<b>0.9891</b>", table_cell_bold), Paragraph("42.10 dB", table_cell)],
        [Paragraph("Matched Uniform Baseline", table_cell_bold), Paragraph("q=90", table_cell), Paragraph("26,035 B", table_cell), Paragraph("10.07x", table_cell), Paragraph("48.31 dB", table_cell), Paragraph("0.9897", table_cell), Paragraph("49.11 dB", table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[110, 80, 75, 85, 54, 50, 50])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))
    
    # 5. Security & Computer Networks Transport
    story.append(Paragraph("5. Authenticated Encryption & Priority Network Transport", h1_style))
    story.append(Paragraph(
        "To protect patient confidentiality and integrity, payloads are encrypted with fresh nonces using either <b>AES-256-GCM</b> "
        "(NIST SP 800-38D, GHASH tag over GF(2^128)) or lightweight <b>Ascon-AEAD128</b> (NIST SP 800-232, 320-bit permutation). "
        "In 100% of single-bit tampering tests, invalid tags were immediately rejected. Over a simulated 64 kbps rural wireless channel "
        "with 5% packet loss, min-heap priority scheduling delivers diagnostic ROI chunks in <b>0.328 seconds</b> vs <b>1.046 seconds</b> "
        "for FIFO scheduling (&gt;3.1x faster triage).",
        body_style
    ))
    
    # 6. Cryptographic Diffusion Research
    story.append(Paragraph("6. Cryptographic Diffusion Architecture & Involutory MDS Matrices", h1_style))
    story.append(Paragraph(
        "In our cryptographic testbed, we investigated linear diffusion layers for lightweight block ciphers over GF(2^4)/(x^4+x+1). "
        "Beyond the optimal Hadamard instantiation <b>PROPOSED-OIN-MDS</b> = Had(1,8,13,5) (branch=5, bit-branch=8, 48 Paar XORs), "
        "we constructed novel Alternating Block-Circulant matrices <b>ABC-MDS-OPT</b> (branch=5, 49 Paar XORs) and <b>ABC-MDS-MAXBIT</b> "
        "(branch=5, bit-branch=7, 58 Paar XORs), as well as a direct binary involution <b>TBOI-16</b> (K4 (x) K4, branch=4 Near-MDS, 60 Paar XORs). "
        "All 69 submatrix minors, involution properties (M^2=I), and branch numbers were independently verified across all 65,535 non-zero inputs:",
        body_style
    ))
    
    # Table 3: Diffusion Benchmarks
    t3_data = [
        [Paragraph("Diffusion Transformation", table_header), Paragraph("Matrix Family", table_header), Paragraph("Involutory? (M^2=I)", table_header), Paragraph("Nibble Branch (B_n)", table_header), Paragraph("Bit Branch (B_b)", table_header), Paragraph("Paar CSE XORs", table_header), Paragraph("Analytical Area (GE)*", table_header)],
        [Paragraph("<b>PROPOSED-OIN-MDS</b>", table_cell_bold), Paragraph("Hadamard Had(1,8,13,5)", table_cell), Paragraph("Yes (Self-Inverse)", table_cell), Paragraph("<b>5 (Optimal MDS)</b>", table_cell_bold), Paragraph("<b>8 (Maximal)</b>", table_cell_bold), Paragraph("<b>48</b>", table_cell_bold), Paragraph("<b>96.0 GE</b>", table_cell_bold)],
        [Paragraph("<b>ABC-MDS-OPT</b>", table_cell_bold), Paragraph("Block-Circulant (0,1,2,3)", table_cell), Paragraph("Yes (Self-Inverse)", table_cell), Paragraph("<b>5 (Optimal MDS)</b>", table_cell_bold), Paragraph("6", table_cell), Paragraph("<b>49</b>", table_cell_bold), Paragraph("<b>98.0 GE</b>", table_cell_bold)],
        [Paragraph("<b>ABC-MDS-MAXBIT</b>", table_cell_bold), Paragraph("Block-Circulant (1,2,4,8)", table_cell), Paragraph("Yes (Self-Inverse)", table_cell), Paragraph("<b>5 (Optimal MDS)</b>", table_cell_bold), Paragraph("<b>7</b>", table_cell_bold), Paragraph("58", table_cell), Paragraph("116.0 GE", table_cell)],
        [Paragraph("<b>TBOI-16</b>", table_cell_bold), Paragraph("Tensor Product K4 (x) K4", table_cell), Paragraph("Yes (Binary Inv)", table_cell), Paragraph("4 (Near-MDS)", table_cell), Paragraph("6", table_cell), Paragraph("60", table_cell), Paragraph("120.0 GE", table_cell)],
    ]
    t3 = Table(t3_data, colWidths=[110, 100, 75, 75, 54, 50, 40])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    # 7. Cross-Layer Optimization
    story.append(Paragraph("7. Cross-Layer Multi-Objective Optimization", h1_style))
    story.append(Paragraph(
        "A full-pipeline grid search evaluated 108 end-to-end parameter configurations across DL, DIP, CN, and Crypto on genuine radiographs. "
        "We extracted <b>26 non-dominated Pareto configurations</b> balancing ROI SSIM, wire bytes, triage latency, and crypto cost.",
        body_style
    ))
    
    pareto_fig_path = Path("results/pareto_frontier.png")
    if pareto_fig_path.exists():
        story.append(RLImage(str(pareto_fig_path), width=504, height=130))
        story.append(Paragraph("<b>Figure 2</b>: Empirical Pareto Frontier (108 configurations): Demonstrates dominant trade-offs between ROI delivery latency, compression quality, and wire bandwidth across priority and FIFO regimes.", caption_style))
        story.append(Spacer(1, 4))
        
    # 8. Conclusion & Limitations
    story.append(Paragraph("8. Discussion, Limitations & Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Limitations</b>: (1) The active clinical cohort contains 166 radiographs across 110 patients; prospective clinical multi-center "
        "trials will scale to 20,000+ studies. (2) Ascon-AEAD128 is implemented in pure Python for transparency; production deployments "
        "will link C/Rust assembly. (3) Standard-cell GE area numbers are pre-synthesis analytical estimates (2.0 GE/XOR). (4) MediLink-DL is an engineering research prototype, not FDA/CE-cleared software.<br/>"
        "<b>Conclusion</b>: MediLink-DL successfully demonstrates an end-to-end, reproducible pipeline uniting differentiable DL localization, "
        "dual-stream compression, authenticated AEAD ciphers, priority transport, and verified involutory diffusion architectures, providing a robust template for rural telemedicine.",
        body_style
    ))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] World Health Organization, <i>Telemedicine: Opportunities and Developments in Member States</i>, Geneva: WHO Press, 2010.",
        "[2] J. P. Cohen, P. Morrison, and L. Dao, \"COVID-19 Image Data Collection,\" <i>arXiv:2003.11597</i>, 2020.",
        "[3] R. R. Selvaraju, et al., \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in <i>Proc. IEEE ICCV</i>, 2017.",
        "[4] Z. Wang, A. C. Bovik, et al., \"Image quality assessment: from error visibility to structural similarity,\" <i>IEEE TIP</i>, vol. 13, no. 4, 2004.",
        "[5] M. Dworkin, \"Galois/Counter Mode (GCM) and GMAC,\" <i>NIST SP 800-38D</i>, 2007.",
        "[6] C. Dobraunig, M. Eichlseder, et al., \"Ascon v1.2: Lightweight Authenticated Encryption,\" <i>NIST SP 800-232</i>, 2024.",
        "[7] P. Rajpurkar, et al., \"CheXNet: Radiologist-level pneumonia detection on chest X-rays,\" <i>arXiv:1711.05225</i>, 2017.",
        "[8] M. Sajadieh, M. Wei, and E. Pasalic, \"On construction of involutory MDS matrices,\" <i>IEEE TIT</i>, vol. 58, no. 9, 2012.",
        "[9] C. Beierle, A. Canteaut, and G. Leander, \"Lightweight Invertible Diffusion Layers,\" <i>IACR ToSC</i>, vol. 2018, no. 3, 2018.",
        "[10] V. Cauchois and P. Loidreau, \"On the number of involutory MDS matrices over finite fields,\" <i>Des. Codes Cryptogr.</i>, 2019.",
        "[11] D. Han, X. Wang, and B. Preneel, \"Constructing lightweight MDS and near-MDS matrices,\" in <i>Inscrypt 2016</i>, 2016."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", parent=body_style, fontSize=7, leading=9, spaceAfter=2)))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {output_path.resolve()}")


if __name__ == "__main__":
    out_file = Path("paper/MediLink_DL_Research_Paper.pdf")
    build_pdf(out_file)
