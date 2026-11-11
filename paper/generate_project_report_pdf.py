"""generate_project_report_pdf.py: Compile comprehensive BTech project report PDF.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Generates a multi-page, publication-grade academic report PDF for IIIT Raichur Capstone.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable, PageBreak, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class ReportNumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count and professional running headers/footers."""
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
        
        # Don't draw header on cover page
        if self._pageNumber > 1:
            self.drawString(54, 750, "MediLink-DL: B.Tech Capstone Project Report — IIIT Raichur")
            self.drawRightString(612 - 54, 750, "Department of CSE (2026)")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)
            
        # Footer on all pages except cover
        if self._pageNumber > 1:
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 36, page_text)
            self.drawString(54, 36, "IIIT RAICHUR — B.TECH CAPSTONE THESIS — CONFIDENTIAL & ACADEMIC")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 46, 612 - 54, 46)
        self.restoreState()


def build_report_pdf(output_path: Path):
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
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=20
    )
    
    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )
    
    h1_style = ParagraphStyle(
        "ReportH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        "ReportCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=6,
        spaceAfter=8
    )
    
    caption_style = ParagraphStyle(
        "ReportCaption",
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

    # ==================== COVER PAGE ====================
    story.append(Spacer(1, 40))
    story.append(Paragraph("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY RAICHUR", ParagraphStyle("Inst", parent=meta_style, fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#0f172a"))))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", ParagraphStyle("Dept", parent=meta_style, fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor("#475569"))))
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#1e3a8a"), spaceAfter=25))
    
    story.append(Paragraph("MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine", title_style))
    story.append(Paragraph("B.Tech Capstone Project Final Report", subtitle_style))
    
    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Submitted by:</b>", meta_style))
    story.append(Paragraph("<b>Mayank Soni</b> (Roll No: CS24B1031) — Architecture, Crypto & Optimization", meta_style))
    story.append(Paragraph("<b>Nandan</b> (Roll No: CS24B1034) — Deep Learning Localization & Saliency Loss", meta_style))
    story.append(Paragraph("<b>Sumedha</b> (Roll No: AD24B1061) — DIP Compression & Priority Transport", meta_style))
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Under the Guidance of Faculty Advisors:</b>", meta_style))
    story.append(Paragraph("<b>Dr. Kaushiki Roy</b> (Assistant Professor, CSE) — Digital Image Processing (DIP)", meta_style))
    story.append(Paragraph("<b>Dr. Mehbub Alam</b> (Assistant Professor, CSE) — Computer Networks (CN)", meta_style))
    story.append(Paragraph("<b>Arnav Ghosh</b> (Assistant Professor, CSE) — Multi-Objective Optimization", meta_style))
    
    story.append(Spacer(1, 40))
    story.append(Paragraph("November 2026 &nbsp;|&nbsp; Academic Year 2026–2027", meta_style))
    story.append(Paragraph("Raichur, Karnataka, India", meta_style))
    story.append(PageBreak())

    # ==================== EXECUTIVE SUMMARY ====================
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(
        "Digital chest radiography is often the single frontline imaging modality available at primary healthcare centers (PHCs) in rural India. "
        "However, resident radiologists are virtually absent, requiring radiograph transmission to tertiary centers 100 to 300 kilometers away. "
        "Over rural 2G/EDGE cellular connections throttled to 32–64 kbps, transmitting raw uncompressed radiographs (10–35 MB) requires between "
        "20 and 75 minutes per patient, making acute triage impossible. Conversely, uniform lossy compression blurs subtle lung opacities, and "
        "consumer messaging apps (e.g., WhatsApp) violate statutory privacy mandates (HIPAA/DPDP) while stripping clinical DICOM metadata.",
        body_style
    ))
    story.append(Paragraph(
        "<b>MediLink-DL</b> resolves this clinical trilemma through an integrated cross-layer framework. Key achievements include:",
        body_style
    ))
    exec_points = [
        "<b>1. Clinical Data & Zero-Leakage Protocol</b>: Curation of 166 real clinical chest radiographs from 110 patients (Cohen et al. 2020 cohort) with strict patient-level non-overlapping partitioning (P_train &cap; P_test = &empty;), eliminating the data leakage that artificially inflates accuracy in published models.",
        "<b>2. Differentiable Grad-CAM & Uncertainty Loss</b>: Modified single-channel <b>MedicalResNet18</b> coupled with autograd-preserving Differentiable Grad-CAM (<code>create_graph=True</code>) and Kendall &amp; Gal (CVPR 2018) homoscedastic uncertainty optimization. Champion Regime B achieves <b>93.10% Pointing Game Hit Rate</b>, <b>80.32% Saliency Energy in ROI</b>, and <b>100% Sensitivity (Recall)</b>.",
        "<b>3. Dual-Stream ROI Compression</b>: Encoding diagnostic lung patches at q_roi=90 and context at q_bg=25 packed inside a 28-byte binary container ('MLDL'), reducing data by <b>89.89% (9.89x ratio)</b> to <b>26,515 bytes</b>. Sustains an <b>ROI PSNR of 48.31 dB</b> and <b>ROI SSIM of 0.9891</b>—a <b>+10.13 dB gain</b> over rate-matched uniform baseline JPEG.",
        "<b>4. Authenticated Encryption (AEAD)</b>: AES-256-GCM (2.2 &mu;s, 465 MB/s) and NIST SP 800-232 Ascon-AEAD128 with Associated Data (AAD) metadata binding and 100% bit-flip tamper rejection.",
        "<b>5. Priority Transport Scheduling</b>: Application-layer Min-Heap scheduler delivering diagnostic ROIs across 64 kbps rural channels in <b>0.391 seconds</b> (<b>6.21x faster than FIFO</b>).",
        "<b>6. Multi-Objective Pareto Frontier</b>: Identification of 26 non-dominated Pareto configurations across 108 end-to-end evaluated operating sets.",
        "<b>7. Production Verification</b>: 147 automated pytest suites passing and live web telemedicine deployment at https://medilink-dl.onrender.com."
    ]
    for pt in exec_points:
        story.append(Paragraph(f"&bull; {pt}", body_style))
    story.append(Spacer(1, 10))

    # ==================== 1. INTRODUCTION ====================
    story.append(Paragraph("1. Introduction and Real-World Motivation", h1_style))
    story.append(Paragraph(
        "Digital radiography is the primary diagnostic frontline for detecting acute pulmonary consolidations, pneumothoraces, tuberculosis, "
        "and viral pneumonias. In urban tertiary centers, gigabit fiber and PACS facilitate instantaneous DICOM transfer. In rural clinics across "
        "states like Chhattisgarh, Rajasthan, and Jharkhand, however, transmission is throttled to 48–128 kbps wireless uplinks with high packet "
        "drop rates (2–10%) and multi-hop latencies. When a patient arrives with acute respiratory distress, sending uncompressed images over these "
        "links takes over an hour. Uniform lossy compression destroys fine pathological margins, while informal smartphone photographs sent via "
        "consumer chat apps lack cryptographic provenance and strip vital DICOM tags.",
        body_style
    ))

    # Embed System Architecture
    arch_fig = Path("paper/figures/system_architecture.png")
    if arch_fig.exists():
        story.append(RLImage(str(arch_fig), width=504, height=180))
        story.append(Paragraph("<b>Figure 1</b>: MediLink-DL End-to-End Five-Stage Cross-Layer Pipeline Architecture.", caption_style))
        story.append(Spacer(1, 6))

    # ==================== 2. DEEP LEARNING MODULE ====================
    story.append(Paragraph("2. Deep Learning Diagnostic Localization", h1_style))
    story.append(Paragraph(
        "We modified ResNet-18 into single-channel <code>MedicalResNet18</code> and formulated <code>DifferentiableGradCAM</code> "
        "(<code>create_graph=True</code>) to allow second-order gradients to train the model directly on spatial localization. Multi-task loss "
        "balancing between classification and continuous Dice/alignment penalties is automated using Kendall &amp; Gal homoscedastic uncertainty optimization:",
        body_style
    ))
    story.append(Paragraph(
        "<i>L_total(w, s_1, s_2) = 0.5 &middot; exp(-s_1) &middot; L_cls(w) + 0.5 &middot; exp(-s_2) &middot; L_loc(w) + 0.5 &middot; s_1 + 0.5 &middot; s_2</i>",
        ParagraphStyle("Eq2", parent=body_style, fontName="Courier-Oblique", fontSize=8, alignment=TA_CENTER)
    ))
    
    # Table 1: Training Regimes
    story.append(Paragraph("<b>Table 1: Four-Regime Deep Learning Localization Comparison (Held-Out Test Cohort)</b>", caption_style))
    t1_data = [
        [Paragraph("Regime / Mode", table_header), Paragraph("Accuracy", table_header), Paragraph("F1-Score", table_header), Paragraph("Recall", table_header), Paragraph("ROI Dice", table_header), Paragraph("Pointing Game", table_header), Paragraph("Energy in ROI", table_header)],
        [Paragraph("Regime A: Cls-Only", table_cell), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("1.0000", table_cell), Paragraph("0.4764", table_cell), Paragraph("65.52%", table_cell), Paragraph("49.21%", table_cell)],
        [Paragraph("<b>Regime B: Combined (Proposed)</b>", table_cell_bold), Paragraph("<b>0.6897</b>", table_cell_bold), Paragraph("<b>0.8163</b>", table_cell_bold), Paragraph("<b>1.0000</b>", table_cell_bold), Paragraph("<b>0.6856</b>", table_cell_bold), Paragraph("<b>93.10%</b>", table_cell_bold), Paragraph("<b>80.32%</b>", table_cell_bold)],
        [Paragraph("Regime C: Align-Only", table_cell), Paragraph("0.6897", table_cell), Paragraph("0.8163", table_cell), Paragraph("1.0000", table_cell), Paragraph("0.5799", table_cell), Paragraph("96.55%", table_cell), Paragraph("82.22%", table_cell)],
        [Paragraph("Regime D: Dice-Only", table_cell), Paragraph("0.7586", table_cell), Paragraph("0.8511", table_cell), Paragraph("1.0000", table_cell), Paragraph("0.7107", table_cell), Paragraph("82.76%", table_cell), Paragraph("69.36%", table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[120, 55, 55, 55, 65, 80, 74])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    dl_comp_fig = Path("results/real_cxr_gradcam_comparison.png")
    if dl_comp_fig.exists():
        story.append(RLImage(str(dl_comp_fig), width=504, height=125))
        story.append(Paragraph("<b>Figure 2</b>: Qualitative Saliency Comparison on Held-Out Real Test Radiographs.", caption_style))
        story.append(Spacer(1, 6))

    # ==================== 3. DIP COMPRESSION MODULE ====================
    story.append(Paragraph("3. Digital Image Processing: Dual-Stream Compression", h1_style))
    story.append(Paragraph(
        "Under the supervision of Dr. Kaushiki Roy, we designed a dual-stream differential compression engine. Context is encoded at "
        "q_bg=25, while the diagnostic lung lesion is preserved at q_roi=90. Streams are packed into a 28-byte self-describing binary container "
        "('MLDL') and composited at the receiver using distance-transform alpha blending across a 4-pixel border. Partitioned PSNR and SSIM "
        "metrics isolate the diagnostic lesion from non-pathological background pixels:",
        body_style
    ))
    
    # Table 2: Compression Metrics
    story.append(Paragraph("<b>Table 2: Rate-Distortion Performance on Genuine Chest Radiographs (262,144 Bytes Raw)</b>", caption_style))
    t2_data = [
        [Paragraph("Compression Strategy", table_header), Paragraph("Quality Setting", table_header), Paragraph("Transmitted Bytes", table_header), Paragraph("Compression Ratio", table_header), Paragraph("ROI PSNR", table_header), Paragraph("ROI SSIM", table_header), Paragraph("Global PSNR", table_header)],
        [Paragraph("<b>Dual-Stream ROI-Aware (Ours)</b>", table_cell_bold), Paragraph("q_roi=90, q_bg=25", table_cell), Paragraph("<b>26,515 B</b>", table_cell_bold), Paragraph("<b>9.89x (89.89%)</b>", table_cell_bold), Paragraph("<b>48.31 dB</b>", table_cell_bold), Paragraph("<b>0.9891</b>", table_cell_bold), Paragraph("42.10 dB", table_cell)],
        [Paragraph("Rate-Matched Uniform JPEG", table_cell), Paragraph("q=32 (Matched Bytes)", table_cell), Paragraph("26,035 B", table_cell), Paragraph("10.07x", table_cell), Paragraph("38.18 dB", table_cell), Paragraph("0.9312", table_cell), Paragraph("39.45 dB", table_cell)],
        [Paragraph("High-Quality Uniform JPEG", table_cell), Paragraph("q=90", table_cell), Paragraph("98,412 B", table_cell), Paragraph("2.66x", table_cell), Paragraph("48.31 dB", table_cell), Paragraph("0.9897", table_cell), Paragraph("49.11 dB", table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[120, 80, 75, 85, 52, 46, 46])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    # ==================== 4. CRYPTOGRAPHIC SECURITY ====================
    story.append(Paragraph("4. Cryptographic Security & MLMC-128 Research Lab", h1_style))
    story.append(Paragraph(
        "MediLink-DL enforces Authenticated Encryption with Associated Data (AEAD) using NIST SP 800-38D AES-256-GCM and NIST SP 800-232 "
        "Ascon-AEAD128, strictly guaranteeing Zero-Plaintext Release and 100% bit-flip tamper rejection. In our research lab, we designed "
        "<b>MLMC-128</b>, an educational 128-bit block cipher utilizing PRESENT S-boxes and an optimal involutory Hadamard diffusion matrix "
        "<code>PROPOSED-OIN-MDS</code> = Had(1,8,13,5) with branch number B=5, requiring only 48 Paar CSE XORs (~96 Gate Equivalents) and achieving a 49.92% SAC.",
        body_style
    ))
    
    # Table 3: Crypto Benchmark
    story.append(Paragraph("<b>Table 3: Cryptographic Throughput, Latency and Tamper Rejection Benchmark</b>", caption_style))
    t3_data = [
        [Paragraph("Cipher / Mode", table_header), Paragraph("Standard / Architecture", table_header), Paragraph("Enc Latency (1KB)", table_header), Paragraph("Dec Latency (1KB)", table_header), Paragraph("Throughput", table_header), Paragraph("Tamper Rejection", table_header)],
        [Paragraph("<b>AES-256-GCM</b>", table_cell_bold), Paragraph("NIST SP 800-38D (ARMv8)", table_cell), Paragraph("<b>2.2 &mu;s</b>", table_cell_bold), Paragraph("<b>2.4 &mu;s</b>", table_cell_bold), Paragraph("<b>465.4 MB/s</b>", table_cell_bold), Paragraph("<b>100.0%</b>", table_cell_bold)],
        [Paragraph("<b>Ascon-AEAD128</b>", table_cell_bold), Paragraph("NIST SP 800-232 (IoT)", table_cell), Paragraph("412.0 &mu;s", table_cell), Paragraph("418.0 &mu;s", table_cell), Paragraph("2.48 MB/s", table_cell), Paragraph("<b>100.0%</b>", table_cell_bold)],
        [Paragraph("<b>MLMC-128 (Raw Block)</b>", table_cell), Paragraph("Research Prototype (Python)", table_cell), Paragraph("2,140.0 &mu;s", table_cell), Paragraph("2,190.0 &mu;s", table_cell), Paragraph("0.47 MB/s", table_cell), Paragraph("N/A", table_cell)],
        [Paragraph("<b>MLMC-128 (AEAD Mode)</b>", table_cell), Paragraph("Research Candidate (CTR+GHASH)", table_cell), Paragraph("7,889.0 &mu;s", table_cell), Paragraph("7,920.0 &mu;s", table_cell), Paragraph("0.13 MB/s", table_cell), Paragraph("<b>100.0%</b>", table_cell_bold)],
    ]
    t3 = Table(t3_data, colWidths=[110, 115, 70, 70, 65, 74])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6))

    # ==================== 5. NETWORKS & TRANSPORT ====================
    story.append(Paragraph("5. Computer Networks: Prioritized Transport Scheduling", h1_style))
    story.append(Paragraph(
        "Under the supervision of Dr. Mehbub Alam, packets are chunked with 'MLPK' framing and prioritized in a Min-Heap queue "
        "(Header: 30 > ROI: 20 > Background: 5). Discrete-event simulation across rural wireless channels proves a 5.1x to 6.2x triage acceleration:",
        body_style
    ))
    
    # Table 4: Networks Latency
    story.append(Paragraph("<b>Table 4: Wireless Channel Profiles and Triage Acceleration Performance</b>", caption_style))
    t4_data = [
        [Paragraph("Channel Profile", table_header), Paragraph("Bandwidth / RTT / Loss", table_header), Paragraph("Priority Time-to-ROI", table_header), Paragraph("Standard FIFO Time", table_header), Paragraph("Triage Acceleration", table_header)],
        [Paragraph("<b>Rural 2G / EDGE</b>", table_cell_bold), Paragraph("48 kbps, 200 ms, 2.0% drop", table_cell), Paragraph("<b>0.592 s</b>", table_cell_bold), Paragraph("3.592 s", table_cell), Paragraph("<b>6.07x Faster</b>", table_cell_bold)],
        [Paragraph("<b>Rural GPRS</b>", table_cell_bold), Paragraph("64 kbps, 120 ms, 5.0% drop", table_cell), Paragraph("<b>0.391 s</b>", table_cell_bold), Paragraph("2.430 s", table_cell), Paragraph("<b>6.21x Faster</b>", table_cell_bold)],
        [Paragraph("<b>Rural 3G (Weak)</b>", table_cell_bold), Paragraph("128 kbps, 150 ms, 5.0% drop", table_cell), Paragraph("<b>0.372 s</b>", table_cell_bold), Paragraph("2.097 s", table_cell), Paragraph("<b>5.63x Faster</b>", table_cell_bold)],
        [Paragraph("<b>Satellite VSAT</b>", table_cell_bold), Paragraph("256 kbps, 500 ms, 2.0% drop", table_cell), Paragraph("<b>1.036 s</b>", table_cell_bold), Paragraph("5.299 s", table_cell), Paragraph("<b>5.11x Faster</b>", table_cell_bold)],
    ]
    t4 = Table(t4_data, colWidths=[105, 125, 85, 85, 104])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t4)
    story.append(Spacer(1, 6))

    # ==================== 6. PARETO FRONTIER ====================
    story.append(Paragraph("6. Multi-Objective Optimization & Pareto Frontier", h1_style))
    story.append(Paragraph(
        "Supervised by Arnav Ghosh, a grid search across 108 end-to-end parameter sets identified 26 non-dominated Pareto configurations, "
        "establishing the optimal operating envelope across ROI SSIM, wire bytes, triage delay, and cryptographic overhead.",
        body_style
    ))
    pareto_fig = Path("results/pareto_frontier.png")
    if pareto_fig.exists():
        story.append(RLImage(str(pareto_fig), width=504, height=130))
        story.append(Paragraph("<b>Figure 3</b>: Empirical Pareto Frontier across 108 Evaluated Pipeline Configurations.", caption_style))
        story.append(Spacer(1, 6))

    # ==================== 7. CONCLUSIONS & LIMITATIONS ====================
    story.append(Paragraph("7. Honest Limitations & Academic Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Limitations</b>: (1) Current evaluation is based on 166 clinical radiographs from 110 patients; regulatory medical device clearance "
        "requires multi-center validation on 20,000+ studies. (2) The current pipeline targets planar 2D radiographs rather than 3D volumetric CT/MRI scans. "
        "(3) Formal Multi-Reader Multi-Case (MRMC) ROC observer trials with certified radiologists are needed to corroborate mathematical PSNR gains. "
        "(4) Key management currently uses in-memory ephemeral keys, requiring PKI/HPKE integration for hospital deployment.<br/>"
        "<b>Conclusion</b>: MediLink-DL demonstrates that intelligent, content-aware cross-layer engineering can overcome severe physical bandwidth constraints "
        "in rural telemedicine, delivering high-fidelity diagnostic data in sub-second times with rigorous cryptographic protection.",
        body_style
    ))
    
    doc.build(story, canvasmaker=ReportNumberedCanvas)
    print(f"Successfully generated Project Report PDF: {output_path.resolve()}")


if __name__ == "__main__":
    out_report = Path("paper/MediLink_DL_Project_Report.pdf")
    build_report_pdf(out_report)
