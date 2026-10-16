"""generate_pdf.py: Compile professional academic research paper PDF using ReportLab Platypus.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
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
        self.drawRightString(612 - 54, 35, page_text)
        self.drawString(54, 35, "Confidential - Academic Capstone Progress Report | MediLink-DL Team")
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
    
    # Custom academic styles
    title_style = ParagraphStyle(
        "AcademicTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8
    )
    
    author_style = ParagraphStyle(
        "AcademicAuthors",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4
    )
    
    context_style = ParagraphStyle(
        "AcademicContext",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14
    )
    
    abstract_heading = ParagraphStyle(
        "AbstractHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
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
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "AcademicH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
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
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
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
    story.append(Paragraph("Undergraduate Research Capstone Project &nbsp;|&nbsp; Digital Image Processing &amp; Computer Networks &nbsp;|&nbsp; October 2026", context_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=10))
    
    # Abstract Box
    abstract_text = (
        "<b>Abstract</b>—Rural primary healthcare clinics in resource-constrained regions often lack on-site radiologist specialists "
        "and depend on fragile, low-bandwidth communication uplinks (2G/EDGE, weak 3G, or satellite connections at 48–128 kbps). "
        "Transmitting uncompressed radiographs (10–35 MB) is impractical, while uniform lossy compression irreversibly blurs "
        "critical pathological zones. This paper introduces <b>MediLink-DL</b>, an integrated, bandwidth-adaptive, and secure "
        "framework combining: (i) an operational dual-stream Region-of-Interest (ROI) compression architecture (q_roi=90, q_bg=25); "
        "(ii) an authenticated encryption envelope utilizing AES-256-GCM with fresh 96-bit nonces; (iii) an application-layer prioritized "
        "transmission scheduler that delivers diagnostic ROI packets ahead of background context; and (iv) a weakly supervised Grad-CAM "
        "saliency module designed to guide spatial ROI selection. Validated on a calibrated 512x512 synthetic chest radiograph phantom "
        "with an indelible watermark, MediLink-DL achieved a <b>26.64x compression ratio</b> (9.84 KB payload from 262.14 KB uncompressed). "
        "Benchmarked against an exactly rate-matched uniform baseline (q=32, 9.62 KB), MediLink-DL improved diagnostic <b>ROI PSNR by +3.50 dB</b> "
        "(38.08 dB vs. 34.58 dB) and <b>ROI SSIM by +0.0853</b> (0.9293 vs. 0.8440). Under simulated rural network conditions, prioritized "
        "scheduling reduced time-to-first-ROI delivery by <b>5.1x to 7.5x</b> compared to standard FIFO transport. In a 100-trial mutation audit, "
        "AES-256-GCM rejected 100% of bit-flip tampering attacks with negligible latency (&lt;0.005 ms). Engineering trade-offs, Grad-CAM resolution "
        "limits, and differences from standardized JPEG2000 Part 2 MAXSHIFT are formally analyzed."
    )
    keywords_text = "<b>Keywords</b>: Medical Image Compression, Telemedicine, Region-of-Interest Coding, Priority Packet Transmission, Authenticated Encryption, Grad-CAM Saliency."
    
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
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Digital radiography (CR/DR) is the primary diagnostic frontline for detecting acute pulmonary consolidations, "
        "pneumothoraces, and infections [1]. In urban medical centers, gigabit networks facilitate instantaneous uncompressed "
        "DICOM transfer. In rural clinics, however, transmission is throttled to 48–128 kbps wireless uplinks with high packet loss (2–10%) "
        "and latency [1]. Transmitting raw uncompressed studies (10–35 MB) incurs multi-minute delays, while standard uniform lossy "
        "compression introduces blocking artifacts uniformly across the entire raster, degrading subtle pathology. MediLink-DL resolves "
        "this by coupling deep-learning-guided regional prioritization with authenticated encryption and priority network scheduling.",
        body_style
    ))
    
    # 2. Problem Statement & Hypotheses
    story.append(Paragraph("2. Problem Statement & Hypotheses", h1_style))
    story.append(Paragraph(
        "Standard codecs apply uniform quantization matrices across all 8x8 DCT blocks. However, diagnostic significance in a chest "
        "radiograph is highly localized: a small 30x30 lesion dictates triage, whereas peripheral tissue serves primarily as context. "
        "We formalize three central hypotheses: (<b>H1</b>) Dual-stream bit allocation will improve ROI PSNR by &ge; 3.0 dB over a rate-matched "
        "uniform baseline; (<b>H2</b>) Priority scheduling of diagnostic chunks will accelerate time-to-first-ROI delivery by &ge; 60% over "
        "FIFO transport; and (<b>H3</b>) AES-256-GCM authenticated encryption will impose &lt; 0.05 ms latency while detecting 100% of bit-flip attacks.",
        body_style
    ))
    
    # 3. System Architecture
    story.append(Paragraph("3. System Architecture", h1_style))
    story.append(Paragraph(
        "MediLink-DL integrates five sequential stages: (1) Image Validation & Synthetic Phantom Ingestion; (2) Dual-Stream ROI-Aware "
        "Differential Compression; (3) AES-256-GCM Authenticated Encryption; (4) Application-Layer Prioritized Network Transport; "
        "and (5) Remote Workstation Reassembly and Partitioned Quality Auditing. A complete vector architectural schematic is archived "
        "in <code>paper/figures/system_architecture.svg</code>.",
        body_style
    ))
    
    # Embed Comparison Figure
    comp_fig_path = Path("results/comparison.png")
    if comp_fig_path.exists():
        story.append(Spacer(1, 4))
        story.append(RLImage(str(comp_fig_path), width=504, height=125))
        story.append(Paragraph("<b>Figure 1</b>: MediLink-DL 5-Panel Visual Comparison: (1) Reference Synthetic Phantom, (2) Ground-Truth Diagnostic ROI Mask, (3) Matched Uniform Baseline (q=32, 9.6 KB), (4) ROI-Aware Dual-Stream (q_roi=90, q_bg=25, 9.8 KB), and (5) Absolute Residual Difference Map.", caption_style))
        story.append(Spacer(1, 4))
        
    # 4. Methodology
    story.append(Paragraph("4. Methodology", h1_style))
    story.append(Paragraph("<b>4.1 Dual-Stream Differential Compression</b>", h2_style))
    story.append(Paragraph(
        "Standard JPEG (ITU-T T.81) does not natively support non-uniform regional quantization tables without non-standard extensions. "
        "MediLink-DL implements an operational dual-stream architecture: the background context is compressed at lossy quality "
        "B_bg = JPEG(I, q_bg=25), while the localized ROI bounding box is compressed at diagnostic quality B_roi = JPEG(I_crop, q_roi=90). "
        "Both streams are serialized into an 'MLDL' binary payload with a 26-byte header containing geometry and stream offsets.",
        body_style
    ))
    
    story.append(Paragraph("<b>4.2 Partitioned Quality Evaluation & Fair Rate Matching</b>", h2_style))
    story.append(Paragraph(
        "To ensure rigorous evaluation, MediLink-DL performs a linear sweep over q in [1, 95] to find the exact uniform quality q* "
        "matching the dual-stream byte count within minimal delta. Mean Squared Error (MSE), PSNR, and SSIM are partitioned over "
        "diagnostic mask pixels (ROI) and contextual pixels (Background):",
        body_style
    ))
    story.append(Paragraph("PSNR_ROI = 10 * log10( 255^2 / MSE_ROI ), &nbsp;&nbsp; SSIM_ROI = mean( SSIM_Map[ Mask &gt; 0 ] )", equation_style))
    
    story.append(Paragraph("<b>4.3 Authenticated Encryption & Prioritized Transport</b>", h2_style))
    story.append(Paragraph(
        "Payloads are encrypted via AES-256-GCM using fresh 96-bit CSPRNG nonces and 128-bit Poly1305/GHASH authentication tags (28-byte overhead). "
        "Encrypted streams are framed into 1024-byte chunks with SHA-256 checksums and scheduled via min-heap priority: "
        "Priority(Header)=30 &gt; Priority(ROI)=20 &gt; Priority(Background)=5. ARQ handles packet loss recovery.",
        body_style
    ))
    
    # 5. Experimental Results
    story.append(Paragraph("5. Experimental Results & Verification", h1_style))
    story.append(Paragraph(
        "All reported values were generated through automated runs and independently verified against source CSV files in <code>results/</code>. "
        "The test subject is a standardized 512x512 8-bit grayscale synthetic CXR phantom (262,144 bytes uncompressed) with an indelible disclaimer watermark.",
        body_style
    ))
    
    # Table 1: Baseline Metrics
    story.append(Paragraph("<b>Table 1: Baseline Rate-Distortion Metrics (results/metrics.csv)</b>", caption_style))
    t1_data = [
        [Paragraph("Metric", table_header), Paragraph("ROI-Aware Coding (q_roi=90, q_bg=25)", table_header), Paragraph("Uniform Baseline (q=32)", table_header), Paragraph("Empirical Delta", table_header)],
        [Paragraph("Compressed Payload", table_cell_bold), Paragraph("9,842 bytes (9.61 KB)", table_cell), Paragraph("9,622 bytes (9.40 KB)", table_cell), Paragraph("+220 bytes (+2.2%)", table_cell)],
        [Paragraph("Compression Ratio", table_cell_bold), Paragraph("26.64x (96.25% reduction)", table_cell), Paragraph("27.24x (96.33% reduction)", table_cell), Paragraph("Rate-Matched", table_cell)],
        [Paragraph("Global PSNR", table_cell_bold), Paragraph("32.51 dB", table_cell), Paragraph("33.04 dB", table_cell), Paragraph("-0.53 dB", table_cell)],
        [Paragraph("Global SSIM", table_cell_bold), Paragraph("0.8049", table_cell), Paragraph("0.8133", table_cell), Paragraph("-0.0084", table_cell)],
        [Paragraph("<b>ROI PSNR (Pathology)</b>", table_cell_bold), Paragraph("<b>38.08 dB</b>", table_cell_bold), Paragraph("34.58 dB", table_cell), Paragraph("<b>+3.50 dB Gain</b>", table_cell_bold)],
        [Paragraph("<b>ROI SSIM (Pathology)</b>", table_cell_bold), Paragraph("<b>0.9293</b>", table_cell_bold), Paragraph("0.8440", table_cell), Paragraph("<b>+0.0853 Gain</b>", table_cell_bold)],
        [Paragraph("Background PSNR", table_cell_bold), Paragraph("32.47 dB", table_cell), Paragraph("33.02 dB", table_cell), Paragraph("-0.55 dB", table_cell)],
        [Paragraph("Background SSIM", table_cell_bold), Paragraph("0.8035", table_cell), Paragraph("0.8131", table_cell), Paragraph("-0.0096", table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[120, 140, 130, 114])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))
    
    # Embed Rate-Distortion & Transmission Plots side by side or stacked
    rd_plot_path = Path("results/experiment_compression.png")
    tx_plot_path = Path("results/experiment_transmission.png")
    
    if rd_plot_path.exists():
        story.append(RLImage(str(rd_plot_path), width=504, height=190))
        story.append(Paragraph("<b>Figure 2</b>: Empirical Rate-Distortion Curves Across Background Qualities (q_bg in [15, 75]): Diagnostic ROI PSNR and SSIM are preserved at a constant high plateau (38.08 dB, ~0.93 SSIM) while the matched uniform baseline degrades severely at lower bitrates.", caption_style))
        story.append(Spacer(1, 6))
        
    # Table 2: Transmission Simulation
    story.append(Paragraph("<b>Table 2: Network Transport Across Simulated Rural Channels (results/experiment_transmission.csv)</b>", caption_style))
    t2_data = [
        [Paragraph("Channel Profile", table_header), Paragraph("Bandwidth / RTT / Loss", table_header), Paragraph("Time to ROI (Priority)", table_header), Paragraph("Time to ROI (FIFO)", table_header), Paragraph("Total Transfer", table_header), Paragraph("Acceleration", table_header)],
        [Paragraph("Rural 2G/EDGE", table_cell_bold), Paragraph("48 kbps, 200 ms, 2%", table_cell), Paragraph("<b>0.592 s</b>", table_cell_bold), Paragraph("3.592 s", table_cell), Paragraph("3.916 s", table_cell), Paragraph("<b>6.07x</b>", table_cell_bold)],
        [Paragraph("Rural 3G (Weak)", table_cell_bold), Paragraph("128 kbps, 150 ms, 5%", table_cell), Paragraph("<b>0.372 s</b>", table_cell_bold), Paragraph("2.097 s", table_cell), Paragraph("2.294 s", table_cell), Paragraph("<b>5.63x</b>", table_cell_bold)],
        [Paragraph("Satellite VSAT", table_cell_bold), Paragraph("256 kbps, 500 ms, 2%", table_cell), Paragraph("<b>1.036 s</b>", table_cell_bold), Paragraph("5.299 s", table_cell), Paragraph("5.822 s", table_cell), Paragraph("<b>5.11x</b>", table_cell_bold)],
        [Paragraph("Lossy Wireless", table_cell_bold), Paragraph("64 kbps, 120 ms, 10%", table_cell), Paragraph("<b>0.384 s</b>", table_cell_bold), Paragraph("2.887 s", table_cell), Paragraph("3.100 s", table_cell), Paragraph("<b>7.51x</b>", table_cell_bold)],
    ]
    t2 = Table(t2_data, colWidths=[95, 115, 80, 75, 75, 64])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))
    
    if tx_plot_path.exists():
        story.append(RLImage(str(tx_plot_path), width=504, height=190))
        story.append(Paragraph("<b>Figure 3</b>: Transmission Delivery Acceleration: Prioritized Diagnostic ROI arrival time (green) compared against Total Full-Frame Transfer Time (blue) across rural channel profiles.", caption_style))
        story.append(Spacer(1, 6))

    # 6. Discussion & Limitations
    story.append(Paragraph("6. Discussion & Limitations", h1_style))
    story.append(Paragraph(
        "<b>6.1 Rate-Distortion & Triage Dynamics</b>: The empirical results validate both H1 and H2. In Table 1, shifting bits from contextual anatomy "
        "into the 1.30% diagnostic area yielded a +3.50 dB ROI PSNR gain with only a -0.53 dB background penalty. In Table 2, prioritized transport delivered "
        "diagnostic data in 0.38–0.59 seconds, unlocking immediate specialist triage while the full image completes loading over the next 2.2–3.9 seconds. "
        "In Table 3 and security audits, AES-256-GCM proved tamper-evident (100% rejection across 100 trials) with zero latency impact (<0.005 ms).",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.2 Academic & Engineering Limitations</b>: (1) <i>Synthetic Data</i>: All tests utilized an anatomically stylized phantom. Clinical validation on DICOM datasets (e.g. NIH ChestX-ray14) remains future work. "
        "(2) <i>JPEG Approximation vs. JPEG2000</i>: Our prototype uses dual-stream compositing, not standardized JPEG2000 Part 2 MAXSHIFT DWT scaling. "
        "(3) <i>Grad-CAM Saliency</i>: The Grad-CAM module operates on coarse convolutional layers, providing heatmaps rather than pixel-exact segmentation boundaries. "
        "(4) <i>Transport Simulation</i>: Discrete-event simulation models queuing and drop rates without reproducing physical RF fading.",
        body_style
    ))

    # 7. Conclusion & References
    story.append(Paragraph("7. Conclusion & Future Work", h1_style))
    story.append(Paragraph(
        "MediLink-DL demonstrates an effective, CPU-compatible research prototype for rural medical radiograph transmission. "
        "By uniting dual-stream ROI coding, prioritized packet scheduling, and authenticated encryption, the system delivers a 26.6x compression ratio, "
        "+3.50 dB ROI PSNR gain, and 5.1x–7.5x triage acceleration. Future work will integrate clinical PACS datasets, test over physical software-defined "
        "radios, and incorporate radiologist observer studies.",
        body_style
    ))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] World Health Organization, <i>Telemedicine: Opportunities and Developments in Member States</i>, Global Observatory for eHealth Series, vol. 2, Geneva: WHO Press, 2010.",
        "[2] C. Christopoulos, A. Skodras, and T. Ebrahimi, \"The JPEG2000 still image coding system: an overview,\" <i>IEEE Trans. Consum. Electron.</i>, vol. 46, no. 4, pp. 1103–1127, 2000.",
        "[3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, \"Image quality assessment: from error visibility to structural similarity,\" <i>IEEE Trans. Image Process.</i>, vol. 13, no. 4, pp. 600–612, 2004.",
        "[4] X. Wang, et al., \"ChestX-ray8: Hospital-scale chest X-ray database and benchmarks,\" in <i>Proc. IEEE CVPR</i>, 2017, pp. 2097–2106.",
        "[5] J. Irvin, et al., \"CheXpert: A large chest radiograph dataset with uncertainty labels,\" in <i>Proc. AAAI Conf. Hum. Comput. Crowdsourcing</i>, vol. 33, no. 1, pp. 590–597, 2019.",
        "[6] P. Rajpurkar, et al., \"CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning,\" <i>arXiv:1711.05225</i>, 2017.",
        "[7] R. R. Selvaraju, et al., \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in <i>Proc. IEEE ICCV</i>, 2017, pp. 618–626.",
        "[8] M. Dworkin, \"Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC,\" <i>NIST Special Publication 800-38D</i>, 2007."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", parent=body_style, fontSize=7.5, leading=10, spaceAfter=3)))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {output_path.resolve()}")


if __name__ == "__main__":
    out_file = Path("paper/MediLink_DL_Research_Paper.pdf")
    build_pdf(out_file)
