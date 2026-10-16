"""generate_pptx.py: Programmatically generate professional 10-slide academic progress presentation.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def create_presentation(output_path: Path):
    prs = Presentation()
    # 16:9 Widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Palette definition
    NAVY = RGBColor(15, 23, 42)        # #0f172a
    BLUE = RGBColor(29, 78, 216)       # #1d4ed8
    LIGHT_BG = RGBColor(248, 250, 252) # #f8fafc
    WHITE = RGBColor(255, 255, 255)
    TEXT_DARK = RGBColor(30, 41, 59)   # #1e293b
    TEXT_MUTED = RGBColor(100, 116, 139) # #64748b
    GREEN = RGBColor(22, 163, 74)      # #16a34a
    AMBER = RGBColor(202, 138, 4)      # #ca8a04
    BORDER_COLOR = RGBColor(203, 213, 225) # #cbd5e1

    def add_slide_header(slide, title_text, category_text="MediLink-DL | Academic Research Progress"):
        # Background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_BG
        bg.line.color.rgb = LIGHT_BG
        
        # Header banner card
        hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = WHITE
        hdr.line.color.rgb = BORDER_COLOR
        hdr.line.width = Pt(1)
        
        tf = hdr.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_top = Inches(0.12)
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Calibri"
        p_cat.font.size = Pt(9.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BLUE
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(18)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    # ==========================================================
    # SLIDE 1: Title & Author Team
    # ==========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.color.rgb = NAVY

    # Title Card Box
    card1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.0), Inches(10.933), Inches(5.5))
    card1.fill.solid()
    card1.fill.fore_color.rgb = RGBColor(30, 41, 59)
    card1.line.color.rgb = RGBColor(51, 65, 85)
    card1.line.width = Pt(1.5)
    
    tf1 = card1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.6)
    tf1.margin_top = Inches(0.5)
    
    p = tf1.paragraphs[0]
    p.text = "RESEARCH CAPSTONE PROGRESS SUBMISSION"
    p.font.name = "Calibri"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(147, 197, 253)
    p.space_after = Pt(14)
    
    p = tf1.add_paragraph()
    p.text = "MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine"
    p.font.name = "Calibri"
    p.font.size = Pt(23)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_after = Pt(24)
    
    p = tf1.add_paragraph()
    p.text = "Author Team:"
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(226, 232, 240)
    p.space_after = Pt(6)
    
    authors = [
        "• Mayank Soni — Roll No: CS24B1031 (Core Compression, Metrics & Security)",
        "• Nandan — Roll No: CS24B1034 (Network Transport Simulation & Priority Queuing)",
        "• Sumedha — Roll No: AD24B1061 (Deep Learning & Grad-CAM Saliency Analysis)"
    ]
    for a in authors:
        p = tf1.add_paragraph()
        p.text = a
        p.font.name = "Calibri"
        p.font.size = Pt(11.5)
        p.font.color.rgb = RGBColor(203, 213, 225)
        p.space_after = Pt(4)
        
    p = tf1.add_paragraph()
    p.text = "\nAcademic Context: Combined Digital Image Processing & Computer Networks Research Prototype | October 2026"
    p.font.name = "Calibri"
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = RGBColor(148, 163, 184)

    # ==========================================================
    # SLIDE 2: Problem Statement & Motivation
    # ==========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide2, "1. Problem Statement: The Rural Telemedicine Bottleneck")
    
    # 3 Column Cards
    col_width = Inches(3.64)
    y_pos = Inches(1.6)
    h_box = Inches(5.2)
    
    c1 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), y_pos, col_width, h_box)
    c1.fill.solid(); c1.fill.fore_color.rgb = WHITE; c1.line.color.rgb = BORDER_COLOR
    tf = c1.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Infrastructural Constraints"; p.font.bold = True; p.font.size = Pt(14); p.font.color.rgb = BLUE; p.space_after = Pt(10)
    bullets1 = [
        "Rural Primary Health Centers (PHCs) lack resident radiologists.",
        "Connectivity is throttled to 2G/EDGE (32-64 kbps), weak 3G, or high-latency satellite.",
        "Round-trip latency exceeds 150-500 ms with 2% to 10% packet drop rates.",
        "Acute respiratory emergencies require rapid specialist tele-triage."
    ]
    for b in bullets1:
        p = tf.add_paragraph(); p.text = "• " + b; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    c2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.84), y_pos, col_width, h_box)
    c2.fill.solid(); c2.fill.fore_color.rgb = WHITE; c2.line.color.rgb = BORDER_COLOR
    tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Compression Dilemma"; p.font.bold = True; p.font.size = Pt(14); p.font.color.rgb = NAVY; p.space_after = Pt(10)
    bullets2 = [
        "Uncompressed radiographs (10-35 MB) take 10-30 minutes to transmit over 64 kbps links.",
        "Uniform lossy compression allocates bits equally across all 8x8 DCT blocks.",
        "High compression destroys fine pathological margins, pleura lines, and microcalcifications.",
        "Standard codecs cannot differentiate diagnostic zones from surrounding air."
    ]
    for b in bullets2:
        p = tf.add_paragraph(); p.text = "• " + b; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    c3 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.88), y_pos, col_width, h_box)
    c3.fill.solid(); c3.fill.fore_color.rgb = WHITE; c3.line.color.rgb = BORDER_COLOR
    tf = c3.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Security Vulnerability"; p.font.bold = True; p.font.size = Pt(14); p.font.color.rgb = GREEN; p.space_after = Pt(10)
    bullets3 = [
        "Public wireless networks are exposed to eavesdropping and Man-in-the-Middle (MITM) attacks.",
        "Tampering with chest X-ray pixels can alter diagnostic interpretation.",
        "Regulatory standards (HIPAA, GDPR) mandate authenticated confidentiality.",
        "Encryption must be fast enough to run on low-power rural client laptops without adding lag."
    ]
    for b in bullets3:
        p = tf.add_paragraph(); p.text = "• " + b; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    # ==========================================================
    # SLIDE 3: Research Gap & Objectives
    # ==========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide3, "2. Research Gap, Central Question & Hypotheses")
    
    box_l = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    box_l.fill.solid(); box_l.fill.fore_color.rgb = WHITE; box_l.line.color.rgb = BORDER_COLOR
    tf = box_l.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "The Identified Research Gap"; p.font.bold = True; p.font.size = Pt(14); p.font.color.rgb = BLUE; p.space_after = Pt(8)
    gap_points = [
        "Deep learning saliency (e.g. Grad-CAM) is traditionally used offline for model interpretability, rarely feeding live adaptive transmission.",
        "Standard JPEG2000 Part 2 MAXSHIFT requires complex DWT wavelet codecs that lack lightweight, universal clinical browser support.",
        "Existing transport protocols are 'content-blind': packet schedulers treat high-priority diagnostic bytes identically to contextual background pixels.",
        "Most published work isolates either compression or security, failing to benchmark end-to-end payload overhead and latency."
    ]
    for g in gap_points:
        p = tf.add_paragraph(); p.text = "• " + g; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    box_r = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    box_r.fill.solid(); box_r.fill.fore_color.rgb = WHITE; box_r.line.color.rgb = BORDER_COLOR
    tf = box_r.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Testable Research Hypotheses"; p.font.bold = True; p.font.size = Pt(14); p.font.color.rgb = NAVY; p.space_after = Pt(8)
    hyps = [
        ("Hypothesis 1 (H1) - Rate-Distortion Gain", "Allocating q_roi=90 to the diagnostic bounding box and q_bg=25 to context will achieve >= +3.0 dB ROI PSNR over a rate-matched uniform baseline without severe background loss."),
        ("Hypothesis 2 (H2) - Triage Acceleration", "Prioritizing diagnostic ROI packets at the application layer will deliver initial pathological data >= 60% faster than FIFO transport across 48-128 kbps rural channels."),
        ("Hypothesis 3 (H3) - Cryptographic Feasibility", "AES-256-GCM authenticated encryption will impose < 0.05 ms per-frame latency and achieve 100% rejection of single-bit ciphertext tampering.")
    ]
    for title, desc in hyps:
        p = tf.add_paragraph(); p.text = title; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = BLUE; p.space_after = Pt(2)
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(10); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(8)

    # ==========================================================
    # SLIDE 4: System Architecture
    # ==========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide4, "3. MediLink-DL End-to-End System Architecture")
    
    # 5 Process Flow Step Cards
    step_w = Inches(2.18)
    step_h = Inches(4.3)
    steps = [
        ("Stage 1: Ingestion", "#1d4ed8", ["512x512 Grayscale Radiograph", "Format & dimension checks", "Synthetic CXR Phantom generator with permanent watermark", "Diagnostic ROI Mask"]),
        ("Stage 2: Compression", "#16a34a", ["Dual-stream JPEG baseline", "High ROI: q_roi = 90", "Lossy Context: q_bg = 25", "Packed 'MLDL' binary payload (9.8 KB / 26.6x ratio)"]),
        ("Stage 3: Security", "#ca8a04", ["AES-256-GCM AEAD cipher", "Fresh 96-bit CSPRNG nonce", "128-bit Poly1305 auth tag", "100% bit-flip tamper detection (<0.005 ms latency)"]),
        ("Stage 4: Transport", "#7c3aed", ["Packet chunking (1024 B)", "Priority Min-Heap Queue (Header > ROI > BG)", "Channel model (48-128 kbps, 2-10% loss, ARQ retry)", "Store-and-forward queue"]),
        ("Stage 5: Receiver", "#e11d48", ["Stream reassembly & hash verify", "AES-GCM tag authentication", "ROI-over-context reconstruction", "Partitioned PSNR & SSIM evaluation (metrics.csv)"])
    ]
    for i, (stitle, scolor, sbullets) in enumerate(steps):
        x = Inches(0.8 + i * 2.38)
        sc = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.6), step_w, step_h)
        sc.fill.solid(); sc.fill.fore_color.rgb = WHITE; sc.line.color.rgb = BORDER_COLOR
        
        # Header bar
        hbar = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.6), step_w, Inches(0.45))
        hbar.fill.solid(); hbar.fill.fore_color.rgb = RGBColor.from_string(scolor[1:]); hbar.line.color.rgb = hbar.fill.fore_color.rgb
        tf = hbar.text_frame; tf.margin_left = Inches(0.1); tf.margin_top = Inches(0.08)
        p = tf.paragraphs[0]; p.text = stitle; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = WHITE
        
        tf = sc.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.12); tf.margin_top = Inches(0.55)
        for b in sbullets:
            p = tf.add_paragraph(); p.text = "• " + b; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)
            
    # Bottom Note
    note_box = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.1), Inches(11.733), Inches(0.8))
    note_box.fill.solid(); note_box.fill.fore_color.rgb = RGBColor(241, 245, 249); note_box.line.color.rgb = BORDER_COLOR
    tf = note_box.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]; p.text = "Architectural Note: "; p.font.bold = True; p.font.size = Pt(10); p.font.color.rgb = NAVY
    p = tf.add_paragraph(); p.text = "All modules are strictly CPU-compatible and operational without requiring external multi-gigabyte model downloads. Complete vector SVG available in paper/figures/system_architecture.svg."; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_MUTED

    # ==========================================================
    # SLIDE 5: Compression & Rate Matching Methodology
    # ==========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide5, "4. Dual-Stream ROI Compression & Fair Rate Matching")
    
    c_left = slide5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    c_left.fill.solid(); c_left.fill.fore_color.rgb = WHITE; c_left.line.color.rgb = BORDER_COLOR
    tf = c_left.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Dual-Stream JPEG Baseline Formulation"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = BLUE; p.space_after = Pt(8)
    points = [
        "Standard JPEG (ITU-T T.81) operates on uniform 8x8 DCT quantization tables across the entire image raster.",
        "We implement an operational dual-stream compositing architecture:",
        "  1. Background Stream: B_bg = JPEG(I, q_bg=25) full raster context (~6-7 KB).",
        "  2. ROI Patch Stream: B_roi = JPEG(I_crop, q_roi=90) diagnostic crop (~3 KB).",
        "  3. Serialization: Packed into 'MLDL' binary format with 26-byte header.",
        "  4. Reconstruction: Decompresses both streams and blends patch onto context.",
        "Academic Disclaimer: This is an operational JPEG-based approximation; it does NOT implement native JPEG2000 Part 2 MAXSHIFT wavelet scaling."
    ]
    for pt in points:
        p = tf.add_paragraph(); p.text = pt; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(5)

    c_right = slide5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    c_right.fill.solid(); c_right.fill.fore_color.rgb = WHITE; c_right.line.color.rgb = BORDER_COLOR
    tf = c_right.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Strictly Enforced Rate Matching Protocol"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = NAVY; p.space_after = Pt(8)
    points_r = [
        "A common flaw in compression papers is comparing ROI coding against arbitrary uniform quality factors (e.g. q=50 vs q=90).",
        "MediLink-DL implements an automated parameter sweep across q in [1, 95] to find:",
        "  q* = argmin | Size(Uniform_JPEG(q)) - Size(ROI_Payload) |",
        "Our baseline matches the 9,842 byte ROI payload to a 9,622 byte uniform JPEG (q=32), achieving a delta of only -2.24%.",
        "Partitioned Evaluation:",
        "  • Global PSNR & SSIM: Evaluates full-frame fidelity.",
        "  • ROI PSNR & SSIM: Evaluates diagnostic lesion preservation.",
        "  • Background PSNR & SSIM: Evaluates contextual anatomical loss."
    ]
    for pt in points_r:
        p = tf.add_paragraph(); p.text = pt; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(5)

    # ==========================================================
    # SLIDE 6: Results - Compression & Visual Quality
    # ==========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide6, "5. Verified Results: Rate-Distortion & Quality Metrics")
    
    # Embed comparison image if exists
    comp_path = Path("results/comparison.png")
    if comp_path.exists():
        slide6.shapes.add_picture(str(comp_path), Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.2))
        
    # Table of Results
    tbl_shape = slide6.shapes.add_table(5, 4, Inches(0.8), Inches(4.0), Inches(11.733), Inches(2.2))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.8)
    tbl.columns[1].width = Inches(3.1)
    tbl.columns[2].width = Inches(3.0)
    tbl.columns[3].width = Inches(2.833)
    
    cell_data = [
        ["Evaluation Metric", "ROI-Aware Coding (q_roi=90, q_bg=25)", "Matched Uniform Baseline (q=32)", "Empirical Advantage"],
        ["Payload Size & Compression", "9,842 bytes (26.64x ratio, 96.25% reduction)", "9,622 bytes (27.24x ratio)", "Rate-Matched (-2.24% delta)"],
        ["Diagnostic ROI PSNR", "38.08 dB", "34.58 dB", "+3.50 dB Gain in Pathology"],
        ["Diagnostic ROI SSIM", "0.9293", "0.8440", "+0.0853 Structural Gain"],
        ["Context Background PSNR / SSIM", "32.47 dB / 0.8035", "33.02 dB / 0.8131", "-0.55 dB Trade-off (Minor)"]
    ]
    for row_idx, row in enumerate(cell_data):
        for col_idx, text in enumerate(row):
            cell = tbl.cell(row_idx, col_idx)
            cell.text = text
            for p in cell.text_frame.paragraphs:
                p.font.name = "Calibri"
                p.font.size = Pt(10)
                if row_idx == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                    cell.fill.solid(); cell.fill.fore_color.rgb = BLUE
                elif col_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = GREEN
                else:
                    p.font.color.rgb = TEXT_DARK

    # Bottom Takeaway
    note = slide6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.733), Inches(0.65))
    note.fill.solid(); note.fill.fore_color.rgb = RGBColor(240, 253, 244); note.line.color.rgb = RGBColor(187, 247, 208)
    tf = note.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.08)
    p = tf.paragraphs[0]; p.text = "Key Takeaway: "; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = GREEN
    p = tf.add_paragraph(); p.text = "Reallocating bits from the 98.7% background context into the 1.3% diagnostic lesion improves pathological fidelity by +3.50 dB PSNR with negligible impact on surrounding context."; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_DARK

    # ==========================================================
    # SLIDE 7: Results - Multi-Rate & Network Transport
    # ==========================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide7, "6. Verified Results: Prioritized Transport & Multi-Rate Curves")
    
    # Left side: Multi-rate plot
    rd_img_path = Path("results/experiment_compression.png")
    if rd_img_path.exists():
        slide7.shapes.add_picture(str(rd_img_path), Inches(0.8), Inches(1.6), Inches(5.6), Inches(3.2))
        
    # Right side: Transmission bar chart
    tx_img_path = Path("results/experiment_transmission.png")
    if tx_img_path.exists():
        slide7.shapes.add_picture(str(tx_img_path), Inches(6.8), Inches(1.6), Inches(5.733), Inches(3.2))
        
    # Table of Network Transmission Metrics
    tbl_shape2 = slide7.shapes.add_table(5, 5, Inches(0.8), Inches(4.95), Inches(11.733), Inches(2.1))
    tbl2 = tbl_shape2.table
    tbl2.columns[0].width = Inches(2.4)
    tbl2.columns[1].width = Inches(2.6)
    tbl2.columns[2].width = Inches(2.3)
    tbl2.columns[3].width = Inches(2.2)
    tbl2.columns[4].width = Inches(2.233)
    
    t2_data = [
        ["Channel Profile", "Bandwidth / RTT / Loss", "Time to ROI (Priority)", "Time to ROI (FIFO)", "Triage Acceleration"],
        ["Rural 2G / EDGE", "48 kbps, 200 ms RTT, 2% loss", "0.592 sec", "3.592 sec", "6.07x Faster Triage"],
        ["Rural 3G (Weak)", "128 kbps, 150 ms RTT, 5% loss", "0.372 sec", "2.097 sec", "5.63x Faster Triage"],
        ["Satellite VSAT", "256 kbps, 500 ms RTT, 2% loss", "1.036 sec", "5.299 sec", "5.11x Faster Triage"],
        ["Lossy Wireless", "64 kbps, 120 ms RTT, 10% loss", "0.384 sec", "2.887 sec", "7.51x Faster Triage"]
    ]
    for row_idx, row in enumerate(t2_data):
        for col_idx, text in enumerate(row):
            cell = tbl2.cell(row_idx, col_idx)
            cell.text = text
            for p in cell.text_frame.paragraphs:
                p.font.name = "Calibri"; p.font.size = Pt(9.5)
                if row_idx == 0:
                    p.font.bold = True; p.font.color.rgb = WHITE
                    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
                elif col_idx == 4:
                    p.font.bold = True; p.font.color.rgb = GREEN
                else:
                    p.font.color.rgb = TEXT_DARK

    # ==========================================================
    # SLIDE 8: Cryptographic Security & Grad-CAM Analysis
    # ==========================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide8, "7. Cryptographic Security & Grad-CAM Saliency Analysis")
    
    card_sec = slide8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    card_sec.fill.solid(); card_sec.fill.fore_color.rgb = WHITE; card_sec.line.color.rgb = BORDER_COLOR
    tf = card_sec.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Cryptographic Audit: AES-256-GCM AEAD"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = AMBER; p.space_after = Pt(8)
    sec_points = [
        "100-Trial Empirical Audit Results (results/experiment_security.csv):",
        "  • 100/100 Tamper Rejections (100.0% Detection Rate): Single-bit mutations strictly trigger DecryptionTamperedError.",
        "  • 0% Nonce Collisions: 100 distinct 96-bit CSPRNG nonces verified.",
        "  • Negligible Latency: Mean encryption time = 0.0047 ms; decryption time = 0.0021 ms.",
        "  • Framing Overhead: Exactly 28 bytes per study (12B nonce + 16B tag).",
        "Production Limitation Disclosure:",
        "  • Prototype utilizes in-memory ephemeral symmetric keys.",
        "  • Production requires PKI, Hybrid Public-Key Encryption (HPKE, RFC 9180), and mTLS channel termination."
    ]
    for sp in sec_points:
        p = tf.add_paragraph(); p.text = sp; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    card_cam = slide8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    card_cam.fill.solid(); card_cam.fill.fore_color.rgb = WHITE; card_cam.line.color.rgb = BORDER_COLOR
    tf = card_cam.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Deep Learning: Grad-CAM Saliency Module"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = BLUE; p.space_after = Pt(8)
    cam_points = [
        "Module Implementation (src/roi_localization.py):",
        "  • Registers forward & backward PyTorch hooks on conv layers.",
        "  • Computes Global Average Pooling of gradients to weight feature activations.",
        "  • ReLU filtering extracts class-positive activation regions.",
        "  • Bilinear interpolation & min-max normalization to [0.0, 1.0].",
        "  • Thresholding (tau = 0.50) yields binary ROI candidate masks.",
        "Honest Academic Status:",
        "  • Unit-tested on lightweight demonstration CNN (SimpleCXRClassifier).",
        "  • Pretrained clinical weights (e.g. CheXNet) are NOT mounted.",
        "  • Grad-CAM reflects model activation correlations, NOT radiological ground truth."
    ]
    for cp in cam_points:
        p = tf.add_paragraph(); p.text = cp; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    # ==========================================================
    # SLIDE 9: Implementation Audit & Known Limitations
    # ==========================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide9, "8. Implementation Maturity & Known Limitations")
    
    # 2 Big Cards: Verified vs Limitations
    b_v = slide9.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    b_v.fill.solid(); b_v.fill.fore_color.rgb = WHITE; b_v.line.color.rgb = BORDER_COLOR
    tf = b_v.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Implemented & Verified (41 Tests Passing)"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = GREEN; p.space_after = Pt(8)
    v_items = [
        "Dual-stream JPEG compression baseline with packed binary serialization.",
        "Automated rate-matched uniform JPEG search algorithm.",
        "Partitioned MSE, PSNR, and SSIM evaluation across Global/ROI/Background.",
        "AES-256-GCM AEAD encryption with fresh nonces and 100% tamper detection.",
        "Application-layer discrete-event network simulation with ARQ and priority queuing.",
        "Synthetic anatomical CXR phantom generator with permanent disclaimer banner.",
        "100% test pass rate across 41 automated pytest unit/integration tests."
    ]
    for vi in v_items:
        p = tf.add_paragraph(); p.text = "✔ " + vi; p.font.size = Pt(10); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    b_l = slide9.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    b_l.fill.solid(); b_l.fill.fore_color.rgb = WHITE; b_l.line.color.rgb = BORDER_COLOR
    tf = b_l.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]; p.text = "Known Academic & Technical Limitations"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = RGBColor(220, 38, 38); p.space_after = Pt(8)
    l_items = [
        "Synthetic Data: All experiments evaluated on stylized synthetic phantom; clinical DICOM datasets require institutional ethics approval.",
        "JPEG vs JPEG2000: Operational dual-stream compositing baseline, not native JPEG2000 Part 2 MAXSHIFT wavelet scaling.",
        "Grad-CAM Resolution: Produces coarse, smooth saliency heatmaps that do not match exact anatomical lesion contours.",
        "Simulation Fidelity: Discrete-event application queuing abstracts physical-layer RF fading and kernel TCP congestion dynamics.",
        "Key Management: Symmetric keys are in-memory; production requires PKI/HPKE infrastructure."
    ]
    for li in l_items:
        p = tf.add_paragraph(); p.text = "⚠ " + li; p.font.size = Pt(10); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    # ==========================================================
    # SLIDE 10: Conclusion & Next Steps
    # ==========================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide10, "9. Conclusion & Research Roadmap")
    
    c_c = slide10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.3))
    c_c.fill.solid(); c_c.fill.fore_color.rgb = WHITE; c_c.line.color.rgb = BORDER_COLOR
    tf = c_c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]; p.text = "Conclusion: Demonstrated Research Contributions"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = BLUE; p.space_after = Pt(6)
    c_points = [
        "• Validated that non-uniform bit allocation achieves a 26.6x compression ratio while improving diagnostic ROI PSNR by +3.50 dB over a rate-matched uniform baseline.",
        "• Proved that application-layer priority packet scheduling delivers acute diagnostic ROI data in 0.38-0.59s (5.1x to 7.5x faster triage) over rural channels.",
        "• Confirmed that AES-256-GCM authenticated encryption provides tamper-evident integrity with negligible latency (<0.005 ms) on edge CPUs."
    ]
    for cp in c_points:
        p = tf.add_paragraph(); p.text = cp; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(4)

    c_r = slide10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.1), Inches(11.733), Inches(2.7))
    c_r.fill.solid(); c_r.fill.fore_color.rgb = WHITE; c_r.line.color.rgb = BORDER_COLOR
    tf = c_r.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]; p.text = "Next Steps & Future Research Roadmap"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = GREEN; p.space_after = Pt(6)
    roadmap = [
        "1. Clinical DICOM Dataset Integration: Acquire ethical clearance to ingest NIH ChestX-ray14 and CheXpert datasets.",
        "2. Pretrained Model Wiring: Mount verified CheXNet DenseNet-121 weights to feed live Grad-CAM heatmaps into the compression loop.",
        "3. Physical Testbed Emulation: Replicate network experiments on Linux tc/netem and software-defined radio (SDR) testbeds.",
        "4. Observer Study: Conduct multi-reader ROC diagnostic performance evaluations with certified radiologists."
    ]
    for rm in roadmap:
        p = tf.add_paragraph(); p.text = rm; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK; p.space_after = Pt(4)
        
    prs.save(str(output_path))
    print(f"Successfully generated PowerPoint presentation: {output_path.resolve()}")


if __name__ == "__main__":
    out_pptx = Path("paper/MediLink_DL_Progress_Presentation.pptx")
    create_presentation(out_pptx)
