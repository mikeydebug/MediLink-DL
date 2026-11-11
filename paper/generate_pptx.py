"""
MediLink-DL: 18-Slide Progress Presentation Generator
Completely rewritten with updated real experimental data.
Output: paper/MediLink_DL_Progress_Presentation.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# COLOR PALETTE
# ---------------------------------------------------------------------------
DARK_NAVY   = RGBColor(13,  27,  42)    # #0d1b2a — dark backgrounds
INDIGO      = RGBColor(26,  35, 126)    # #1a237e — header bars
WHITE       = RGBColor(255, 255, 255)
RED_ACCENT  = RGBColor(229,  57,  53)   # #e53935 — key numbers / callouts
LIGHT_BG    = RGBColor(248, 249, 255)   # near-white slide backgrounds
TEXT_DARK   = RGBColor(21,  27,  38)    # near-black body text
TABLE_ALT   = RGBColor(232, 234, 246)   # #e8eaf6 — alternating table rows
GREEN       = RGBColor(27,  94,  32)    # #1b5e20 — positive results
MUTED       = RGBColor(96, 125, 139)    # #607d8b — footnotes / captions
ORANGE      = RGBColor(230, 81,   0)    # orange accent
PURPLE      = RGBColor(74,  20, 140)    # purple accent
BLUE_CARD   = RGBColor(21,  101, 192)   # card header blue
TABLE_HDR   = RGBColor(26,  35, 126)    # table header = INDIGO

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(blank_layout)


def fill_bg(slide, color: RGBColor):
    """Fill slide background with a solid color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color: RGBColor, line_color=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, left, top, width, height, text, font_size, bold=False,
                color=TEXT_DARK, align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return txBox


def add_header_bar(slide, title: str):
    """Add full-width INDIGO header bar at the top."""
    bar = add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.6), INDIGO)
    txBox = slide.shapes.add_textbox(Inches(0.15), Inches(0.05), SLIDE_W - Inches(0.3), Inches(0.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"
    return bar


def add_multiline_textbox(slide, left, top, width, height, lines, default_size=13,
                           default_color=TEXT_DARK, default_bold=False, align=PP_ALIGN.LEFT):
    """
    lines: list of dicts or strings.
    Dict keys: text, size, bold, color, italic, align
    """
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        if isinstance(line, str):
            line = {"text": line}
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = line.get("align", align)
        run = p.add_run()
        run.text = line.get("text", "")
        run.font.size = Pt(line.get("size", default_size))
        run.font.bold = line.get("bold", default_bold)
        run.font.italic = line.get("italic", False)
        run.font.color.rgb = line.get("color", default_color)
        run.font.name = "Calibri"
    return txBox


def add_table(slide, left, top, width, height, data, col_widths=None,
              header_color=TABLE_HDR, alt_color=TABLE_ALT,
              header_font_size=11, body_font_size=11, highlight_row=None):
    """
    data: list of lists (first row = header).
    highlight_row: 1-based data row index to highlight (not counting header).
    """
    rows = len(data)
    cols = len(data[0])
    tbl = slide.shapes.add_table(rows, cols, left, top, width, height).table

    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = int(width * cw / total)

    for r_idx, row in enumerate(data):
        is_header = (r_idx == 0)
        is_highlight = (highlight_row is not None and r_idx == highlight_row)
        for c_idx, cell_text in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = str(cell_text)
            # fill
            fill = cell.fill
            fill.solid()
            if is_header:
                fill.fore_color.rgb = header_color
            elif is_highlight:
                fill.fore_color.rgb = alt_color
            elif r_idx % 2 == 0:
                fill.fore_color.rgb = RGBColor(255, 255, 255)
            else:
                fill.fore_color.rgb = RGBColor(240, 242, 255)
            # font
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.CENTER
                for run in para.runs:
                    run.font.name = "Calibri"
                    run.font.size = Pt(header_font_size if is_header else body_font_size)
                    run.font.bold = is_header
                    run.font.color.rgb = WHITE if is_header else TEXT_DARK
    return tbl


def add_card(slide, left, top, width, height, header_text, body_lines,
             header_color=INDIGO, bg_color=LIGHT_BG):
    """Draw a card with colored header + body text."""
    # Card background
    add_rect(slide, left, top, width, height, bg_color, line_color=RGBColor(200,200,220))
    # Header strip
    hdr_h = Inches(0.38)
    add_rect(slide, left, top, width, hdr_h, header_color)
    # Header text
    add_textbox(slide, left + Inches(0.08), top + Inches(0.04),
                width - Inches(0.16), hdr_h - Inches(0.04),
                header_text, 12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # Body
    body_top = top + hdr_h + Inches(0.08)
    body_h = height - hdr_h - Inches(0.1)
    add_multiline_textbox(slide, left + Inches(0.1), body_top,
                          width - Inches(0.2), body_h, body_lines,
                          default_size=11, default_color=TEXT_DARK)


# ---------------------------------------------------------------------------
# SLIDE BUILDERS
# ---------------------------------------------------------------------------

def slide_01_title(prs):
    slide = blank_slide(prs)
    fill_bg(slide, DARK_NAVY)

    # MediLink-DL brand name
    add_textbox(slide, Inches(1), Inches(0.6), Inches(11.333), Inches(0.9),
                "MediLink-DL", 42, bold=True, color=RED_ACCENT, align=PP_ALIGN.CENTER)

    # Full title
    add_textbox(slide, Inches(0.8), Inches(1.4), Inches(11.7), Inches(1.5),
                "A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework\n"
                "for Medical Image Transmission in Rural Telemedicine",
                22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Divider line
    line_shape = add_rect(slide, Inches(2), Inches(2.9), Inches(9.333), Inches(0.03), RED_ACCENT)

    # Authors
    add_textbox(slide, Inches(1), Inches(3.0), Inches(11.333), Inches(0.5),
                "Mayank Soni (CS24B1031)   ·   Nandan (CS24B1034)   ·   Sumedha (AD24B1061)",
                14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Faculty
    add_textbox(slide, Inches(1), Inches(3.55), Inches(11.333), Inches(0.9),
                "Faculty Advisors",
                12, bold=True, color=RGBColor(144,164,174), align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1), Inches(3.95), Inches(11.333), Inches(0.6),
                "Dr. Kaushiki Roy (DIP)   ·   Dr. Mehbub Alam (CN)   ·   Arnav Ghosh (Optimization)",
                13, color=RGBColor(176,190,197), align=PP_ALIGN.CENTER)

    # Institution + date
    add_textbox(slide, Inches(1), Inches(6.7), Inches(11.333), Inches(0.5),
                "IIIT Raichur  |  November 2026",
                12, color=MUTED, align=PP_ALIGN.CENTER)


def slide_02_problem(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "The Problem: 70 Minutes for One X-Ray")

    # Quote box background
    quote_box = add_rect(slide, Inches(0.4), Inches(0.75), Inches(12.533), Inches(1.9),
                         RGBColor(227, 232, 255), line_color=INDIGO)
    add_textbox(slide, Inches(0.55), Inches(0.8), Inches(12.2), Inches(1.8),
                "\u201cA chest X-ray from a rural PHC in Chhattisgarh weighs 25 MB. "
                "The connection runs at 48 kbps. At that rate, the image takes over 70 minutes "
                "to reach the specialist radiologist 300 km away. The patient cannot wait.\u201d",
                16, bold=False, color=DARK_NAVY, align=PP_ALIGN.CENTER)

    # Three cards
    card_top = Inches(2.78)
    card_h   = Inches(4.2)
    card_w   = Inches(3.9)
    gap      = Inches(0.3)

    add_card(slide, Inches(0.35), card_top, card_w, card_h,
             "\U0001F4F6  Bandwidth Challenge", [
                 {"text": "2G / EDGE rural connectivity", "bold": True},
                 {"text": ""},
                 "Typical bandwidth: 48 – 128 kbps",
                 "Round-trip latency: > 150 ms",
                 "Packet loss: 2 – 10%",
                 "",
                 "Raw 512\u00d7512 PNG \u2248 262 KB",
                 "DICOM at 25 MB = 70+ min to transmit",
             ], header_color=BLUE_CARD)

    add_card(slide, Inches(0.35) + card_w + gap, card_top, card_w, card_h,
             "\U0001F5BC  Compression Gap", [
                 {"text": "One-size-fits-all codecs", "bold": True},
                 {"text": ""},
                 "Uniform JPEG blurs lung lesions",
                 "Background & ROI treated identically",
                 "",
                 "Diagnosis-critical pixels \u2248 15%",
                 "of image but need > 90 PSNR",
                 "",
                 "Background can be 25\u00d7 more compressed",
                 "without clinical impact",
             ], header_color=ORANGE)

    add_card(slide, Inches(0.35) + (card_w + gap) * 2, card_top, card_w, card_h,
             "\U0001F512  Security Mandate", [
                 {"text": "Untrusted public networks", "bold": True},
                 {"text": ""},
                 "PHC \u2192 hospital over public internet",
                 "No VPN, no dedicated line",
                 "",
                 "HIPAA / GDPR mandate:",
                 "\u2022 Authenticated encryption",
                 "\u2022 Tamper detection",
                 "\u2022 Binding to patient identity",
             ], header_color=RED_ACCENT)


def slide_03_dataset(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Clinical Dataset: 166 Real Chest Radiographs")

    # Left column: dataset facts
    add_rect(slide, Inches(0.3), Inches(0.75), Inches(5.8), Inches(5.1),
             RGBColor(232, 234, 246), line_color=RGBColor(180,190,230))
    add_multiline_textbox(slide, Inches(0.45), Inches(0.85), Inches(5.5), Inches(5.0), [
        {"text": "Dataset Facts", "size": 15, "bold": True, "color": INDIGO},
        {"text": ""},
        {"text": "Source: Cohen et al. (2020) COVID-19 CXR", "size": 13},
        {"text": "Total radiographs: 166", "size": 13, "bold": True},
        {"text": "Unique patients: 110", "size": 13},
        {"text": "Views: PA / AP (posteroanterior / anteroposterior)", "size": 13},
        {"text": "Resolution: 512\u00d7512 grayscale (normalised to [0,1])", "size": 13},
        {"text": ""},
        {"text": "Class distribution:", "size": 13, "bold": True},
        {"text": "  \u2022 COVID-19 positive: 83 images", "size": 13},
        {"text": "  \u2022 COVID-19 negative: 83 images", "size": 13},
        {"text": ""},
        {"text": "Split strategy: Patient-level (not image-level)", "size": 13, "bold": True, "color": INDIGO},
        {"text": "Prevents data leakage from patient anatomy sharing", "size": 11, "italic": True},
    ])

    # Right column: split table
    add_rect(slide, Inches(6.4), Inches(0.75), Inches(6.6), Inches(2.6),
             RGBColor(232, 234, 246), line_color=RGBColor(180,190,230))
    add_textbox(slide, Inches(6.55), Inches(0.85), Inches(6.3), Inches(0.4),
                "Patient-Level Split", 15, bold=True, color=INDIGO)

    split_data = [
        ["Split", "Patients", "Radiographs"],
        ["Train",      "77",        "111"],
        ["Validation", "16",         "26"],
        ["Test",       "17",         "29"],
    ]
    add_table(slide, Inches(6.5), Inches(1.3), Inches(6.3), Inches(1.9),
              split_data, col_widths=[3,2,3])

    # Verification note
    add_rect(slide, Inches(6.4), Inches(3.5), Inches(6.6), Inches(1.1),
             RGBColor(255,243,224), line_color=ORANGE)
    add_multiline_textbox(slide, Inches(6.55), Inches(3.6), Inches(6.3), Inches(1.0), [
        {"text": "P_train \u2229 P_val = \u2205,  P_train \u2229 P_test = \u2205",
         "size": 12, "bold": True, "color": DARK_NAVY},
        {"text": "Verified programmatically in test suite", "size": 11, "color": MUTED},
    ])

    # Why note
    add_rect(slide, Inches(0.3), Inches(5.95), Inches(12.7), Inches(1.3),
             RGBColor(255,250,240), line_color=ORANGE)
    add_multiline_textbox(slide, Inches(0.5), Inches(6.05), Inches(12.3), Inches(1.2), [
        {"text": "\u26a0\ufe0f  Why patient-level splits?", "size": 12, "bold": True, "color": ORANGE},
        {"text": "Image-level splitting lets the model memorise patient anatomy across splits, "
                 "artificially inflating test accuracy. By assigning every radiograph of a patient "
                 "to exactly one partition, we ensure the model generalises to unseen patients.",
         "size": 11, "color": TEXT_DARK},
    ])


def slide_04_architecture(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "MediLink-DL: Five-Stage Cross-Layer Pipeline")

    # Intro
    add_textbox(slide, Inches(0.4), Inches(0.72), Inches(12.5), Inches(0.45),
                "Every stage is designed, implemented, tested, and benchmarked end-to-end:",
                13, color=TEXT_DARK)

    stage_colors = [BLUE_CARD, PURPLE, GREEN, ORANGE, RED_ACCENT]
    stage_titles = ["1. Ingestion", "2. AI Localization", "3. Dual-Stream Compress",
                    "4. Encrypt", "5. Priority Transport"]
    stage_bodies = [
        ["512\u00d7512 grayscale radiograph", "DICOM / PNG format validation",
         "Normalise pixel values to [0,1]", "Patient metadata extraction"],
        ["MedicalResNet18 (1-ch conv1)", "Differentiable Grad-CAM", "Threshold \u03c4=0.35",
         "ROI bounding box extraction", "Pointing Game: 93.10%"],
        ["q_roi = 90 (ROI patch)", "q_bg = 25 (background)", "28-byte MLDL container",
         "Magic: \u2018MLDL\u2019 | ver | H,W | bbox", "Output: 26,515 bytes (9.89\u00d7)"],
        ["AES-256-GCM (NIST SP 800-38D)", "or Ascon-AEAD128 (SP 800-232)",
         "Fresh 96-bit nonce per packet", "AAD: StudyUID \u2225 PatientID \u2225 Timestamp",
         "Tamper detection: 100%"],
        ["Min-heap packet scheduler", "Priority: Header=30 > ROI=20 > BG=5",
         "T_ROI (Priority) = 0.391 s", "T_ROI (FIFO) = 2.430 s",
         "Speedup: 6.21\u00d7 on GPRS"],
    ]

    box_w = Inches(2.42)
    box_h = Inches(5.5)
    start_x = Inches(0.25)
    gap = Inches(0.1)
    top = Inches(1.25)

    for i in range(5):
        x = start_x + i * (box_w + gap)
        # Header
        add_rect(slide, x, top, box_w, Inches(0.45), stage_colors[i])
        add_textbox(slide, x + Inches(0.06), top + Inches(0.05),
                    box_w - Inches(0.12), Inches(0.4),
                    stage_titles[i], 12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        # Body
        add_rect(slide, x, top + Inches(0.45), box_w, box_h - Inches(0.45),
                 RGBColor(245, 246, 255), line_color=stage_colors[i])
        lines = [{"text": t, "size": 11, "color": TEXT_DARK} for t in stage_bodies[i]]
        add_multiline_textbox(slide, x + Inches(0.1), top + Inches(0.55),
                              box_w - Inches(0.2), box_h - Inches(0.65),
                              lines)
        # Arrow (except last)
        if i < 4:
            ax = x + box_w + Inches(0.01)
            add_textbox(slide, ax, top + Inches(2.3), gap + Inches(0.05), Inches(0.4),
                        "\u2192", 18, bold=True, color=INDIGO, align=PP_ALIGN.CENTER)


def slide_05_model(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Deep Learning: MedicalResNet18")

    # Left column
    add_rect(slide, Inches(0.3), Inches(0.72), Inches(6.0), Inches(6.5),
             RGBColor(240,242,255), line_color=INDIGO)
    add_multiline_textbox(slide, Inches(0.45), Inches(0.82), Inches(5.7), Inches(6.3), [
        {"text": "Architecture Modifications", "size": 15, "bold": True, "color": INDIGO},
        {"text": ""},
        {"text": "Why modify ResNet-18?", "size": 13, "bold": True},
        {"text": "Chest X-rays are single-channel grayscale \u2014 not RGB.", "size": 12},
        {"text": "Standard ImageNet conv1 expects 3 channels.", "size": 12},
        {"text": ""},
        {"text": "\u2022 conv1: 3-channel \u2192 1-channel input", "size": 13, "bold": True, "color": BLUE_CARD},
        {"text": "   7\u00d77 kernel, stride 2, padding 3", "size": 12},
        {"text": "\u2022 4 residual blocks (unchanged)", "size": 13},
        {"text": "   \u2192 512\u00d77\u00d77 feature maps at layer4[-1]", "size": 12},
        {"text": "\u2022 Global Average Pooling", "size": 13},
        {"text": "\u2022 FC layer \u2192 2 logits", "size": 13},
        {"text": "   Binary: COVID positive / negative", "size": 12},
        {"text": ""},
        {"text": "\u2022 Pretrained ImageNet weights", "size": 13},
        {"text": "   conv1 re-initialised, all other layers fine-tuned", "size": 12},
    ])

    # Right column
    add_rect(slide, Inches(6.55), Inches(0.72), Inches(6.45), Inches(6.5),
             RGBColor(255,243,224), line_color=ORANGE)
    add_multiline_textbox(slide, Inches(6.7), Inches(0.82), Inches(6.1), Inches(6.3), [
        {"text": "Differentiable Grad-CAM", "size": 15, "bold": True, "color": ORANGE},
        {"text": ""},
        {"text": "Standard Grad-CAM (inference only):", "size": 13, "bold": True},
        {"text": "Computed after training is complete.", "size": 12},
        {"text": "Gradients are detached from the computation graph.", "size": 12},
        {"text": "Cannot influence model weights.", "size": 12},
        {"text": ""},
        {"text": "Our approach (training time):", "size": 13, "bold": True, "color": RED_ACCENT},
        {"text": "torch.autograd.grad(..., create_graph=True)", "size": 11,
         "color": DARK_NAVY},
        {"text": "Second-order gradients flow BACK through the saliency map during training.",
         "size": 12},
        {"text": ""},
        {"text": "\u201cThe heatmap actively influences its own training.\u201d",
         "size": 13, "bold": True, "italic": True, "color": INDIGO},
        {"text": ""},
        {"text": "Effect: forces the model to attend to lung parenchyma\n"
                 "and not shortcut on scanner artifacts, diaphragm,\n"
                 "or image borders.", "size": 12},
    ])


def slide_06_loss(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Multi-Task Loss: Kendall & Gal (CVPR 2018) Uncertainty Weighting")

    # Problem statement
    add_rect(slide, Inches(0.4), Inches(0.72), Inches(12.5), Inches(0.8),
             RGBColor(232,234,246), line_color=INDIGO)
    add_textbox(slide, Inches(0.55), Inches(0.78), Inches(12.2), Inches(0.7),
                "Problem: We need the model to simultaneously classify disease AND localise lesions. "
                "How do we balance the two loss terms without manual tuning?",
                13, color=TEXT_DARK)

    # Formula box
    add_rect(slide, Inches(0.8), Inches(1.65), Inches(11.7), Inches(1.5),
             DARK_NAVY, line_color=RED_ACCENT)
    add_textbox(slide, Inches(1.0), Inches(1.75), Inches(11.3), Inches(1.3),
                "L_total  =  \u00bd\u00b7exp(\u2212s\u2081)\u00b7L_cls  +  \u00bd\u00b7exp(\u2212s\u2082)\u00b7L_loc  +  \u00bd\u00b7s\u2081  +  \u00bd\u00b7s\u2082",
                26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Explanation boxes
    terms = [
        ("s\u2081 = log(\u03c3\u2081\u00b2)", "Learnable log-variance for classification task.\nOptimised jointly with network weights by AdamW."),
        ("s\u2082 = log(\u03c3\u2082\u00b2)", "Learnable log-variance for localisation task.\nOptimised jointly with network weights by AdamW."),
        ("L_cls", "Cross-entropy classification loss.\nMeasures COVID +/\u2212 prediction error."),
        ("L_loc", "(1\u2212\u03b2)\u00b7L_align + \u03b2\u00b7L_dice\nSpatial localisation: alignment + contour quality."),
    ]
    box_w = Inches(2.95)
    box_h = Inches(1.9)
    gap = Inches(0.2)
    start_x = Inches(0.4)
    top = Inches(3.3)
    for i, (term, desc) in enumerate(terms):
        x = start_x + i * (box_w + gap)
        add_rect(slide, x, top, box_w, box_h, RGBColor(240,242,255), line_color=INDIGO)
        add_rect(slide, x, top, box_w, Inches(0.42), INDIGO)
        add_textbox(slide, x + Inches(0.06), top + Inches(0.04), box_w - Inches(0.12), Inches(0.38),
                    term, 14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(slide, x + Inches(0.1), top + Inches(0.5), box_w - Inches(0.2), box_h - Inches(0.6),
                    desc, 11, color=TEXT_DARK)

    # Key insight
    add_rect(slide, Inches(0.4), Inches(5.35), Inches(12.5), Inches(0.95),
             RGBColor(255,235,238), line_color=RED_ACCENT)
    add_textbox(slide, Inches(0.6), Inches(5.45), Inches(12.1), Inches(0.85),
                "\U0001F9E0  Both s\u2081 and s\u2082 are trained by AdamW alongside network weights \u2014 "
                "the model discovers the right balance automatically. No manual \u03bb tuning required.",
                13, bold=True, color=DARK_NAVY, align=PP_ALIGN.CENTER)

    # Localization loss breakdown
    add_multiline_textbox(slide, Inches(0.4), Inches(6.4), Inches(12.5), Inches(0.9), [
        {"text": "L_align: KL divergence between Grad-CAM heatmap and clinical lesion annotation",
         "size": 11, "color": MUTED},
        {"text": "L_dice: Soft Dice loss on binarised heatmap vs ground-truth mask  |  "
                 "\u03b2 = 0.5 (equal weighting)", "size": 11, "color": MUTED},
    ])


def slide_07_ablation(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Ablation Study: 4 Training Regimes on 29 Test Radiographs (17 Patients)")

    data = [
        ["Regime", "Accuracy", "F1 Score", "ROI Dice", "Pointing Game", "Energy in ROI"],
        ["A: Classification-Only", "68.97%", "0.8163", "0.4764", "65.52%", "49.21%"],
        ["B: Combined Loss (Proposed)", "68.97%", "0.8163", "0.6856", "93.10%", "80.32%"],
        ["C: Alignment-Only", "68.97%", "0.8163", "0.5799", "96.55%", "82.22%"],
        ["D: Dice-Only", "75.86%", "0.8511", "0.7107", "82.76%", "69.36%"],
    ]

    add_table(slide, Inches(0.3), Inches(0.72), Inches(12.7), Inches(3.2),
              data, col_widths=[4, 2, 2, 2, 3, 3],
              header_font_size=12, body_font_size=12,
              highlight_row=2)  # row index 2 = Regime B

    # Explanation cards
    card_top = Inches(4.05)
    card_h = Inches(2.8)
    card_w = Inches(4.0)
    gap = Inches(0.25)

    add_card(slide, Inches(0.3), card_top, card_w, card_h, "What is Pointing Game?", [
        "A hit is scored when the highest-activation",
        "pixel of the Grad-CAM heatmap falls inside",
        "the annotated lesion region.",
        "",
        {"text": "93.10%", "size": 20, "bold": True, "color": RED_ACCENT,
         "align": PP_ALIGN.CENTER},
        {"text": "Regime B hit rate", "size": 11, "italic": True,
         "align": PP_ALIGN.CENTER},
    ], header_color=INDIGO)

    add_card(slide, Inches(0.3) + card_w + gap, card_top, card_w, card_h, "What is Energy in ROI?", [
        "Total heatmap activation mass that falls",
        "INSIDE the clinical bounding box vs total.",
        "Measures gradient leakage outside lesion.",
        "",
        {"text": "80.32%", "size": 20, "bold": True, "color": RED_ACCENT,
         "align": PP_ALIGN.CENTER},
        {"text": "Regime B energy containment", "size": 11, "italic": True,
         "align": PP_ALIGN.CENTER},
    ], header_color=GREEN)

    add_card(slide, Inches(0.3) + (card_w + gap) * 2, card_top, card_w, card_h,
             "Universal Guarantee", [
        {"text": "100% Recall", "size": 24, "bold": True, "color": RED_ACCENT,
         "align": PP_ALIGN.CENTER},
        {"text": "across ALL 4 regimes", "size": 12, "italic": True,
         "align": PP_ALIGN.CENTER},
        {"text": ""},
        "Zero missed COVID cases.",
        "Zero false negatives on viral lesions.",
        "",
        {"text": "Safety-first: miss-rate = 0", "size": 12, "bold": True,
         "color": GREEN},
    ], header_color=RED_ACCENT)


def slide_08_regime_b(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Why We Chose Regime B as the Operational Model")

    card_top = Inches(0.75)
    card_h = Inches(4.4)
    card_w = Inches(4.0)
    gap = Inches(0.25)

    add_card(slide, Inches(0.3), card_top, card_w, card_h, "Regime C: Alignment-Only", [
        {"text": "Best Pointing Game: 96.55%", "size": 13, "bold": True, "color": GREEN},
        {"text": "Energy in ROI: 82.22%", "size": 13},
        {"text": ""},
        "The gradient PEAK hits the lesion centre,",
        "but the heatmap BLEEDS outside the",
        "clinical bounding box.",
        "",
        "Consequence: when cropping the ROI patch",
        "for compression, we over-estimate the",
        "region and include non-diagnostic tissue.",
        "",
        {"text": "\u26a0 Peak accuracy \u2260 spatial containment",
         "size": 12, "bold": True, "color": ORANGE},
    ], header_color=BLUE_CARD)

    add_card(slide, Inches(0.3) + card_w + gap, card_top, card_w, card_h, "Regime D: Dice-Only", [
        {"text": "Best Dice: 0.7107", "size": 13, "bold": True, "color": GREEN},
        {"text": "Best Accuracy: 75.86%", "size": 13, "bold": True, "color": GREEN},
        {"text": "Energy in ROI: 69.36%", "size": 13},
        {"text": ""},
        "Good lesion CONTOUR quality, but",
        "attention leaks into surrounding tissue.",
        "",
        "Energy in ROI drops to 69.36% \u2014 nearly",
        "31% of gradient mass escapes outside",
        "the pathological region.",
        "",
        {"text": "\u26a0 Good outline, poor containment",
         "size": 12, "bold": True, "color": ORANGE},
    ], header_color=PURPLE)

    add_card(slide, Inches(0.3) + (card_w + gap) * 2, card_top, card_w, card_h,
             "Regime B: Combined Loss \u2714 Selected", [
        {"text": "Pointing Game: 93.10%", "size": 13, "bold": True, "color": GREEN},
        {"text": "Energy in ROI: 80.32%", "size": 13, "bold": True, "color": GREEN},
        {"text": ""},
        "Best COMBINED clinical guarantee.",
        "",
        "Gradient is locked inside the",
        "pathological region \u2014 not just peaked",
        "at the lesion centre.",
        "",
        "Compression ROI bounding box is",
        "tight and diagnostically faithful.",
        "",
        {"text": "\u2714 Optimal for telemedicine deployment",
         "size": 12, "bold": True, "color": GREEN},
    ], header_color=GREEN)

    # Bottom callout
    add_rect(slide, Inches(0.3), Inches(5.3), Inches(12.7), Inches(1.1),
             DARK_NAVY)
    add_textbox(slide, Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.0),
                "100% Recall across ALL regimes \u2014 zero false negatives on viral lesions. No COVID case missed.",
                20, bold=True, color=RED_ACCENT, align=PP_ALIGN.CENTER)


def slide_09_compression(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Dual-Stream ROI-Aware Compression")

    # Flow diagram using text boxes
    diagram_lines = [
        "        [Full 512\u00d7512 Radiograph  |  262,144 raw bytes]",
        "                              \u2193",
        "          [Grad-CAM Mask  \u03c4 = 0.35  \u2192  ROI bounding box]",
        "               \u2193                                    \u2193",
        "   [Background Stream]              [ROI Patch Stream]",
        "    JPEG quality q = 25               JPEG quality q = 90",
        "       ~17,000 bytes                     ~9,000 bytes",
        "       (non-diagnostic)                 (diagnostic lesion)",
        "               \u2193                                    \u2193",
        "        \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 [28-byte MLDL Binary Container] \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518",
        "   Magic:'MLDL' | Version | H, W | q_roi, q_bg | bbox coords | stream lengths",
        "                              \u2193",
        "              [26,515 bytes transmitted over the wire]",
        "              [Compression ratio: 9.89\u00d7  |  Reduction: 89.89%]",
    ]

    add_rect(slide, Inches(0.4), Inches(0.72), Inches(12.5), Inches(5.5),
             DARK_NAVY)
    add_multiline_textbox(slide, Inches(0.55), Inches(0.78), Inches(12.2), Inches(5.4),
                          [{"text": t, "size": 13, "color": WHITE} for t in diagram_lines])

    # Stats row
    stats = [
        ("9.89\u00d7", "Compression Ratio"),
        ("89.89%", "Data Reduction"),
        ("48.31 dB", "ROI PSNR"),
        ("0.9891", "ROI SSIM"),
        ("28 bytes", "Container Header"),
    ]
    bw = Inches(2.4)
    bh = Inches(0.9)
    gx = Inches(0.1)
    sx = Inches(0.4)
    sy = Inches(6.4)
    for i, (num, label) in enumerate(stats):
        x = sx + i * (bw + gx)
        add_rect(slide, x, sy, bw, bh, INDIGO)
        add_textbox(slide, x + Inches(0.05), sy + Inches(0.03), bw - Inches(0.1), Inches(0.42),
                    num, 20, bold=True, color=RED_ACCENT, align=PP_ALIGN.CENTER)
        add_textbox(slide, x + Inches(0.05), sy + Inches(0.46), bw - Inches(0.1), Inches(0.4),
                    label, 10, color=WHITE, align=PP_ALIGN.CENTER)


def slide_10_compression_results(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Rate-Matched Comparison: Same Byte Budget, Different Quality")

    # Table
    data = [
        ["Scheme", "Bytes", "Reduction", "ROI PSNR", "ROI SSIM"],
        ["Dual-Stream (Ours)",             "26,515", "89.89%", "48.31 dB", "0.9891"],
        ["Rate-Matched Uniform JPEG (q=32)", "26,035", "90.07%", "38.18 dB", "0.9312"],
        ["High-Quality Uniform JPEG (q=90)", "98,412", "62.46%", "48.31 dB", "0.9897"],
    ]
    tbl_h = Inches(1.9)
    add_table(slide, Inches(0.3), Inches(0.78), Inches(12.7), tbl_h, data,
              col_widths=[5, 2.5, 2.5, 2.5, 2.5],
              header_font_size=12, body_font_size=12)

    # Big callout
    add_rect(slide, Inches(0.3), Inches(2.85), Inches(12.7), Inches(0.85), DARK_NAVY)
    add_textbox(slide, Inches(0.5), Inches(2.95), Inches(12.3), Inches(0.75),
                "+10.13 dB  ROI PSNR gain \u2014 at the SAME byte budget",
                26, bold=True, color=RED_ACCENT, align=PP_ALIGN.CENTER)

    # Explanation
    add_multiline_textbox(slide, Inches(0.4), Inches(3.82), Inches(6.0), Inches(3.5), [
        {"text": "What this means clinically:", "size": 14, "bold": True, "color": INDIGO},
        {"text": ""},
        {"text": "10 dB PSNR corresponds to roughly a 3\u00d7 improvement in pixel fidelity "
                 "in the diagnostic region. Uniform JPEG at q=32 blurs lung lesions to meet "
                 "the same byte budget. Our dual-stream preserves them at q=90.",
         "size": 13},
        {"text": ""},
        {"text": "Fair comparison note:", "size": 13, "bold": True, "color": INDIGO},
        {"text": "We ran an automated binary search to find q=32 as the uniform JPEG quality "
                 "that matches our byte budget (26,515 bytes). No cherry-picking.",
         "size": 12, "italic": True, "color": MUTED},
    ])

    # Try to embed comparison image
    comp_img = "/Users/mayanksoni/Desktop/MediLink-DL/results/comparison.png"
    if not os.path.exists(comp_img):
        comp_img = "/Users/mayanksoni/Desktop/MediLink-DL/paper/comparison.png"
    if os.path.exists(comp_img):
        slide.shapes.add_picture(comp_img, Inches(6.6), Inches(3.75), Inches(6.4), Inches(3.5))
    else:
        add_rect(slide, Inches(6.6), Inches(3.75), Inches(6.4), Inches(3.5),
                 RGBColor(220,225,240), line_color=INDIGO)
        add_textbox(slide, Inches(6.65), Inches(5.1), Inches(6.3), Inches(0.6),
                    "[Compression comparison figure]", 12, color=MUTED, align=PP_ALIGN.CENTER)


def slide_11_crypto(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Authenticated Encryption: Three Ciphers Benchmarked")

    data = [
        ["Cipher", "Standard", "Encrypt Latency", "Throughput", "Tamper Detection"],
        ["AES-256-GCM",     "NIST SP 800-38D",  "2.2 \u00b5s",    "465 MB/s",   "100% (1024-byte chunk)"],
        ["Ascon-AEAD128",   "NIST SP 800-232",  "412 \u00b5s",    "2.48 MB/s",  "100%"],
        ["MLMC-128 (Research)", "Custom / Novel","7,889 \u00b5s",  "0.13 MB/s",  "100%"],
    ]
    add_table(slide, Inches(0.3), Inches(0.78), Inches(12.7), Inches(2.0), data,
              col_widths=[4, 4, 3, 3, 4],
              header_font_size=12, body_font_size=12)

    # Key insight box
    add_rect(slide, Inches(0.3), Inches(2.95), Inches(12.7), Inches(1.5),
             DARK_NAVY)
    add_multiline_textbox(slide, Inches(0.5), Inches(3.05), Inches(12.3), Inches(1.35), [
        {"text": "\U0001F4A1  Real-World Overhead Context:", "size": 14, "bold": True, "color": WHITE},
        {"text": "On a 64 kbps rural channel, transmitting 1,024 bytes takes ~128 ms. "
                 "AES-256-GCM encryption of that same payload takes 0.0022 ms. "
                 "Cryptographic overhead = 0.002% of airtime.",
         "size": 13, "color": RGBColor(200,220,255)},
        {"text": "Security is essentially free on the wireless link.",
         "size": 14, "bold": True, "color": RED_ACCENT},
    ])

    # AAD binding
    add_rect(slide, Inches(0.3), Inches(4.6), Inches(12.7), Inches(1.1),
             RGBColor(232,234,246), line_color=INDIGO)
    add_multiline_textbox(slide, Inches(0.5), Inches(4.7), Inches(12.3), Inches(1.0), [
        {"text": "AAD (Additional Authenticated Data) binding:", "size": 13, "bold": True, "color": INDIGO},
        {"text": "StudyUID \u2225 PatientID \u2225 Timestamp \u2225 HeaderChecksum",
         "size": 13, "color": DARK_NAVY},
        {"text": "Any bit-flip in the ciphertext OR the routing metadata triggers immediate decryption failure.",
         "size": 12, "color": MUTED},
    ])

    # Three cipher comparison
    cipher_top = Inches(5.85)
    cipher_h   = Inches(1.4)
    cipher_w   = Inches(4.0)
    gap = Inches(0.25)

    for i, (name, detail, col) in enumerate([
        ("AES-256-GCM", "NIST-standardised. Hardware AES-NI support on all modern CPUs.\n"
                        "Best choice for cloud gateways and urban relay nodes.", BLUE_CARD),
        ("Ascon-AEAD128", "NIST SP 800-232 (2024). Lightweight sponge construction.\n"
                          "Target: IoT / microcontroller gateway nodes.", GREEN),
        ("MLMC-128", "Novel research cipher. PRESENT S-box + circulant MDS.\n"
                     "Designed for future hardware synthesis evaluation.", PURPLE),
    ]):
        x = Inches(0.3) + i * (cipher_w + gap)
        add_rect(slide, x, cipher_top, cipher_w, cipher_h, RGBColor(240,242,255), line_color=col)
        add_rect(slide, x, cipher_top, cipher_w, Inches(0.38), col)
        add_textbox(slide, x + Inches(0.06), cipher_top + Inches(0.04),
                    cipher_w - Inches(0.12), Inches(0.34),
                    name, 12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(slide, x + Inches(0.1), cipher_top + Inches(0.44),
                    cipher_w - Inches(0.2), cipher_h - Inches(0.5),
                    detail, 10, color=TEXT_DARK)


def slide_12_mlmc(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Research Lab: MLMC-128 Lightweight Block Cipher")

    # Left: Design
    add_rect(slide, Inches(0.3), Inches(0.72), Inches(5.9), Inches(6.5),
             RGBColor(240,235,255), line_color=PURPLE)
    add_multiline_textbox(slide, Inches(0.45), Inches(0.82), Inches(5.6), Inches(6.3), [
        {"text": "Design", "size": 15, "bold": True, "color": PURPLE},
        {"text": ""},
        {"text": "\u2022 128-bit block cipher", "size": 13, "bold": True},
        {"text": "\u2022 PRESENT S-boxes (4-bit, optimal non-linearity)", "size": 12},
        {"text": "\u2022 Circulant MDS diffusion layer", "size": 12},
        {"text": ""},
        {"text": "PROPOSED-OIN-MDS Matrix:", "size": 13, "bold": True, "color": PURPLE},
        {"text": "Hadamard instantiation Had(1, 8, 13, 5)", "size": 12},
        {"text": "over GF(2\u2074)/(x\u2074+x+1)", "size": 12},
        {"text": ""},
        {"text": "Key Properties:", "size": 13, "bold": True},
        {"text": "\u2022 Involutory: M\u00b2 = I (self-inverse)", "size": 12, "color": GREEN},
        {"text": "   No separate inverse circuit needed", "size": 11, "color": MUTED},
        {"text": "\u2022 48 Paar CSE XORs \u2248 96 Gate Equivalents", "size": 12},
        {"text": "\u2022 Branch number B_nibble = 5 (optimal MDS)", "size": 12},
        {"text": ""},
        {"text": "Implementation: Pure Python + GF arithmetic", "size": 11, "color": MUTED},
    ])

    # Right: Verification
    add_rect(slide, Inches(6.45), Inches(0.72), Inches(6.55), Inches(6.5),
             RGBColor(255,243,224), line_color=ORANGE)
    add_multiline_textbox(slide, Inches(6.6), Inches(0.82), Inches(6.25), Inches(6.3), [
        {"text": "Formal Verification", "size": 15, "bold": True, "color": ORANGE},
        {"text": ""},
        {"text": "SAC Test Result:", "size": 13, "bold": True},
        {"text": "49.92% bit-flip avalanche across 4,800 trials", "size": 12},
        {"text": "(Theoretical ideal = 50.00%  \u2014  difference < 0.1%)", "size": 11, "color": MUTED},
        {"text": ""},
        {"text": "4 Impossibility Theorems Proved:", "size": 13, "bold": True, "color": RED_ACCENT},
        {"text": ""},
        {"text": "1. No 4\u00d74 Hankel matrix can be simultaneously", "size": 12, "bold": True},
        {"text": "   MDS and involutory.", "size": 12},
        {"text": ""},
        {"text": "2. Symmetric Cauchy matrices collapse under", "size": 12, "bold": True},
        {"text": "   self-inversion (det \u2261 0 in GF(2^k)).", "size": 12},
        {"text": ""},
        {"text": "3. Galois matrix symmetry obstruction:", "size": 12, "bold": True},
        {"text": "   \u03b1 = \u03b2 only trivially (\u03b1 \u2208 {0,1}).", "size": 12},
        {"text": ""},
        {"text": "4. Tensor product A\u2297B bounded:", "size": 12, "bold": True},
        {"text": "   B \u2264 4; cannot achieve optimal MDS.", "size": 12},
    ])


def slide_13_transport(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Priority Packet Scheduling: 6.21\u00d7 Faster Clinical Triage")

    data = [
        ["Channel", "Bandwidth / RTT", "Priority T_ROI", "FIFO T_ROI", "Speedup"],
        ["Rural 2G/EDGE", "48 kbps, 200ms RTT",  "0.592 s", "3.592 s", "6.07\u00d7"],
        ["Rural GPRS",    "64 kbps, 120ms RTT",  "0.391 s", "2.430 s", "6.21\u00d7"],
        ["Weak 3G",       "128 kbps, 150ms RTT", "0.372 s", "2.097 s", "5.63\u00d7"],
        ["Satellite VSAT","256 kbps, 500ms RTT", "1.036 s", "5.299 s", "5.11\u00d7"],
    ]
    add_table(slide, Inches(0.3), Inches(0.78), Inches(12.7), Inches(2.4), data,
              col_widths=[4, 4, 3, 3, 2.5],
              header_font_size=12, body_font_size=12)

    # Scheduling explanation
    add_rect(slide, Inches(0.3), Inches(3.3), Inches(12.7), Inches(1.1), DARK_NAVY)
    add_multiline_textbox(slide, Inches(0.5), Inches(3.4), Inches(12.3), Inches(1.0), [
        {"text": "Min-Heap Priority Scheduler:", "size": 14, "bold": True, "color": WHITE},
        {"text": "Header chunk (28 bytes) = Priority 30   \u203a\u203a   "
                 "ROI chunks = Priority 20   \u203a\u203a   Background chunks = Priority 5",
         "size": 13, "color": RGBColor(200,220,255)},
    ])

    add_textbox(slide, Inches(0.4), Inches(4.52), Inches(12.5), Inches(0.6),
                "Under FIFO scheduling, the radiologist waits through 65% background pixels before "
                "seeing ANY diagnostic data. Priority scheduling flips that \u2014 ROI arrives first.",
                13, color=TEXT_DARK)

    # Try to embed experiment_transmission image
    tx_img = "/Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.png"
    if not os.path.exists(tx_img):
        tx_img = "/Users/mayanksoni/Desktop/MediLink-DL/paper/experiment_transmission.png"
    if os.path.exists(tx_img):
        slide.shapes.add_picture(tx_img, Inches(0.4), Inches(5.15), Inches(12.5), Inches(2.1))
    else:
        add_rect(slide, Inches(0.4), Inches(5.15), Inches(12.5), Inches(2.1),
                 RGBColor(220,225,240), line_color=INDIGO)
        add_textbox(slide, Inches(0.5), Inches(5.9), Inches(12.3), Inches(0.6),
                    "[Transmission timeline experiment figure]", 12, color=MUTED, align=PP_ALIGN.CENTER)


def slide_14_pareto(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Multi-Objective Pareto Optimization: 108 Configurations")

    # Grid layout explanation
    add_rect(slide, Inches(0.3), Inches(0.78), Inches(7.6), Inches(3.3),
             RGBColor(240,242,255), line_color=INDIGO)
    add_multiline_textbox(slide, Inches(0.45), Inches(0.88), Inches(7.3), Inches(3.1), [
        {"text": "Parameter Search Grid", "size": 14, "bold": True, "color": INDIGO},
        {"text": ""},
        {"text": "Compression parameters:", "size": 12, "bold": True},
        {"text": "  \u03c4 (Grad-CAM threshold) \u2208 {0.25, 0.35, 0.50}", "size": 12},
        {"text": "  q_roi (ROI quality) \u2208 {85, 90, 95}", "size": 12},
        {"text": "  q_bg (background quality) \u2208 {15, 25, 40}", "size": 12},
        {"text": ""},
        {"text": "Security parameters:", "size": 12, "bold": True},
        {"text": "  Cipher: AES-256-GCM vs Ascon-AEAD128", "size": 12},
        {"text": ""},
        {"text": "Transport parameters:", "size": 12, "bold": True},
        {"text": "  Scheduler: Priority vs FIFO", "size": 12},
        {"text": "  Bandwidth: 48 / 64 / 128 kbps", "size": 12},
        {"text": ""},
        {"text": "Total: 3\u00d73\u00d73\u00d72\u00d72\u00d73 = 108 configurations evaluated end-to-end",
         "size": 13, "bold": True, "color": RED_ACCENT},
    ])

    # Results panel
    add_rect(slide, Inches(8.1), Inches(0.78), Inches(5.0), Inches(3.3),
             RGBColor(255,243,224), line_color=ORANGE)
    add_multiline_textbox(slide, Inches(8.25), Inches(0.88), Inches(4.7), Inches(3.1), [
        {"text": "Pareto Results", "size": 14, "bold": True, "color": ORANGE},
        {"text": ""},
        {"text": "26", "size": 36, "bold": True, "color": RED_ACCENT, "align": PP_ALIGN.CENTER},
        {"text": "non-dominated Pareto configurations", "size": 12,
         "italic": True, "align": PP_ALIGN.CENTER},
        {"text": ""},
        {"text": "Objectives optimised simultaneously:", "size": 12, "bold": True},
        {"text": "  \u2191 Maximise ROI SSIM", "size": 12},
        {"text": "  \u2193 Minimise total bytes transmitted", "size": 12},
        {"text": "  \u2193 Minimise T_roi (ROI delivery time)", "size": 12},
        {"text": "  \u2193 Minimise crypto CPU overhead", "size": 12},
    ])

    # Champion configuration
    add_rect(slide, Inches(0.3), Inches(4.22), Inches(12.7), Inches(2.0), DARK_NAVY)
    add_textbox(slide, Inches(0.5), Inches(4.32), Inches(12.3), Inches(0.5),
                "\U0001F3C6  Operational Champion Configuration",
                16, bold=True, color=WHITE)
    champ_data = [
        ["\u03c4 = 0.35", "q_roi = 90", "q_bg = 25", "AES-256-GCM", "Priority Scheduler", "64 kbps"],
    ]
    add_table(slide, Inches(0.5), Inches(4.9), Inches(12.3), Inches(1.1),
              [["Threshold", "ROI Quality", "BG Quality", "Cipher", "Transport", "Bandwidth"]] + champ_data,
              col_widths=[2.5, 2.5, 2.5, 3, 3, 2],
              header_font_size=11, body_font_size=12)

    # Pareto image
    pareto_img = "/Users/mayanksoni/Desktop/MediLink-DL/results/pareto_frontier.png"
    if not os.path.exists(pareto_img):
        pareto_img = "/Users/mayanksoni/Desktop/MediLink-DL/paper/pareto_frontier.png"
    if os.path.exists(pareto_img):
        slide.shapes.add_picture(pareto_img, Inches(0.3), Inches(6.35), Inches(12.7), Inches(1.0))


def slide_15_results(prs):
    slide = blank_slide(prs)
    fill_bg(slide, DARK_NAVY)
    add_header_bar(slide, "Summary of Verified Results")

    metrics = [
        ("93.10%",    "Pointing Game Hit Rate\n(Regime B, 29 test radiographs)"),
        ("100%",      "Recall: Zero Missed\nCOVID Cases (all regimes)"),
        ("9.89\u00d7",       "Compression Ratio\n(89.89% Data Reduction)"),
        ("+10.13 dB", "ROI PSNR Gain over\nRate-Matched Baseline"),
        ("6.21\u00d7",       "Faster Clinical Triage\n(Priority vs FIFO, GPRS)"),
        ("147/147",   "Automated Tests\nPassing \u2714"),
    ]

    cols = 3
    bw = Inches(4.0)
    bh = Inches(2.5)
    gx = Inches(0.2)
    gy = Inches(0.25)
    start_x = Inches(0.4)
    start_y = Inches(0.72)

    for i, (num, label) in enumerate(metrics):
        col = i % cols
        row = i // cols
        x = start_x + col * (bw + gx)
        y = start_y + row * (bh + gy)
        add_rect(slide, x, y, bw, bh, RGBColor(26,30,60), line_color=INDIGO)
        add_textbox(slide, x + Inches(0.1), y + Inches(0.3), bw - Inches(0.2), Inches(1.1),
                    num, 36, bold=True, color=RED_ACCENT, align=PP_ALIGN.CENTER)
        add_textbox(slide, x + Inches(0.1), y + Inches(1.4), bw - Inches(0.2), Inches(1.0),
                    label, 12, color=WHITE, align=PP_ALIGN.CENTER)


def slide_16_demo(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Live System: MediLink-DL Web Dashboard")

    # URL highlight
    add_rect(slide, Inches(0.4), Inches(0.78), Inches(12.5), Inches(0.75), DARK_NAVY)
    add_textbox(slide, Inches(0.5), Inches(0.84), Inches(12.3), Inches(0.65),
                "https://medilink-dl.onrender.com",
                24, bold=True, color=RED_ACCENT, align=PP_ALIGN.CENTER)

    tabs = [
        ("1. Live Telemedicine Pipeline",
         "Select any of the 29 test CXR images. Adjust \u03c4 (threshold) and compression quality "
         "sliders interactively. Watch real PSNR and SSIM update live as you change parameters. "
         "See the Grad-CAM heatmap, ROI bounding box, and compressed output side-by-side."),
        ("2. Pareto Optimization & Regimes",
         "Interactive Pareto frontier explorer showing all 108 configurations. "
         "Click any point to see its full parameter settings and results. "
         "Compare all 4 training regimes on the ablation table."),
        ("3. NIST SP 800-232 Cryptographic Lab",
         "Live Ascon-AEAD128 encrypt/decrypt with tamper detection. "
         "Flip any bit in the ciphertext and watch decryption fail immediately. "
         "AAD binding demonstration with StudyUID and PatientID."),
        ("4. Research Novelty & Viva Defense",
         "Q&A mode for viva preparation. "
         "Covers all system components, design decisions, and experimental results. "
         "Panel for exploring the 4 impossibility theorems for MLMC-128."),
        ("5. Cryptographic Research Lab",
         "MLMC-128 diffusion matrix explorer. "
         "SAC test runner, GF(2^4) arithmetic visualiser, "
         "branch number computation, and involutory property verifier."),
    ]

    tab_top = Inches(1.65)
    tab_h = Inches(1.03)
    gap = Inches(0.07)

    for i, (title, body) in enumerate(tabs):
        y = tab_top + i * (tab_h + gap)
        add_rect(slide, Inches(0.3), y, Inches(12.7), tab_h,
                 RGBColor(240,242,255), line_color=INDIGO)
        add_rect(slide, Inches(0.3), y, Inches(0.5), tab_h, INDIGO)
        add_textbox(slide, Inches(0.9), y + Inches(0.08), Inches(12.0), Inches(0.38),
                    title, 13, bold=True, color=INDIGO)
        add_textbox(slide, Inches(0.9), y + Inches(0.46), Inches(12.0), Inches(0.5),
                    body, 11, color=TEXT_DARK)


def slide_17_limitations(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Honest Assessment: What We Know and What We Don't")

    # Left column: Limitations
    add_rect(slide, Inches(0.3), Inches(0.72), Inches(6.0), Inches(6.5),
             RGBColor(255,235,238), line_color=RED_ACCENT)
    add_multiline_textbox(slide, Inches(0.45), Inches(0.82), Inches(5.7), Inches(6.3), [
        {"text": "\u26a0  Current Limitations", "size": 15, "bold": True, "color": RED_ACCENT},
        {"text": ""},
        {"text": "1. Dataset scale:", "size": 13, "bold": True},
        {"text": "   166 images is a proof-of-concept.", "size": 12},
        {"text": "   Regulatory pathway (FDA 510k) requires", "size": 12},
        {"text": "   20,000+ image clinical validation.", "size": 12},
        {"text": ""},
        {"text": "2. No radiologist observer study:", "size": 13, "bold": True},
        {"text": "   PSNR > 48 dB is mathematically excellent", "size": 12},
        {"text": "   but unconfirmed by clinical reading.", "size": 12},
        {"text": "   MRMC ROC study required.", "size": 12},
        {"text": ""},
        {"text": "3. Discrete-event simulation:", "size": 13, "bold": True},
        {"text": "   Network model doesn't capture RF fading,", "size": 12},
        {"text": "   TCP congestion, or OS scheduler jitter.", "size": 12},
        {"text": ""},
        {"text": "4. Key management:", "size": 13, "bold": True},
        {"text": "   Symmetric keys are in-memory only.", "size": 12},
        {"text": "   Production needs PKI / HPKE / mTLS.", "size": 12},
    ])

    # Right column: Future Work
    add_rect(slide, Inches(6.55), Inches(0.72), Inches(6.45), Inches(6.5),
             RGBColor(232,255,235), line_color=GREEN)
    add_multiline_textbox(slide, Inches(6.7), Inches(0.82), Inches(6.15), Inches(6.3), [
        {"text": "\U0001F680  Future Work", "size": 15, "bold": True, "color": GREEN},
        {"text": ""},
        {"text": "1. Multi-center DICOM datasets:", "size": 13, "bold": True},
        {"text": "   NIH ChestX-ray14 (112,120 images)", "size": 12},
        {"text": "   CheXpert (224,316 images)", "size": 12},
        {"text": "   Full DICOM pipeline with metadata", "size": 12},
        {"text": ""},
        {"text": "2. Volumetric imaging:", "size": 13, "bold": True},
        {"text": "   3D extension for CT / MRI stacks", "size": 12},
        {"text": "   Temporal ROI tracking across slices", "size": 12},
        {"text": ""},
        {"text": "3. Hardware synthesis:", "size": 13, "bold": True},
        {"text": "   MLMC-128 RTL on TSMC 28nm standard cells", "size": 12},
        {"text": "   Gate count and power benchmarking", "size": 12},
        {"text": ""},
        {"text": "4. Clinical validation:", "size": 13, "bold": True},
        {"text": "   Multi-reader MRMC ROC study", "size": 12},
        {"text": "   Board-certified radiologist panel", "size": 12},
        {"text": "   Comparison against DICOM WADO-RS", "size": 12},
    ])


def slide_18_acknowledgements(prs):
    slide = blank_slide(prs)
    fill_bg(slide, LIGHT_BG)
    add_header_bar(slide, "Acknowledgements")

    # Main acknowledgement text
    add_rect(slide, Inches(0.3), Inches(0.78), Inches(12.7), Inches(3.5),
             RGBColor(240,242,255), line_color=INDIGO)
    add_multiline_textbox(slide, Inches(0.5), Inches(0.9), Inches(12.3), Inches(3.3), [
        {"text": "Faculty Advisors", "size": 14, "bold": True, "color": INDIGO},
        {"text": ""},
        {"text": "We owe particular thanks to Dr. Kaushiki Roy for her guidance through the DIP module \u2014 "
                 "specifically her feedback on partitioned fidelity metrics and why global PSNR doesn\u2019t "
                 "tell you enough about diagnostic quality.", "size": 13},
        {"text": ""},
        {"text": "Dr. Mehbub Alam helped us model the wireless channel realistically and think about priority "
                 "scheduling at the right abstraction layer \u2014 his CN coursework directly shaped "
                 "the min-heap transport design.", "size": 13},
        {"text": ""},
        {"text": "Arnav Ghosh pushed us to formalise the Pareto analysis rather than just picking the "
                 "best-looking configuration \u2014 the 108-configuration grid and non-domination analysis "
                 "are a direct result of that guidance.", "size": 13},
        {"text": ""},
        {"text": "We also thank Cohen et al. for making their COVID-19 chest X-ray dataset open access.",
         "size": 13, "italic": True},
    ])

    # References
    add_textbox(slide, Inches(0.4), Inches(4.42), Inches(12.5), Inches(0.4),
                "Key References", 14, bold=True, color=INDIGO)

    refs = [
        ["[1]", "Cohen et al. (2020)", "COVID-19 Chest X-ray Dataset. GitHub: ieee8023/covid-chestxray-dataset"],
        ["[2]", "Selvaraju et al. (ICCV 2017)", "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"],
        ["[3]", "Kendall & Gal (CVPR 2018)", "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"],
        ["[4]", "NIST SP 800-38D (2007)", "Recommendation for Block Cipher Modes: Galois/Counter Mode (GCM)"],
        ["[5]", "NIST SP 800-232 (2024)", "Recommendation for the Ascon Cryptographic Suite (Ascon-AEAD128)"],
        ["[6]", "Wang et al. (IEEE TIP 2004)", "Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)"],
    ]
    add_table(slide, Inches(0.3), Inches(4.9), Inches(12.7), Inches(2.4),
              [["#", "Authors / Standard", "Title / Description"]] + refs,
              col_widths=[0.5, 3.5, 9.0],
              header_font_size=10, body_font_size=10)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    prs = new_prs()

    print("Building slide 1: Title")
    slide_01_title(prs)
    print("Building slide 2: Problem")
    slide_02_problem(prs)
    print("Building slide 3: Dataset")
    slide_03_dataset(prs)
    print("Building slide 4: Architecture")
    slide_04_architecture(prs)
    print("Building slide 5: Model")
    slide_05_model(prs)
    print("Building slide 6: Loss")
    slide_06_loss(prs)
    print("Building slide 7: Ablation")
    slide_07_ablation(prs)
    print("Building slide 8: Regime B")
    slide_08_regime_b(prs)
    print("Building slide 9: Compression diagram")
    slide_09_compression(prs)
    print("Building slide 10: Compression results")
    slide_10_compression_results(prs)
    print("Building slide 11: Cryptography")
    slide_11_crypto(prs)
    print("Building slide 12: MLMC-128")
    slide_12_mlmc(prs)
    print("Building slide 13: Transport")
    slide_13_transport(prs)
    print("Building slide 14: Pareto")
    slide_14_pareto(prs)
    print("Building slide 15: Results summary")
    slide_15_results(prs)
    print("Building slide 16: Demo")
    slide_16_demo(prs)
    print("Building slide 17: Limitations")
    slide_17_limitations(prs)
    print("Building slide 18: Acknowledgements")
    slide_18_acknowledgements(prs)

    out_path = os.path.join(os.path.dirname(__file__), "MediLink_DL_Progress_Presentation.pptx")
    prs.save(out_path)
    print(f"\nSuccessfully generated PowerPoint presentation")
    print(f"Output: {out_path}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
