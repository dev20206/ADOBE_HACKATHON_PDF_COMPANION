# backend/process_pdfs.py
# Solution for Round 1A: Extracts structured outlines from all PDFs in the input directory.

import fitz  # PyMuPDF
import json
import os
import re
from collections import Counter
from pathlib import Path

def clean_text(text):
    """Removes control characters and normalizes whitespace."""
    if not text:
        return ""
    # Remove control characters and multiple spaces
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def analyze_font_styles(doc):
    """
    Analyzes document font styles to find a baseline for heading detection.
    It now returns the most common (body size) and second most common (potential heading) font sizes.
    """
    spans_data = []
    # Analyze a few pages to get a representative sample of font sizes
    pages_to_analyze = min(10, len(doc))
    
    for page_num in range(pages_to_analyze):
        page = doc[page_num]
        try:
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get('type') == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            # Ignore very small text
                            if span.get('size', 0) > 7:
                                spans_data.append(round(span.get('size', 0), 1))
        except Exception as e:
            print(f"Warning: Error analyzing page {page_num}: {e}")
            continue
    
    if not spans_data:
        # Provide a sensible default if no text is found
        return {'body_size': 12.0, 'heading_size': 14.0}
    
    # Get the two most common font sizes
    common_sizes = Counter(spans_data).most_common(2)
    
    # The most common is likely the body text
    body_size = common_sizes[0][0]
    
    # The second most common might be a heading, or just another body font.
    # If there's only one font size, assume headings are slightly larger.
    heading_size = common_sizes[1][0] if len(common_sizes) > 1 else body_size * 1.2
    
    # Ensure heading_size is actually larger than body_size
    if heading_size <= body_size:
        heading_size = body_size * 1.2

    return {'body_size': body_size, 'heading_size': heading_size}

def extract_outline_from_pdf(doc):
    """Extracts the document title and structured outline with improved heading detection."""
    style_stats = analyze_font_styles(doc)
    body_size = style_stats['body_size']
    heading_size = style_stats['heading_size']
    
    outline = []
    
    # --- Title Extraction ---
    doc_title = (hasattr(doc, 'metadata') and doc.metadata.get('title')) or ""
    if not doc_title.strip():
        doc_title = Path(doc.name).stem.replace('_', ' ').replace('-', ' ') if hasattr(doc, 'name') and doc.name else "Untitled Document"
    doc_title = clean_text(doc_title)

    # --- Outline Extraction ---
    for page_num, page in enumerate(doc):
        try:
            blocks = page.get_text("dict").get("blocks", [])
            # Sort blocks by vertical position
            blocks = sorted(blocks, key=lambda b: b.get('bbox', [0,0,0,0])[1])
            
            for block in blocks:
                if block.get('type') == 0:  # Text block
                    for line in block.get("lines", []):
                        # Skip lines with many spans, as they are likely paragraphs
                        if len(line.get("spans", [])) > 5:
                            continue

                        text = "".join(span.get('text', '') for span in line.get("spans", []))
                        text = clean_text(text)
                        
                        # Filter out short or non-alphanumeric lines
                        if not text or len(text) < 4 or len(text) > 250 or not re.search(r'[a-zA-Z]', text):
                            continue

                        first_span = line.get("spans", [{}])[0]
                        font_size = round(first_span.get('size', 0), 1)
                        is_bold = (first_span.get('flags', 0) & 16) > 0
                        
                        level = None
                        # Heuristic 1: Font size is significantly larger than body text
                        if font_size > body_size * 1.35:
                            level = "H1"
                        elif font_size > body_size * 1.15:
                            level = "H2"
                        # Heuristic 2: Font size is slightly larger and bold
                        elif font_size > body_size * 1.05 and is_bold:
                            level = "H3"
                        # Heuristic 3: Font is the determined "heading_size"
                        elif abs(font_size - heading_size) < 0.15:
                             level = "H3"

                        if level:
                            outline.append({
                                "level": level, 
                                "text": text, 
                                "page": page_num + 1
                            })
                            
        except Exception as e:
            print(f"Warning: Error processing page {page_num + 1}: {e}")
            continue
                        
    return doc_title, outline

def process_all_pdfs():
    """Main function to run the batch processing."""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process...")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        try:
            with fitz.open(str(pdf_path)) as doc:
                doc_title, outline = extract_outline_from_pdf(doc)
            
            output_data = {"title": doc_title, "outline": outline}
            
            output_filename = output_dir / f"{pdf_path.stem}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Success -> {output_filename} ({len(outline)} headings found)")
            
        except Exception as e:
            print(f"  ✗ ERROR: Failed to process {pdf_path.name}. Reason: {e}")

if __name__ == "__main__":
    process_all_pdfs()
