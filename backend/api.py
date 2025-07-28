# backend/api.py
# Final API Server with all logic self-contained for reliability.
# Includes improved outline extraction and integrated, more robust insights generation.

from flask import Flask, jsonify, abort, request, send_from_directory
from flask_cors import CORS
import os
import json
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import re
from collections import Counter
from pathlib import Path
import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, '..', 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
MODEL_PATH = os.path.join(BASE_DIR, 'all-MiniLM-L6-v2')

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Load Model at Startup ---
print("Loading sentence-transformer model...")
try:
    if os.path.exists(MODEL_PATH):
        model = SentenceTransformer(MODEL_PATH)
        print("✓ Model loaded successfully.")
    else:
        print(f"Model path not found: {MODEL_PATH}")
        print("Trying to load model from HuggingFace...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded from HuggingFace.")
except Exception as e:
    model = None
    print(f"⚠️ CRITICAL WARNING: Failed to load model. Error: {e}")

# --- Persona and Insights Logic (Integrated) ---
PERSONA_QUERIES = {
    "phd_researcher": {"job": "Literature Review & Methodology Analysis", "queries": ["research methodology and experimental design", "datasets and benchmarks used for evaluation", "primary results and conclusions of the study", "limitations of the work and future research directions"]},
    "investment_analyst": {"job": "Financial & Market Analysis", "queries": ["financial performance revenue trends and profit margins", "R&D spending and technology strategy", "market size competitive landscape and growth projections"]},
    "student": {"job": "Exam Prep & Key Concepts", "queries": ["key concepts and definitions", "summary of the main topic", "important formulas theories and principles", "illustrative examples and case studies"]},
    "journalist": {"job": "Fact-Finding & Quoting Sources", "queries": ["direct quotes and statements from individuals", "timeline of key events", "official statements from the organization", "data points and statistics"]},
    "salesperson": {"job": "Product & Market Fit", "queries": ["product features and customer benefits", "target audience and customer profile", "comparison with competitor products", "pricing cost and value proposition"]},
    "lawyer": {"job": "Legal & Compliance Review", "queries": ["legal clauses terms and conditions", "liability risk and indemnity sections", "regulatory compliance and legal agreements", "definitions of legal terms"]}
}

def chunk_pdf_text_for_insights(doc):
    """Chunks text into meaningful paragraphs for insight analysis."""
    chunks = []
    for page_num, page in enumerate(doc):
        try:
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5 and block[6] == 0:
                    text = clean_text(block[4])
                    word_count = len(text.split())
                    if 10 <= word_count <= 800:
                        chunks.append({"text": text, "page": page_num + 1})
        except Exception as e:
            print(f"Warning: Error chunking page {page_num + 1}: {e}")
            continue
    return chunks

def find_section_title_for_insights(chunk, outline_data):
    """Finds the most relevant heading for a given text chunk."""
    if not outline_data or 'outline' not in outline_data:
        return outline_data.get("title", "Introduction") if outline_data else "Introduction"
    relevant_headings = [h for h in outline_data["outline"] if h["page"] <= chunk["page"]]
    return relevant_headings[-1]["text"] if relevant_headings else outline_data.get("title", "Introduction")


# --- Outline Processing Logic ---
def clean_text(text):
    """Universal text cleaner."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def analyze_font_styles(doc):
    """Analyzes document font styles to find a baseline for heading detection."""
    spans_data = []
    pages_to_analyze = min(10, len(doc))
    for page_num in range(pages_to_analyze):
        try:
            page = doc[page_num]  # Correctly define the page object
            for block in page.get_text("dict").get("blocks", []):
                if block.get('type') == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get('size', 0) > 7:
                                spans_data.append({'size': round(span.get('size', 0), 1), 'font': span.get('font', 'Unknown')})
        except Exception as e:
            print(f"Warning: Error analyzing page styles {page_num}: {e}")
    if not spans_data: return {'body_size': 12.0, 'body_font': 'Unknown'}
    body_size = Counter(s['size'] for s in spans_data).most_common(1)[0][0]
    body_font = Counter(s['font'] for s in spans_data if s['size'] == body_size).most_common(1)[0][0]
    return {'body_size': body_size, 'body_font': body_font}

def extract_outline_from_pdf(doc):
    """Extracts a structured outline from a PDF, with balanced logic for academic papers."""
    style_stats = analyze_font_styles(doc)
    body_size, body_font = style_stats['body_size'], style_stats['body_font']
    outline = []
    doc_title = (hasattr(doc, 'metadata') and doc.metadata.get('title')) or ""
    if not doc_title.strip():
        doc_title = Path(doc.name).stem.replace('_', ' ').replace('-', ' ') if hasattr(doc, 'name') and doc.name else "Untitled Document"
    doc_title = clean_text(doc_title)

    for page_num, page in enumerate(doc):
        try:
            blocks = sorted(page.get_text("dict").get("blocks", []), key=lambda b: b.get('bbox', [0,0,0,0])[1])
            for block in blocks:
                if block.get('type') == 0:
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans: continue
                        line_bbox = line.get('bbox', (0,0,0,0))
                        if line_bbox[1] < page.rect.height * 0.08 or line_bbox[3] > page.rect.height * 0.92: continue
                        text = clean_text("".join(s.get('text', '') for s in spans))
                        if not text or len(text) < 3 or len(text) > 250 or not re.search(r'[a-zA-Z]', text) or text.endswith(('.', ',', ';', ':')) or len(spans) > 5: continue
                        
                        first_span = spans[0]
                        font_size = round(first_span.get('size', 0), 1)
                        font_name = first_span.get('font', 'Unknown')
                        is_bold = (first_span.get('flags', 0) & 16) > 0
                        is_all_caps = text.isupper() and len(text.split()) < 7 and len(text) > 3
                        is_numbered = re.match(r'^(?:[IVXLCDM]+\.|[A-Z]\.|\d+\.)(?:\d+\.)*\s+', text) is not None
                        level = None
                        if font_size > body_size * 1.35 and is_bold: level = "H1"
                        elif is_numbered and font_size >= body_size and is_bold: level = "H2"
                        elif font_size > body_size * 1.15 and is_bold: level = "H2"
                        elif font_size > body_size * 1.05 and is_bold: level = "H3"
                        elif font_size == body_size and is_bold and is_all_caps: level = "H3"
                        elif font_size == body_size and is_bold and body_font not in font_name: level = "H3"
                        if level and (not outline or outline[-1]['text'] != text):
                            outline.append({"level": level, "text": text, "page": page_num + 1})
        except Exception as e:
            print(f"Warning: Error processing page {page_num + 1}: {e}")
    return doc_title, outline

# --- API Routes ---
@app.route('/api/upload', methods=['POST'])
def upload_and_process_file():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(INPUT_DIR, filename)
        file.save(pdf_path)
        try:
            with fitz.open(pdf_path) as doc:
                doc_title, outline = extract_outline_from_pdf(doc)
            file_stem = os.path.splitext(filename)[0]
            output_filename = os.path.join(OUTPUT_DIR, f"{file_stem}.json")
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump({"title": doc_title, "outline": outline}, f, indent=2, ensure_ascii=False)
            return jsonify({"message": "File processed successfully", "file_stem": file_stem}), 200
        except Exception as e:
            if os.path.exists(pdf_path): os.remove(pdf_path)
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/files', methods=['GET'])
def list_processed_files():
    if not os.path.exists(OUTPUT_DIR): return jsonify({"error": "Output directory not found"}), 404
    try:
        files = [os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
        return jsonify(sorted(files))
    except Exception as e:
        return jsonify({"error": f"Failed to list files: {str(e)}"}), 500

@app.route('/api/outline/<string:filename_stem>', methods=['GET'])
def get_outline(filename_stem):
    file_path = os.path.join(OUTPUT_DIR, f"{filename_stem}.json")
    if not os.path.exists(file_path): abort(404, description=f"Outline for '{filename_stem}' not found.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        abort(500, description=f"Failed to read outline file: {e}")

@app.route('/api/pdf/<string:filename_stem>', methods=['GET'])
def get_pdf(filename_stem):
    pdf_filename = f"{filename_stem}.pdf"
    if not os.path.exists(os.path.join(INPUT_DIR, pdf_filename)):
        abort(404, description=f"PDF file '{pdf_filename}' not found.")
    return send_from_directory(INPUT_DIR, pdf_filename, as_attachment=False)

@app.route('/api/insights', methods=['GET'])
def get_insights():
    persona = request.args.get('persona', 'student')
    file_stem = request.args.get('file_stem')
    if not model: abort(500, "Sentence transformer model is not loaded.")
    if not file_stem: return jsonify({"error": "A file_stem parameter is required."}), 400
    
    try:
        pdf_path = os.path.join(INPUT_DIR, f"{file_stem}.pdf")
        if not os.path.exists(pdf_path): return jsonify({"error": f"File '{file_stem}.pdf' not found."}), 404
        
        persona_info = PERSONA_QUERIES[persona]
        query_embeddings = model.encode(persona_info["queries"], show_progress_bar=False)
        
        all_chunks = []
        with fitz.open(pdf_path) as doc:
            pdf_chunks = chunk_pdf_text_for_insights(doc)
            if pdf_chunks:
                chunk_texts = [c['text'] for c in pdf_chunks]
                chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)
                similarities = cosine_similarity(chunk_embeddings, query_embeddings)
                for i, chunk in enumerate(pdf_chunks):
                    all_chunks.append({**chunk, "relevance": float(np.max(similarities[i]))})
        
        if not all_chunks:
            return jsonify({"metadata": {"persona": persona, "job": persona_info["job"], "documents_processed": [f"{file_stem}.pdf"]}, "extracted_sections": []})
        
        all_chunks.sort(key=lambda x: x['relevance'], reverse=True)
        top_chunks = all_chunks[:15]
        
        outline_path = os.path.join(OUTPUT_DIR, f"{file_stem}.json")
        outline_data = {}
        if os.path.exists(outline_path):
            with open(outline_path, 'r', encoding='utf-8') as f: outline_data = json.load(f)
        
        extracted_sections = []
        for i, chunk in enumerate(top_chunks):
            title = find_section_title_for_insights(chunk, outline_data)
            refined_text = chunk['text']
            if len(refined_text) > 400: refined_text = refined_text[:400] + "..."
            extracted_sections.append({"document": f"{file_stem}.pdf", "page": chunk['page'], "title": title, "importance_rank": i + 1, "subsection_analysis": {"refined_text": refined_text, "page_number": chunk['page']}})
        
        return jsonify({"metadata": {"documents_processed": [f"{file_stem}.pdf"], "persona": persona, "job": persona_info["job"], "timestamp": datetime.datetime.now().isoformat()}, "extracted_sections": extracted_sections})
        
    except Exception as e:
        print(f"Error during insight extraction for {file_stem}: {e}")
        import traceback
        traceback.print_exc()
        abort(500, description=str(e))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None, "input_dir_exists": os.path.exists(INPUT_DIR), "output_dir_exists": os.path.exists(OUTPUT_DIR)})

if __name__ == '__main__':
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model path: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5001, debug=True)
