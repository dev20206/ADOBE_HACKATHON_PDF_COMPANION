# backend/api.py
# Final API Server with endpoints for Round 1A, 1B, and serving PDF files.

from flask import Flask, jsonify, abort, request, send_from_directory
from flask_cors import CORS
import os
import json
from sentence_transformers import SentenceTransformer
from round_1b_insights import run_persona_extraction

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

# --- API Routes ---

@app.route('/api/files', methods=['GET'])
def list_processed_files():
    """Lists all available JSON outline files."""
    if not os.path.exists(OUTPUT_DIR):
        return jsonify({"error": "Output directory not found. Please run the Round 1A processing first."}), 404
    
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
        file_stems = sorted([os.path.splitext(f)[0] for f in files])
        return jsonify(file_stems)
    except Exception as e:
        return jsonify({"error": f"Failed to list files: {str(e)}"}), 500

@app.route('/api/outline/<string:filename_stem>', methods=['GET'])
def get_outline(filename_stem):
    """Serves the content of a specific JSON outline file."""
    file_path = os.path.join(OUTPUT_DIR, f"{filename_stem}.json")
    
    if not os.path.exists(file_path):
        abort(404, description=f"Outline for '{filename_stem}' not found.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        abort(500, description=f"Failed to read outline file. Error: {e}")

@app.route('/api/pdf/<string:filename_stem>', methods=['GET'])
def get_pdf(filename_stem):
    """Serves a specific PDF file from the input directory."""
    try:
        pdf_filename = f"{filename_stem}.pdf"
        if not os.path.exists(os.path.join(INPUT_DIR, pdf_filename)):
            abort(404, description=f"PDF file '{pdf_filename}' not found.")
        return send_from_directory(INPUT_DIR, pdf_filename, as_attachment=False)
    except Exception as e:
        abort(500, description=f"Failed to serve PDF: {str(e)}")

@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Generates and returns persona-driven insights."""
    persona = request.args.get('persona', 'student')
    
    if not model:
        abort(500, "Sentence transformer model is not loaded.")
    
    try:
        # Get all PDF files from input directory
        if not os.path.exists(INPUT_DIR):
            return jsonify({"error": "Input directory not found."}), 404
            
        pdf_files = [
            os.path.join(INPUT_DIR, f) 
            for f in os.listdir(INPUT_DIR) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            return jsonify({"error": "No PDF files found in the input directory."}), 404
        
        print(f"Processing {len(pdf_files)} PDF files for persona: {persona}")
        insights_data = run_persona_extraction(pdf_files, persona, model, OUTPUT_DIR)
        return jsonify(insights_data)
        
    except Exception as e:
        print(f"Error during insight extraction: {e}")
        import traceback
        traceback.print_exc()
        abort(500, description=str(e))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "input_dir_exists": os.path.exists(INPUT_DIR),
        "output_dir_exists": os.path.exists(OUTPUT_DIR)
    })

if __name__ == '__main__':
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model path: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5001, debug=True)
