# backend/round_1b_insights.py
# Solution for Round 1B: Generates persona-driven insights.
# Includes relevance threshold to improve quality.

import fitz
import json
import os
import re
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PERSONA_QUERIES = {
    "phd_researcher": {
        "job": "Literature Review & Methodology Analysis",
        "queries": [
            "research methodology and experimental design", 
            "datasets and benchmarks used for evaluation", 
            "primary results and conclusions of the study", 
            "limitations of the work and future research directions"
        ]
    },
    "investment_analyst": {
        "job": "Financial & Market Analysis",
        "queries": [
            "financial performance revenue trends and profit margins", 
            "R&D spending and technology strategy", 
            "market size competitive landscape and growth projections"
        ]
    },
    "student": {
        "job": "Exam Prep & Key Concepts",
        "queries": [
            "key concepts and definitions", 
            "summary of the main topic", 
            "important formulas theories and principles", 
            "illustrative examples and case studies"
        ]
    },
    "journalist": {
        "job": "Fact-Finding & Quoting Sources",
        "queries": [
            "direct quotes and statements from individuals", 
            "timeline of key events", 
            "official statements from the organization", 
            "data points and statistics"
        ]
    },
    "salesperson": {
        "job": "Product & Market Fit",
        "queries": [
            "product features and customer benefits", 
            "target audience and customer profile", 
            "comparison with competitor products", 
            "pricing cost and value proposition"
        ]
    },
    "lawyer": {
        "job": "Legal & Compliance Review",
        "queries": [
            "legal clauses terms and conditions", 
            "liability risk and indemnity sections", 
            "regulatory compliance and legal agreements", 
            "definitions of legal terms"
        ]
    }
}

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_pdf_text(doc):
    """Chunks text into meaningful paragraphs with relaxed word count."""
    chunks = []
    
    for page_num, page in enumerate(doc):
        try:
            # Get text blocks
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if len(block) >= 5 and block[6] == 0:  # Text block
                    text = clean_text(block[4])
                    
                    # Relaxed filter for word count to capture more potential insights
                    word_count = len(text.split())
                    if 10 <= word_count <= 800:  # More flexible paragraph size
                        chunks.append({
                            "text": text, 
                            "page": page_num + 1
                        })
                        
        except Exception as e:
            print(f"Warning: Error processing page {page_num + 1} in document: {e}")
            continue
    
    return chunks

def find_section_title(chunk, outline_data):
    """Finds the most relevant heading for a given text chunk."""
    if not outline_data or 'outline' not in outline_data:
        return outline_data.get("title", "Introduction") if outline_data else "Introduction"
    
    # Find headings that appear on or before the chunk's page
    relevant_headings = [
        h for h in outline_data["outline"] 
        if h["page"] <= chunk["page"]
    ]
    
    if relevant_headings:
        # Return the last (most recent) heading before or on this page
        return relevant_headings[-1]["text"]
    else:
        return outline_data.get("title", "Introduction")

def run_persona_extraction(pdf_paths, persona, model, outline_dir):
    """Main logic for persona-based extraction with a relevance threshold."""
    RELEVANCE_THRESHOLD = 0.35 # Minimum similarity score to be considered an insight

    if persona not in PERSONA_QUERIES:
        raise ValueError(f"Persona '{persona}' is not defined. Available personas: {list(PERSONA_QUERIES.keys())}")

    persona_info = PERSONA_QUERIES[persona]
    
    print(f"Encoding queries for persona: {persona}")
    try:
        query_embeddings = model.encode(persona_info["queries"], show_progress_bar=False)
    except Exception as e:
        raise Exception(f"Failed to encode queries: {e}")

    all_chunks = []
    processed_docs = []
    
    for pdf_path in pdf_paths:
        try:
            print(f"Processing PDF: {os.path.basename(pdf_path)}")
            
            with fitz.open(pdf_path) as doc:
                pdf_chunks = chunk_pdf_text(doc)
                
                if not pdf_chunks:
                    print(f"  Warning: No chunks extracted from {os.path.basename(pdf_path)}")
                    continue
                
                print(f"  Extracted {len(pdf_chunks)} chunks for analysis")
                
                # Encode chunk texts
                chunk_texts = [c['text'] for c in pdf_chunks]
                try:
                    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)
                except Exception as e:
                    print(f"  Error encoding chunks: {e}")
                    continue
                
                # Calculate similarities
                similarities = cosine_similarity(chunk_embeddings, query_embeddings)
                
                # Add chunks that meet the relevance threshold
                for i, chunk in enumerate(pdf_chunks):
                    max_similarity = float(np.max(similarities[i]))
                    if max_similarity >= RELEVANCE_THRESHOLD:
                        all_chunks.append({
                            "text": chunk['text'], 
                            "page": chunk['page'],
                            "doc_path": pdf_path, 
                            "relevance": max_similarity
                        })
                
                processed_docs.append(os.path.basename(pdf_path))
                
        except Exception as e:
            print(f"Warning: Could not process {pdf_path} for insights. Error: {e}")
            continue

    if not all_chunks:
        print("No chunks met the relevance threshold.")
        return {
            "metadata": {
                "documents_processed": processed_docs,
                "persona": persona, 
                "job": persona_info["job"],
                "timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": []
        }

    # Sort by relevance (highest first)
    all_chunks.sort(key=lambda x: x['relevance'], reverse=True)

    # Extract top sections from the relevant chunks
    extracted_sections = []
    top_chunks = all_chunks[:15]  # Return top 15 sections
    print(f"Found {len(top_chunks)} relevant sections above threshold.")
    
    for i, chunk in enumerate(top_chunks):
        doc_basename = os.path.basename(chunk['doc_path'])
        doc_stem = os.path.splitext(doc_basename)[0]
        
        # Try to find section title from outline
        title = "Introduction"
        outline_path = os.path.join(outline_dir, f"{doc_stem}.json")
        
        if os.path.exists(outline_path):
            try:
                with open(outline_path, 'r', encoding='utf-8') as f:
                    outline_data = json.load(f)
                title = find_section_title(chunk, outline_data)
            except Exception as e:
                print(f"Warning: Could not read outline for {doc_stem}: {e}")
        
        # Truncate text for refined_text preview
        refined_text = chunk['text']
        if len(refined_text) > 400:
            refined_text = refined_text[:400] + "..."
        
        extracted_sections.append({
            "document": doc_basename, 
            "page": chunk['page'], 
            "title": title,
            "importance_rank": i + 1,
            "subsection_analysis": {
                "refined_text": refined_text,
                "page_number": chunk['page']
            }
        })
        
    return {
        "metadata": {
            "documents_processed": processed_docs,
            "persona": persona, 
            "job": persona_info["job"],
            "timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections
    }
