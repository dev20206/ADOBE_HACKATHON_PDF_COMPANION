# Adobe India Hackathon 2025 - Challenge 1A Solution

## Overview

This repository contains a complete and compliant solution for Challenge 1A. The goal is to process all PDF files from a given input directory, extract a structured outline (Title, H1, H2, H3), and output the results as individual JSON files.

This solution is built to be performant, robust, and adhere strictly to all hackathon constraints, including offline execution, resource limits, and a specified Docker environment.

## Core Approach

Instead of relying on a large, external Machine Learning model, this solution uses a lightweight, intelligent, and fast hybrid approach:

1.  **High-Speed Parsing:** It leverages the `PyMuPDF` library, which is known for its exceptional performance in parsing PDF documents and extracting rich metadata (text, font size, font name, boldness, position) without high memory overhead.
2.  **Statistical Style Analysis:** Before processing, the script performs a quick statistical analysis of the entire document's font styles. It identifies the most common "body text" font size. This provides a dynamic baseline, making the heading detection robust across different documents that use different style templates.
3.  **Rule-Based Classification:** Headings are identified using a set of rules based on the extracted metadata relative to the document's baseline style. A text block is classified as a heading if it meets criteria such as:
    * **Font Size:** Significantly larger than the body text.
    * **Font Weight:** Is bold.
    * **Brevity:** Is a short line of text (unlikely to be a full paragraph).
    * **Position:** Its indentation level can imply structure.

This approach is language-agnostic, fast, and requires no model files, easily satisfying the **≤ 200MB model size** and **≤ 10-second execution time** constraints.

## How It Meets the Constraints

* **Execution Time (≤ 10s):** `PyMuPDF` is extremely fast. The statistical analysis and rule-based classification are computationally inexpensive, ensuring processing of a 50-page PDF is completed well within the time limit.
* **Model Size (≤ 200MB):** No ML model is used, so the model footprint is 0MB.
* **Network (None):** The Dockerfile installs all dependencies during the `docker build` phase. The container runs with `--network none` without any issues.
* **Runtime (CPU/AMD64):** The solution is pure Python and runs efficiently on a standard CPU. The Dockerfile specifies the `linux/amd64` platform.
* **I/O Directories:** The script is hardcoded to read from `/app/input` and write to `/app/output` as required.
* **Open Source:** All libraries used (`PyMuPDF`, `numpy`) are open source.