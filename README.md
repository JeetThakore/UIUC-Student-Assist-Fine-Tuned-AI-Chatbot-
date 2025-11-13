# UIUC Student Assist — Fine-Tuned AI Chatbot for Campus Information

A fine-tuned **FLAN-T5 chatbot** that helps Illinois students get instant, context-aware answers about campus life : powered by **LangChain**, **FAISS**, and **Streamlit**.

---


<img width="1490" height="806" alt="image" src="https://github.com/user-attachments/assets/9a231025-35b4-4baa-8a2d-3ac0326b18b8" />

## Overview

**UIUC Student Assist** is a fine-tuned **FLAN-T5 model** trained on 1,500+ UIUC PDFs to provide precise, domain-specific answers. The system integrates **LangChain**, **FAISS**, and **Streamlit** to form a complete retrieval-augmented generation (RAG) pipeline. It enables students to access accurate information instantly, reducing query resolution time by 70%.

**Tech Stack:** Python, LangChain, FAISS, HuggingFace Transformers, Streamlit

---

## Why I Built This

* To simplify campus life and make information access effortless.
* To provide instant guidance on academics, housing, and university resources.
* To reduce hesitation and anxiety around asking for help.
* To contribute meaningfully to the UIUC community through applied AI.
* To help every student feel supported and informed throughout their journey.

---

## Project Architecture

| Phase                                            | Description                                                                                        | Tools                            |
| :----------------------------------------------- | :------------------------------------------------------------------------------------------------- | :------------------------------- |
| **Phase 1 — Document Ingestion & Vectorization** | Loads PDFs, splits text, creates embeddings, stores vectors in FAISS.                              | LangChain, HuggingFaceEmbeddings |
| **Phase 2 — LLM Fine-Tuning**                    | Fine-tunes `google/flan-t5-base` on UIUC document chunks for better context and response accuracy. | Transformers, Datasets, Trainer  |
| **Phase 3 — Streamlit Q&A App**                  | Builds a student-facing web app connected to the FAISS database and fine-tuned model.              | Streamlit, LangChain, FAISS      |

---

## Key Features

* Context-aware answers trained on 1,500+ UIUC PDFs
* Fine-tuned FLAN-T5 for domain-specific adaptation
* FAISS vector database for high-speed semantic retrieval
* Streamlit-based interactive web app for real-time Q&A
* Modular and extensible for other universities or organizations

---

## Getting Started

1. **Install dependencies** — All scripts include automatic package checks.
2. **Ingest documents (Phase 1)** — Add PDFs to the data folder and run Phase1.
3. **Fine-tune model (Phase 2)** — Execute fine-tuning for quick domain adaptation of FLAN-T5.
4. **Launch app (Phase 3)** — Run Streamlit and start querying your assistant.


---

## Credits

Developed for academic research and campus productivity by **Jeet Thakore**.

This repository provides a complete, production-ready implementation for building a **domain-specific retrieval-augmented chatbot** using fine-tuned LLMs and vector search.
