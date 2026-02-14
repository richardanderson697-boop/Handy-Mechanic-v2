# Enterprise-Grade Automotive Diagnostic RAG System

## Overview
This repository hosts the architectural specifications and implementation guides for a safety-critical Automotive Diagnostic System. The system utilizes Retrieval-Augmented Generation (RAG) to provide accurate vehicle diagnostics by combining user input (text, audio, photos) with a curated automotive knowledge base (TSBs, Repair Manuals).

## Architecture
The system is designed with a specific focus on safety and accuracy, featuring:

1.  **Audio Processing Pipeline**: Noise reduction and feature extraction (MFCC) to identify mechanical failure sounds.
2.  **Multi-Modal Embedding**: Combines text descriptions and audio features into a shared semantic space.
3.  **Vector Database**: High-performance retrieval using Pinecone/Weaviate.
4.  **Safety Validation Layer**: A distinct layer to assess risk and confidence before presenting diagnoses to the user.

## Repository Structure

- `docs/requirements/`: Safety constraints and high-level product goals.
- `docs/architecture/`: Technical implementation details, database schemas, and code snippets.
- `docs/meta/`: Internal project metadata.

## Technology Stack

- **Runtime**: Node.js
- **AI/ML**: OpenAI (ada-002), Custom Audio Classifiers (CNN)
- **Database**: Pinecone (Vector), MongoDB (Metadata)
- **Processing**: Librosa (Audio analysis context)

## Getting Started

Please review `docs/architecture/technical_implementation.md` for the database setup scripts and knowledge base structure definition.