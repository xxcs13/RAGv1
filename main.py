#!/usr/bin/env python3
"""
Enhanced RAG System - Main Entry Point

A comprehensive Retrieval-Augmented Generation system supporting:
- Multi-format document parsing (PDF, PPTX, Excel)
- Advanced chunking with cross-page awareness
- Vector database persistence with ChromaDB
- Hybrid retrieval with LLM reranking
- Structured answer generation
- Performance monitoring and logging
"""

import sys
import time
import os
from vectorstore import VectorStoreManager
from workflow import GraphState, build_init_workflow, build_query_workflow
from utils import get_user_files


def main():
    """Main execution function for the RAG system."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        interactive_mode = False
        print(f"Command line mode - Question: {question}")
    else:
        # Interactive mode
        question = ""
        interactive_mode = True
        print("Interactive mode")
    
    print("RAG system initialization...")
    
    # Build workflows
    init_graph = build_init_workflow()
    query_graph = build_query_workflow()
    
    vs_manager = VectorStoreManager()
    
    # Initialize variables for consistency
    docs = []
    parsed_reports = []
    vectorstore = None
    
    if vs_manager.vectorstore_exists():
        try:
            # Load both vectorstore and associated document metadata
            vectorstore, parsed_reports = vs_manager.load_existing_vectorstore()
            
            if vectorstore is not None and parsed_reports:
                stats = vs_manager.get_vectorstore_stats(vectorstore)
                print(f"Found existing vector database: {stats}")
                print(f"Loaded document metadata for {len(parsed_reports)} documents")
                # Set docs to empty since we're using existing vectorstore
                docs = []
            else:
                print("Failed to load existing database or metadata - creating new database")
                vectorstore = None
                parsed_reports = []
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Creating new database from documents...")
            vectorstore = None
            parsed_reports = []
    else:
        print("No existing vector database found.")
        vectorstore = None
    
    # If no existing vectorstore or failed to load, create new one
    if vectorstore is None:
        print("Please provide documents for processing:")
        user_files = get_user_files()
        
        if not user_files:
            print("No files provided. Exiting.")
            exit(1)
        
        print(f"Processing {len(user_files)} documents...")
        
        initial_state = GraphState(
            docs=user_files,
            question="",
            skip_parsing=False
        )
        
        try:
            state_after_embed = init_graph.invoke(initial_state)
            print("Documents processed successfully!")
            
            docs = state_after_embed['docs']
            vectorstore = state_after_embed['vectorstore']
            parsed_reports = state_after_embed['parsed_reports']
            
            # Save the metadata for future use
            if parsed_reports and vectorstore:
                vs_manager.save_document_metadata(parsed_reports)
            
            if parsed_reports:
                total_pages = sum(len(report['report']['content']['pages']) for report in parsed_reports)
                total_chunks = len(docs)
                print(f"Documents: {len(parsed_reports)}, Pages: {total_pages}, Chunks: {total_chunks}")
                
                # If no chunks were created, there's likely an issue with the document parsing
                if total_chunks == 0:
                    print("Warning: No text chunks were generated from the documents.")
                    print("This could be due to:")
                    print("  - Empty documents")
                    print("  - Text extraction issues") 
                    print("  - Chunking configuration problems")
                    
                    # Check if we have any content in the parsed reports
                    for report in parsed_reports:
                        file_path = report['file_path']
                        pages = report['report']['content']['pages']
                        print(f"\nDebug info for {file_path}:")
                        for page in pages[:2]:  # Show first 2 pages
                            text = page.get('text', '')
                            print(f"  Page {page['page']}: {len(text)} chars")
                            if text:
                                print(f"    Preview: {text[:100]}...")
                    
                    print("\nCannot proceed without text chunks. Please check your documents.")
                    exit(1)
            
        except Exception as e:
            print(f"Error processing documents: {e}")
            print("Check: dependencies, file formats, file paths")
            exit(1)
    
    # Ensure we have both vectorstore and parsed_reports before proceeding
    if not vectorstore:
        print("Error: No vectorstore available")
        exit(1)
    
    if not parsed_reports:
        print("Error: No document metadata available")
        exit(1)
    
    if interactive_mode:
        print("==== Ready for questions. Type 'quit' to exit. ====")
        
        while True:
            question = input("\nQuestion: ")
            
            if question.strip().lower() == "quit":
                print("Exiting...")
                break
            
            start_time = time.time()
            
            state = GraphState(
                docs=docs,
                vectorstore=vectorstore,
                question=question,
                parsed_reports=parsed_reports,
                start_time=start_time
            )
            
            result = query_graph.invoke(state)
            
            print("\n" + "*" * 50)
            print("RESULTS")
            print("*" * 50 + "\n")
            print(f"Confidence: {result['structured_answer'].get('confidence_level', 'unknown')}")
            print(f"Documents: {len(result['reranked_results'])}")
            
            if result['reranked_results']:
                scores = [r.get('combined_score', 0) for r in result['reranked_results']]
                if scores:
                    print(f"Score Range: {min(scores):.3f} - {max(scores):.3f}")
            
            print("\n--- Reasoning ---")
            print(result['structured_answer'].get('reasoning_summary', 'No summary'))
            
            print("\n--- Analysis ---")
            analysis = result['structured_answer'].get('step_by_step_analysis', 'No analysis')
            print(analysis[:400] + "..." if len(analysis) > 400 else analysis)
            
            print("\n--- Sources ---")
            sources = result['structured_answer'].get('relevant_sources', [])
            if sources:
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            else:
                print("  No sources identified")
            
            print("\n" + "=" * 50)
            print("ANSWER")
            print("=" * 50)
            print("\n" + result['answer'])
            print("\n" + "=" * 50)
    else:
        # Single question mode
        if not question.strip():
            print("Error: No question provided")
            exit(1)
        
        start_time = time.time()
        
        state = GraphState(
            docs=docs,
            vectorstore=vectorstore,
            question=question,
            parsed_reports=parsed_reports,
            start_time=start_time
        )
        
        result = query_graph.invoke(state)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Confidence: {result['structured_answer'].get('confidence_level', 'unknown')}")
        print(f"Documents: {len(result['reranked_results'])}")
        
        if result['reranked_results']:
            scores = [r.get('combined_score', 0) for r in result['reranked_results']]
            if scores:
                print(f"Score Range: {min(scores):.3f} - {max(scores):.3f}")
        
        print("\n--- Reasoning ---")
        print(result['structured_answer'].get('reasoning_summary', 'No summary'))
        
        print("\n--- Analysis ---")
        analysis = result['structured_answer'].get('step_by_step_analysis', 'No analysis')
        print(analysis[:400] + "..." if len(analysis) > 400 else analysis)
        
        print("\n--- Sources ---")
        sources = result['structured_answer'].get('relevant_sources', [])
        if sources:
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")
        else:
            print("  No sources identified")
        
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result['answer'])
        print("=" * 60)


if __name__ == "__main__":
    main()
