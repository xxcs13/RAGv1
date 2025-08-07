"""
LangGraph workflow orchestration for RAG system.
"""
import os
import time
import pandas as pd
from typing import Any, Sequence, Union
from dataclasses import dataclass, field
from langchain.schema import Document
from langgraph.graph import StateGraph, END

from parsing import UnifiedDocumentParser
from chunking import CrossPageTextSplitter
from vectorstore import VectorStoreManager
from retrieval import HybridRetriever, assemble_context
from generation import AnswerGenerator
from utils import calculate_throughput


@dataclass
class GraphState:
    """State management for the RAG workflow graph."""
    docs: Sequence[Union[str, Document]] = field(default_factory=list)
    vectorstore: Any = None
    question: str = ""
    retrieved_docs: list[Document] = field(default_factory=list)
    answer: str = ""
    
    # Enhanced state fields
    parsed_reports: list[dict] = field(default_factory=list)
    vector_results: list[dict] = field(default_factory=list)
    reranked_results: list[dict] = field(default_factory=list)
    final_context: str = ""
    structured_answer: dict = field(default_factory=dict)
    skip_parsing: bool = False
    
    # Performance tracking fields
    start_time: float = 0.0
    end_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    throughput_tokens_per_second: float = 0.0


def ingest_node(state: GraphState) -> GraphState:
    """Parse and ingest documents using unified parsing system."""
    print("Starting document ingestion...")
    
    parsed_reports = []
    parser = UnifiedDocumentParser()
    text_splitter = CrossPageTextSplitter()
    
    if state.docs and isinstance(state.docs[0], str):
        successful_count = 0
        failed_count = 0
        
        for file_path in state.docs:
            if os.path.exists(str(file_path)):
                try:
                    report = parser.parse_document(str(file_path))
                    
                    if report['metainfo']['document_type'] == 'failed':
                        print(f"Skipping failed document: {file_path}")
                        failed_count += 1
                        continue
                    
                    chunks = text_splitter.split_document(report)
                    
                    for chunk in chunks:
                        chunk.metadata.update({
                            "source_file": os.path.basename(str(file_path)),
                            "document_type": report['metainfo'].get('document_type', 'unknown'),
                            "sha1_name": report['metainfo'].get('sha1_name', '')
                        })
                    
                    parsed_reports.append({
                        'file_path': file_path,
                        'report': report,
                        'chunks': chunks
                    })
                    
                    successful_count += 1
                    print(f"Successfully parsed: {file_path} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    print(f"Failed to parse {file_path}: {e}")
                    failed_count += 1
            else:
                print(f"Warning: File not found: {file_path}")
                failed_count += 1
        
        print(f"Parsing summary: {successful_count} successful, {failed_count} failed")
        
        if successful_count == 0:
            print("Error: No documents were successfully parsed. Check file formats and paths.")
            raise RuntimeError("Document parsing failed for all files")
    
    all_chunks = []
    for parsed_report in parsed_reports:
        all_chunks.extend(parsed_report['chunks'])
    
    print(f"Ingested {len(parsed_reports)} documents with {len(all_chunks)} chunks")
    
    return GraphState(
        docs=all_chunks,
        vectorstore=state.vectorstore,
        question=state.question,
        parsed_reports=parsed_reports
    )


def embed_node(state: GraphState) -> GraphState:
    """Create vector embeddings for document chunks."""
    print("Creating vector embeddings...")
    
    if state.vectorstore is None and state.docs:
        vs_manager = VectorStoreManager()
        document_list = [doc for doc in state.docs if isinstance(doc, Document)]
        
        if document_list:
            vectorstore = vs_manager.create_vectorstore(document_list, state.parsed_reports)
            print(f"Created vector store with {len(document_list)} documents")
        else:
            vectorstore = None
    else:
        vectorstore = state.vectorstore
    
    return GraphState(
        docs=state.docs,
        vectorstore=vectorstore,
        question=state.question,
        parsed_reports=state.parsed_reports
    )


def retrieval_node(state: GraphState) -> GraphState:
    """Execute complete retrieval pipeline."""
    print(f"Starting retrieval for question: {state.question[:100]}...")
    
    retrieval_start = time.time()
    
    retriever = HybridRetriever(state.vectorstore, state.parsed_reports)
    
    reranked_results = retriever.retrieve(
        query=state.question,
        llm_reranking_sample_size=30,
        documents_batch_size=2,
        top_n=10,
        llm_weight=0.7
    )
    
    retrieval_time = time.time() - retrieval_start
    print(f"Retrieval completed with {len(reranked_results)} results in {retrieval_time:.3f}s")
    
    final_context = assemble_context(reranked_results)
    
    retrieved_docs = []
    for result in reranked_results:
        doc = Document(
            page_content=result['text'],
            metadata=result['metadata']
        )
        retrieved_docs.append(doc)
    
    print(f"Retrieved {len(reranked_results)} final documents")
    
    return GraphState(
        docs=state.docs,
        vectorstore=state.vectorstore,
        question=state.question,
        retrieved_docs=retrieved_docs,
        parsed_reports=state.parsed_reports,
        vector_results=[],
        reranked_results=reranked_results,
        final_context=final_context,
        start_time=state.start_time,
        end_time=state.end_time,
        retrieval_time=retrieval_time,
        generation_time=state.generation_time,
        total_time=state.total_time,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        throughput_tokens_per_second=state.throughput_tokens_per_second
    )


def rag_node(state: GraphState) -> GraphState:
    """Generate structured answers using enhanced RAG system."""
    print("Generating structured answer...")
    
    generator = AnswerGenerator()
    result = generator.generate_answer(state.question, state.final_context)
    
    print(f"Answer confidence: {result['structured_answer'].get('confidence_level', 'unknown')}")
    print(f"Sources used: {len(result['structured_answer'].get('relevant_sources', []))}")
    print(f"Generation completed in {result['generation_time']:.3f}s")
    print(f"Tokens: {result['input_tokens']} input + {result['output_tokens']} output = {result['total_tokens']} total")
    print(f"Throughput: {result['throughput']} tokens/second")
    
    return GraphState(
        docs=state.docs,
        vectorstore=state.vectorstore,
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=result['final_answer'],
        parsed_reports=state.parsed_reports,
        vector_results=state.vector_results,
        reranked_results=state.reranked_results,
        final_context=state.final_context,
        structured_answer=result['structured_answer'],
        start_time=state.start_time,
        end_time=state.end_time,
        retrieval_time=state.retrieval_time,
        generation_time=result['generation_time'],
        total_time=state.total_time,
        input_tokens=result['input_tokens'],
        output_tokens=result['output_tokens'],
        throughput_tokens_per_second=result['throughput']
    )


def log_node(state: GraphState) -> GraphState:
    """Log detailed metrics and results including performance data."""
    print("Logging enhanced metrics...")
    
    end_time = time.time()
    total_time = end_time - state.start_time if state.start_time > 0 else 0.0
    
    total_tokens = state.input_tokens + state.output_tokens
    precise_throughput = calculate_throughput(total_tokens, total_time)
    
    metrics = {}
    if state.reranked_results:
        scores = [r.get('combined_score', 0) for r in state.reranked_results]
        if scores:
            metrics = {
                "avg_score": round(sum(scores) / len(scores), 4),
                "max_score": round(max(scores), 4),
                "score_range": round(max(scores) - min(scores), 4),
                "final_results": len(state.reranked_results)
            }
    
    performance_metrics = {
        "retrieval_time_s": round(state.retrieval_time, 3),
        "generation_time_s": round(state.generation_time, 3),
        "total_time_s": round(total_time, 3),
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "total_tokens": total_tokens,
        "throughput_tokens_per_second": precise_throughput,
        "start_time": state.start_time,
        "end_time": end_time
    }
    
    log_entry = {
        "question": state.question,
        "answer": state.answer,
        "confidence_level": state.structured_answer.get('confidence_level', 'unknown'),
        "relevant_sources": state.structured_answer.get('relevant_sources', []),
        "reasoning_summary": state.structured_answer.get('reasoning_summary', ''),
        "retrieval_metrics": metrics,
        "performance_metrics": performance_metrics,
        "used_existing_vectordb": False
    }
    
    df = pd.DataFrame([log_entry])
    log_file = "qa_log.csv"
    
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
        print(f"Created new log file: {log_file}")
    else:
        df.to_csv(log_file, mode="a", header=False, index=False)
        print(f"Appended to log file: {log_file}")
    
    print("\n" + "+" * 50)
    print("PERFORMANCE SUMMARY")
    print("+" * 50 + "\n")
    print(f"Retrieval Time: {performance_metrics['retrieval_time_s']}s")
    print(f"Generation Time: {performance_metrics['generation_time_s']}s")
    print(f"Total Processing Time: {performance_metrics['total_time_s']}s")
    print(f"Input Tokens: {performance_metrics['input_tokens']:,}")
    print(f"Output Tokens: {performance_metrics['output_tokens']:,}")
    print(f"Total Tokens: {performance_metrics['total_tokens']:,}")
    print(f"Throughput: {performance_metrics['throughput_tokens_per_second']} tokens/second")
    print("\n" + "+" * 50)
    
    updated_state = GraphState(
        docs=state.docs,
        vectorstore=state.vectorstore,
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=state.answer,
        parsed_reports=state.parsed_reports,
        vector_results=state.vector_results,
        reranked_results=state.reranked_results,
        final_context=state.final_context,
        structured_answer=state.structured_answer,
        start_time=state.start_time,
        end_time=end_time,
        retrieval_time=state.retrieval_time,
        generation_time=state.generation_time,
        total_time=total_time,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        throughput_tokens_per_second=precise_throughput
    )
    
    return updated_state


# Build workflows
def build_init_workflow() -> StateGraph:
    """Build initialization workflow for document processing."""
    init_workflow = StateGraph(GraphState)
    init_workflow.add_node("ingest", ingest_node)
    init_workflow.add_node("embed", embed_node)
    init_workflow.set_entry_point("ingest")
    init_workflow.add_edge("ingest", "embed")
    init_workflow.add_edge("embed", END)
    return init_workflow.compile()


def build_query_workflow() -> StateGraph:
    """Build query workflow for question answering."""
    query_workflow = StateGraph(GraphState)
    query_workflow.add_node("retrieval", retrieval_node)
    query_workflow.add_node("rag", rag_node)
    query_workflow.add_node("log", log_node)
    query_workflow.set_entry_point("retrieval")
    query_workflow.add_edge("retrieval", "rag")
    query_workflow.add_edge("rag", "log")
    query_workflow.add_edge("log", END)
    return query_workflow.compile()
