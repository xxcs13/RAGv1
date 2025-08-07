# RAG System

A comprehensive Retrieval-Augmented Generation system built with `LangChain` and `LangGraph` frameworks, supporting multi-format document parsing, advanced chunking strategies, hybrid retrieval mechanisms, and structured answer generation with comprehensive logging and performance monitoring.

## System Overview

This RAG system provides a complete document processing and question-answering pipeline that can handle PDF, PPTX, and Excel files. The system utilizes `ChromaDB` for vector storage, `OpenAI GPT-4.1-mini` for language generation, `text-embedding-3-small` for embeddings, and implements sophisticated text chunking with cross-page awareness. The workflow is orchestrated using `LangGraph` state machines for reliable and scalable processing.

## Workflow Architecture

To be edited....

### Processing Workflow

1. **Document Ingestion**: Multi-format document parsing with layout detection and content extraction
2. **Text Chunking**: Advanced chunking with cross-page awareness and parent-child relationships
3. **Vector Embedding**: Batch processing with token limit management using `text-embedding-3-small`
4. **Storage**: Persistent vector database with metadata preservation using `ChromaDB`
5. **Retrieval**: Hybrid search combining vector similarity and keyword matching
6. **Reranking**: LLM-based relevance scoring for optimal context selection
7. **Generation**: Structured answer generation with confidence scoring and source attribution

## Installation and Setup

### Environment Setup

```bash
# Create conda environment
conda create -n rag python=3.10
conda activate rag

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ENABLE_TELEMETRY=false  # Optional: disable telemetry for privacy
```

### Dependencies

Key libraries include:
- `langchain` and `langchain-openai` for LLM integration
- `langgraph` for workflow orchestration
- `chromadb` for vector database
- `openai` GPT-4.1-mini for language generation
- `text-embedding-3-small` for document embeddings
- `pdfplumber` and `pypdf` for PDF processing
- `python-pptx` for PowerPoint processing
- `pandas` for Excel processing
- `tiktoken` for token counting

## Usage

### Command Line Mode

```bash
# Single question processing
python main.py "What are the key financial metrics mentioned in the documents?"
```

### Interactive Mode

```bash
# Start interactive session
python main.py

# Follow prompts to add documents and ask questions
# Type 'quit' to exit
```

### Document Processing

The system supports multiple document formats:
- **PDF**: Advanced layout detection with multi-column support
- **PPTX**: Complete content extraction including tables, charts, and images
- **Excel**: Multi-sheet processing with data preservation

## Core Modules

### Configuration Management (`config.py`)
- Environment variable handling
- Model configuration and constants
- System-wide settings management

### Document Parsing (`parsing.py`)
- `PDFParser`: Advanced PDF text extraction with layout analysis
- `PPTXParser`: Comprehensive PowerPoint content extraction
- `ExcelParser`: Multi-sheet Excel data processing
- `UnifiedDocumentParser`: Format detection and routing

### Text Processing (`chunking.py`)
- `CrossPageTextSplitter`: Context-aware chunking across page boundaries
- `ParentPageAggregator`: Hierarchical chunk organization
- Token-aware splitting with configurable overlap

### Vector Database (`vectorstore.py`)
- `VectorStoreManager`: Persistent storage with metadata recovery
- Batch processing for large document sets
- Automatic retry logic for API rate limits

### Retrieval System (`retrieval.py`)
- `HybridRetriever`: Combined vector and keyword search
- LLM-based reranking for relevance optimization
- Configurable retrieval parameters

### Answer Generation (`generation.py`)
- `AnswerGenerator`: Structured response generation using `GPT-4.1-mini`
- Confidence scoring and uncertainty handling
- Source attribution and reasoning chains

### Workflow Orchestration (`workflow.py`)
- `LangGraph` state machine implementation
- Separate pipelines for document processing and querying
- Error handling and state management

## Features

### Advanced Document Processing
- Multi-column PDF layout detection
- Table and chart extraction from presentations
- Cross-page text chunking with context preservation
- Metadata-rich document representation

### Intelligent Retrieval
- Hybrid search combining semantic and keyword matching
- LLM-powered relevance reranking
- Configurable retrieval parameters
- Source document tracking

### Structured Generation
- Confidence-scored responses using `GPT-4.1-mini`
- Step-by-step reasoning chains
- Source attribution with page references
- Uncertainty acknowledgment

### Performance Monitoring
- Comprehensive query logging
- Processing time tracking
- Token usage monitoring
- Error rate analysis

### Scalability Features
- Batch processing for large document sets
- Persistent vector database with incremental updates
- Automatic retry logic for API failures
- Memory-efficient chunking strategies

## System Architecture

The system follows a modular architecture with clear separation of concerns:

- **Data Layer**: Document parsing and storage management
- **Processing Layer**: Text chunking and vector embedding
- **Retrieval Layer**: Hybrid search and reranking
- **Generation Layer**: LLM-based answer synthesis using `GPT-4.1-mini`
- **Orchestration Layer**: Workflow management and error handling

Each module is designed for independent testing and maintenance, with well-defined interfaces and comprehensive error handling.

## Performance Considerations

- Batch processing prevents OpenAI API token limit violations
- Persistent vector storage eliminates reprocessing overhead
- Hybrid retrieval balances accuracy and speed
- Token counting prevents context window overflow
- Incremental document addition for large knowledge bases

## Monitoring and Logging

The system provides comprehensive logging including:
- Query processing times and token usage
- Retrieval effectiveness metrics
- Generation quality indicators
- Error tracking and debugging information
- Performance analytics for optimization

## Extensibility

The modular design supports easy extension:
- Additional document format parsers
- Custom chunking strategies
- Alternative embedding models
- Enhanced retrieval algorithms
- Specialized generation pipelines
