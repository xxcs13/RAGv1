"""
Data models and state management for the RAG system.
"""
from typing import List, Any, Dict, Union, Literal, Sequence
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain.schema.document import Document


@dataclass
class GraphState:
    """State management for the RAG workflow graph."""
    docs: Sequence[Union[str, Document]] = field(default_factory=list)
    vectorstore: Any = None
    question: str = ""
    retrieved_docs: List[Document] = field(default_factory=list)
    answer: str = ""
    
    # Enhanced state fields
    parsed_reports: List[Dict] = field(default_factory=list)
    vector_results: List[Dict] = field(default_factory=list)
    reranked_results: List[Dict] = field(default_factory=list)
    final_context: str = ""
    structured_answer: Dict = field(default_factory=dict)
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


class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="Analysis of the block and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1")


class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
