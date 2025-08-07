"""
Document retrieval system with vector search, parent aggregation, and LLM reranking.
"""
import re
import json
import time
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema.messages import BaseMessage
from concurrent.futures import ThreadPoolExecutor
from chunking import ParentPageAggregator
from models import RetrievalRankingSingleBlock, RetrievalRankingMultipleBlocks


class LLMReranker:
    """LLM-based document reranking for improved relevance."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.system_prompt_multiple = """
You are an expert document relevance evaluator. Your task is to analyze text blocks and determine their relevance to a specific query.

For each text block provided, you must:
1. Analyze how well the content directly addresses the query
2. Consider factual accuracy and completeness of information
3. Evaluate the specificity and detail level
4. Assess the recency and context relevance

Scoring Guidelines:
- 1.0: Highly relevant, directly answers the query with comprehensive information
- 0.8-0.9: Very relevant, contains substantial useful information 
- 0.6-0.7: Moderately relevant, provides some useful context or partial answers
- 0.4-0.5: Slightly relevant, mentions related topics but limited direct value
- 0.0-0.3: Not relevant, off-topic or provides no useful information

Important Instructions:
   - Always provide relevance scores between 0.0 and 1.0
   - Base scores only on content that directly relates to the query
   - Be clear and concise in justifications
   - Do not infer information beyond what's explicitly stated
"""
    
    def rerank_documents(self, query: str, documents: List[Dict], 
                        documents_batch_size: int = 3, llm_weight: float = 0.7) -> List[Dict]:
        """Rerank pages using LLM with relevance score adjustment."""
        if not documents:
            return []
        
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        def process_batch(batch):
            texts = [doc['text'] for doc in batch]
            llm_scores = self._rerank_batch(texts, query)
            
            results = []
            for doc, llm_score in zip(batch, llm_scores):
                doc_with_score = doc.copy()
                doc_with_score['llm_score'] = llm_score
                doc_with_score['relevance_score'] = llm_score
                
                # Convert distance to similarity score (0-1 range)
                # Use exponential decay to handle distances > 1.0
                distance = doc.get('distance', 0.5)
                vector_similarity = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
                combined_score = llm_weight * llm_score + vector_weight * vector_similarity
                doc_with_score['combined_score'] = round(combined_score, 4)
                results.append(doc_with_score)
            
            return results
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            batch_results = list(executor.map(process_batch, doc_batches))
        
        all_results = []
        for batch in batch_results:
            all_results.extend(batch)
        
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return all_results
    
    def _rerank_batch(self, texts: List[str], question: str) -> List[float]:
        """Enhanced unified reranking with detailed reasoning and robust parsing."""
        if not texts:
            return []
        
        blocks_text = ""
        for i, text in enumerate(texts, 1):
            max_length = 1200 if len(texts) == 1 else 800
            truncated_text = text[:max_length] + "..." if len(text) > max_length else text
            blocks_text += f"\nBlock {i}:\n{truncated_text}\n"
        
        user_prompt = f"""
Query: {question}

Text Blocks:
{blocks_text}
"""
        
        schema_str = """
{
  "block_rankings": [
    {
      "reasoning": "string",
      "relevance_score": float
    }
  ]
}
"""
        
        full_prompt = f"{self.system_prompt_multiple}\n\nYour response must be a valid JSON object matching this schema:\n{schema_str}\n\nProvide one ranking object for each of the {len(texts)} blocks in order.\n\n{user_prompt}"
        
        try:
            response = self.llm.invoke(full_prompt)
            if isinstance(response, BaseMessage):
                response_content = str(response.content)
            else:
                response_content = str(response)
            
            rankings = self._parse_rankings_response(response_content, len(texts))
            return [ranking.relevance_score for ranking in rankings.block_rankings]
            
        except Exception as e:
            print(f"Warning: Error in reranking batch: {e}")
            return [0.5] * len(texts)
    
    def _parse_rankings_response(self, response_content: str, expected_count: int) -> RetrievalRankingMultipleBlocks:
        """Parse ranking response with comprehensive fallback strategies."""
        try:
            parsed = json.loads(response_content)
            return RetrievalRankingMultipleBlocks(**parsed)
        except (json.JSONDecodeError, ValueError):
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'\{.*?\}'
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response_content, re.DOTALL)
                if json_match:
                    try:
                        extracted = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        parsed = json.loads(extracted)
                        return RetrievalRankingMultipleBlocks(**parsed)
                    except:
                        continue
            
            score_matches = re.findall(r'(?:score|relevance)[:\s]*([0-9.]+)', response_content, re.IGNORECASE)
            scores = [max(0.0, min(1.0, float(match))) for match in score_matches[:expected_count]]
            
            if len(scores) < expected_count:
                number_matches = re.findall(r'[0-9.]+', response_content)
                potential_scores = []
                for match in number_matches:
                    try:
                        score = float(match)
                        if 0.0 <= score <= 1.0:
                            potential_scores.append(score)
                    except ValueError:
                        continue
                
                while len(scores) < expected_count and potential_scores:
                    scores.append(potential_scores.pop(0))
            
            while len(scores) < expected_count:
                scores.append(0.5)
            
            rankings = []
            for i, score in enumerate(scores):
                rankings.append(RetrievalRankingSingleBlock(
                    reasoning=f"Fallback parsing for block {i+1} - extracted score from unstructured response",
                    relevance_score=score
                ))
            
            return RetrievalRankingMultipleBlocks(block_rankings=rankings)


class VectorRetriever:
    """Vector-based document retrieval using embedding model and vector database."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, top_k: int = 30) -> List[Dict]:
        """Retrieve chunks using vector similarity search."""
        if not self.vectorstore:
            return []
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        results = []
        for doc, score in docs_with_scores:
            result = {
                'text': doc.page_content,
                'page': doc.metadata.get('page', 0),
                'chunk': doc.metadata.get('chunk', 1),
                'distance': float(score),
                'source_file': doc.metadata.get('source_file', ''),
                'document_type': doc.metadata.get('document_type', 'unknown'),
                'metadata': doc.metadata
            }
            results.append(result)
        
        return results


class HybridRetriever:
    """Complete retrieval system following the five-stage pipeline."""
    
    def __init__(self, vectorstore, parsed_reports: List[Dict]):
        self.vector_retriever = VectorRetriever(vectorstore)
        self.parent_aggregator = ParentPageAggregator(parsed_reports)
        self.reranker = LLMReranker()
        
    def retrieve(
        self, 
        query: str, 
        llm_reranking_sample_size: int = 30,
        documents_batch_size: int = 2,
        top_n: int = 10,
        llm_weight: float = 0.7
    ) -> List[Dict]:
        """Complete retrieval pipeline with vector search, parent aggregation and LLM reranking."""
        chunk_results = self.vector_retriever.retrieve(
            query=query,
            top_k=llm_reranking_sample_size
        )
        
        parent_results = self.parent_aggregator.aggregate_to_parent_pages(chunk_results)
        
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=parent_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]


def assemble_context(results: List[Dict]) -> str:
    """Assemble final context from retrieved results."""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        text_content = result['text']
        page_info = result.get('page', 'Unknown')
        source_file = result.get('source_file', 'Unknown')
        
        context_parts.append(f"Document {i} (Page {page_info}, {source_file}):\n{text_content}")
    
    return '\n\n'.join(context_parts)
