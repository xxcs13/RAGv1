"""
Text chunking and aggregation for RAG system.
"""
import re
import os
from typing import List, Dict, Any, Tuple, Sequence, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dataclasses import dataclass, field


class CrossPageTextSplitter:
    """Enhanced document chunking with cross-page support."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        """Initialize cross-page text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4.1-mini",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_document(self, document_data: Dict) -> List[Document]:
        """Split document into chunks with format-specific handling."""
        doc_type = document_data['metainfo']['document_type']
        
        if doc_type == 'excel':
            chunks = self._split_excel_document(document_data)
        elif doc_type == 'pdf':
            chunks = self._split_pdf_document(document_data)
        elif doc_type == 'pptx':
            chunks = self._split_pptx_document(document_data)
        else:
            chunks = self._split_cross_page_document(document_data)
        
        return chunks
    
    def _split_cross_page_document(self, document_data: Dict) -> List[Document]:
        """Split document content across page boundaries for better semantic continuity."""
        pages = document_data['content']['pages']
        if not pages:
            return []
        
        combined_text, page_boundaries = self._combine_pages_with_markers(pages)
        if not combined_text.strip():
            return []
        
        text_chunks = self.text_splitter.split_text(combined_text)
        
        documents = []
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
                
            clean_chunk = self._remove_page_markers(chunk)
            chunk_start = combined_text.find(chunk)
            chunk_end = chunk_start + len(chunk)
            page_range = self._get_page_range(chunk_start, chunk_end, page_boundaries)
            
            metadata = {
                "chunk": i + 1,
                "total_chunks": len(text_chunks),
                "content_type": "cross_page_text"
            }
            
            if len(page_range) == 1:
                metadata["page"] = page_range[0]
            else:
                metadata["page"] = page_range[0]
                metadata["page_range"] = ",".join(map(str, page_range))
                metadata["spans_pages"] = True
            
            documents.append(Document(
                page_content=clean_chunk.strip(),
                metadata=metadata
            ))
        
        return documents
    
    def _combine_pages_with_markers(self, pages: List[Dict]) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Combine pages into continuous text with boundary markers."""
        combined_parts = []
        page_boundaries = []
        current_pos = 0
        
        for page_data in pages:
            page_num = page_data['page']
            page_text = page_data.get('text', '').strip()
            
            if not page_text:
                continue
            
            page_marker = f"\n--- PAGE {page_num} ---\n"
            page_content = page_marker + page_text + "\n"
            
            start_pos = current_pos
            end_pos = current_pos + len(page_content)
            
            page_boundaries.append((page_num, start_pos, end_pos))
            combined_parts.append(page_content)
            current_pos = end_pos
        
        return ''.join(combined_parts), page_boundaries
    
    def _remove_page_markers(self, text: str) -> str:
        """Remove page markers from chunk text."""
        return re.sub(r'\n--- PAGE \d+ ---\n', '\n', text)
    
    def _get_page_range(self, chunk_start: int, chunk_end: int, page_boundaries: List[Tuple[int, int, int]]) -> List[int]:
        """Determine which pages a chunk spans based on position."""
        covered_pages = []
        
        for page_num, start_pos, end_pos in page_boundaries:
            if chunk_start < end_pos and chunk_end > start_pos:
                covered_pages.append(page_num)
        
        return sorted(covered_pages) if covered_pages else [1]
    
    def _split_excel_document(self, document_data: Dict) -> List[Document]:
        """Excel-specific chunking: each page (sheet) becomes one complete chunk."""
        chunks = []
        pages = document_data['content']['pages']
        
        for page in pages:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # Create one chunk per Excel sheet (page)
            chunk = Document(
                page_content=page_text.strip(),
                metadata={
                    "page": page['page'],
                    "chunk": 1,
                    "total_chunks": 1,
                    "content_type": "excel_full_sheet",
                    "sheet_name": self._extract_sheet_name(page_text),
                    "chunking_strategy": "full_sheet_per_chunk"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_sheet_name(self, sheet_text: str) -> str:
        """Extract sheet name from Excel sheet text."""
        lines = sheet_text.split('\n')
        for line in lines:
            if line.startswith('Sheet:'):
                sheet_name = line.replace('Sheet:', '').strip()
                return sheet_name if sheet_name else 'Unknown Sheet'
        return 'Unknown Sheet'
    
    def _split_pdf_document(self, document_data: Dict) -> List[Document]:
        """PDF-specific chunking with layout analysis and column detection."""
        chunks = []
        
        for page in document_data['content']['pages']:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # Get layout information if available
            layout_type = page.get('layout_type', 'single_column')
            column_count = page.get('column_count', 1)
            extraction_method = page.get('extraction_method', 'standard')
            
            page_chunks = self._split_pdf_page(
                page_text, 
                page['page'], 
                layout_type, 
                column_count,
                extraction_method
            )
            chunks.extend(page_chunks)
        
        return chunks
    
    def _split_pdf_page(self, page_text: str, page_num: int, layout_type: str, 
                       column_count: int, extraction_method: str) -> List[Document]:
        """Split PDF page while preserving layout-aware structure."""
        # Choose chunking strategy based on layout analysis
        if layout_type == 'single_column' or column_count <= 1:
            return self._chunk_single_column_pdf(page_text, page_num, extraction_method)
        elif layout_type == 'multi_column' and column_count > 1:
            return self._chunk_multi_column_pdf(page_text, page_num, column_count)
        else:
            # Complex or unstructured layout
            return self._chunk_complex_pdf(page_text, page_num)
    
    def _chunk_single_column_pdf(self, page_text: str, page_num: int, 
                                 extraction_method: str) -> List[Document]:
        """Handle single-column PDF pages with standard chunking."""
        text_chunks = self.text_splitter.split_text(page_text)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
            
            documents.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "page": page_num,
                    "chunk": i + 1,
                    "total_chunks": len(text_chunks),
                    "content_type": "pdf_single_column",
                    "layout_type": "single_column",
                    "extraction_method": extraction_method
                }
            ))
        
        return documents
    
    def _chunk_multi_column_pdf(self, page_text: str, page_num: int, 
                                column_count: int) -> List[Document]:
        """Handle multi-column PDF layouts with column-aware chunking."""
        documents = []
        
        # Strategy 1: Look for column markers (explicit column separation)
        if '||COLUMN||' in page_text:
            # Split by column markers and process each column independently
            columns = page_text.split('||COLUMN||')
            chunk_counter = 1
            
            for column_idx, column_content in enumerate(columns):
                if not column_content.strip():
                    continue
                
                # Process each column independently with its own chunking
                column_chunks = self.text_splitter.split_text(column_content.strip())
                
                for chunk_idx, chunk in enumerate(column_chunks):
                    if not chunk.strip():
                        continue
                    
                    documents.append(Document(
                        page_content=chunk.strip(),
                        metadata={
                            "page": page_num,
                            "chunk": chunk_counter,
                            "total_chunks": None,  # Will be set later
                            "column": column_idx + 1,
                            "column_chunk": chunk_idx + 1,
                            "content_type": "pdf_multi_column",
                            "layout_type": "multi_column",
                            "column_count": column_count
                        }
                    ))
                    chunk_counter += 1
        else:
            # Strategy 2: Standard chunking for multi-column content without explicit markers
            text_chunks = self.text_splitter.split_text(page_text)
            
            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():
                    continue
                
                documents.append(Document(
                    page_content=chunk.strip(),
                    metadata={
                        "page": page_num,
                        "chunk": i + 1,
                        "total_chunks": len(text_chunks),
                        "content_type": "pdf_multi_column_merged",
                        "layout_type": "multi_column",
                        "column_count": column_count
                    }
                ))
        
        # Update total_chunks for column-based chunking
        total_chunks = len(documents)
        for doc in documents:
            if doc.metadata.get("total_chunks") is None:
                doc.metadata["total_chunks"] = total_chunks
        
        return documents
    
    def _chunk_complex_pdf(self, page_text: str, page_num: int) -> List[Document]:
        """Handle complex PDF layouts with fallback chunking."""
        text_chunks = self.text_splitter.split_text(page_text)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
            
            documents.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "page": page_num,
                    "chunk": i + 1,
                    "total_chunks": len(text_chunks),
                    "content_type": "pdf_complex_layout",
                    "layout_type": "complex",
                    "extraction_method": "fallback"
                }
            ))
        
        return documents
    
    def _split_pptx_document(self, document_data: Dict) -> List[Document]:
        """PPTX-specific chunking: each slide becomes one chunk."""
        chunks = []
        pages = document_data['content']['pages']
        
        for page in pages:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # For PPTX, each slide is treated as a complete unit
            chunk = Document(
                page_content=page_text.strip(),
                metadata={
                    "page": page['page'],
                    "chunk": 1,
                    "total_chunks": 1,
                    "content_type": "pptx_full_slide",
                    "slide_number": page['page'],
                    "chunking_strategy": "full_slide_per_chunk"
                }
            )
            chunks.append(chunk)
        
        return chunks


class ParentPageAggregator:
    """Enhanced parent page retrieval with support for cross-page chunks."""
    
    def __init__(self, parsed_reports: List[Dict]):
        """Initialize with parsed document reports."""
        self.parsed_reports = parsed_reports
        self.page_content_map = self._build_page_content_map()
    
    def _build_page_content_map(self) -> Dict[int, str]:
        """Build mapping from page numbers to full page content."""
        page_map = {}
        for report in self.parsed_reports:
            for page_data in report['report']['content']['pages']:
                page_num = page_data['page']
                page_map[page_num] = page_data['text']
        return page_map
    
    def aggregate_to_parent_pages(self, chunk_results: List[Dict]) -> List[Dict]:
        """Extract parent pages from chunks with cross-page support."""
        seen_page_combinations = set()
        parent_results = []
        
        for chunk_result in chunk_results:
            page_coverage = self._get_chunk_page_coverage(chunk_result)
            page_combination_key = tuple(sorted(page_coverage))
            
            if page_combination_key not in seen_page_combinations:
                seen_page_combinations.add(page_combination_key)
                combined_content = self._get_combined_page_content(page_coverage)
                
                parent_result = {
                    'text': combined_content,
                    'page': page_coverage[0],
                    'page_range': ",".join(map(str, page_coverage)) if len(page_coverage) > 1 else None,
                    'spans_pages': len(page_coverage) > 1,
                    'distance': chunk_result['distance'],
                    'source_file': chunk_result['source_file'],
                    'document_type': chunk_result['document_type'],
                    'metadata': chunk_result['metadata']
                }
                parent_results.append(parent_result)
        
        return parent_results
    
    def _get_chunk_page_coverage(self, chunk_result: Dict) -> List[int]:
        """Determine which pages a chunk covers."""
        metadata = chunk_result.get('metadata', {})
        
        if metadata.get('spans_pages', False) and 'page_range' in metadata:
            page_range_str = metadata['page_range']
            return [int(page) for page in page_range_str.split(',')]
        else:
            return [chunk_result['page']]
    
    def _get_combined_page_content(self, page_numbers: List[int]) -> str:
        """Combine content from multiple pages."""
        if len(page_numbers) == 1:
            return self.page_content_map.get(page_numbers[0], '')
        
        combined_parts = []
        for page_num in sorted(page_numbers):
            page_content = self.page_content_map.get(page_num, '')
            if page_content.strip():
                combined_parts.append(f"[Page {page_num}]\n{page_content}")
        
        return '\n\n--- PAGE BREAK ---\n\n'.join(combined_parts)
