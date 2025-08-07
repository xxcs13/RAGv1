"""
Vector database management for RAG system.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """Vector database persistence and loading with document metadata recovery."""
    
    def __init__(self, persist_directory: str = "chromadb_test"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.metadata_file = os.path.join(persist_directory, "document_metadata.json")
    
    def vectorstore_exists(self) -> bool:
        """Check if vector database exists in persist directory."""
        if not os.path.exists(self.persist_directory):
            return False
        
        required_files = ['chroma.sqlite3']
        for file in required_files:
            if not os.path.exists(os.path.join(self.persist_directory, file)):
                return False
        
        return True
    
    def save_document_metadata(self, parsed_reports: List[Dict]):
        """Save document metadata for reconstruction when loading existing vectorstore."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            metadata = {
                'parsed_reports': []
            }
            
            for report in parsed_reports:
                # Save essential information needed for ParentPageAggregator
                report_metadata = {
                    'file_path': str(report['file_path']),
                    'report': {
                        'metainfo': report['report']['metainfo'],
                        'content': report['report']['content']
                    }
                }
                metadata['parsed_reports'].append(report_metadata)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"Saved document metadata to {self.metadata_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save document metadata: {e}")
    
    def load_document_metadata(self) -> List[Dict]:
        """Load document metadata for existing vectorstore."""
        try:
            if not os.path.exists(self.metadata_file):
                print(f"Warning: No metadata file found at {self.metadata_file}")
                return []
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            parsed_reports = metadata.get('parsed_reports', [])
            print(f"Loaded metadata for {len(parsed_reports)} documents")
            
            return parsed_reports
            
        except Exception as e:
            print(f"Warning: Failed to load document metadata: {e}")
            return []
    
    def load_existing_vectorstore(self) -> Tuple[Optional[Chroma], List[Dict]]:
        """Load existing vector database and associated document metadata."""
        try:
            if not self.vectorstore_exists():
                return None, []
            
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Load associated document metadata
            parsed_reports = self.load_document_metadata()
            
            # Test vectorstore and get count
            try:
                test_results = vectorstore.similarity_search("test", k=1)
                collection = vectorstore.get()
                doc_count = len(collection['documents']) if collection and 'documents' in collection else 0
                print(f"Loaded existing vector database with {doc_count} documents")
            except Exception as count_error:
                print(f"Loaded existing vector database (count unavailable: {count_error})")
            
            return vectorstore, parsed_reports
            
        except Exception as e:
            print(f"Error loading existing vector database: {e}")
            return None, []
    
    def create_vectorstore(self, documents: List[Document], parsed_reports: List[Dict] = None) -> Chroma:
        """Create new vector database from documents and save metadata."""
        # Create vectorstore with batch processing to avoid token limits (300,000 tokens)
        batch_size = 100  # Process documents in batches of 100
        total_docs = len(documents)
        
        print(f"Creating vector database with {total_docs} documents in batches of {batch_size}...")
        
        vectorstore = None
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            try:
                if vectorstore is None:
                    # Create initial vectorstore with first batch
                    vectorstore = Chroma.from_documents(
                        batch,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                else:
                    # Add subsequent batches to existing vectorstore
                    vectorstore.add_documents(batch)
                    
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                if "max_tokens_per_request" in str(e):
                    print("Reducing batch size and retrying...")
                    # If still too many tokens, process smaller batches
                    smaller_batch_size = 50
                    for j in range(0, len(batch), smaller_batch_size):
                        small_batch = batch[j:j+smaller_batch_size]
                        print(f"  Processing smaller batch {j//smaller_batch_size + 1} ({len(small_batch)} documents)...")
                        try:
                            if vectorstore is None:
                                vectorstore = Chroma.from_documents(
                                    small_batch,
                                    embedding=self.embeddings,
                                    persist_directory=self.persist_directory
                                )
                            else:
                                vectorstore.add_documents(small_batch)
                        except Exception as small_e:
                            print(f"Error in smaller batch: {small_e}")
                            raise small_e
                else:
                    raise e
        
        # Save document metadata for future use
        if parsed_reports:
            self.save_document_metadata(parsed_reports)
        
        print(f"Successfully created vector database with {total_docs} documents")
        return vectorstore
    
    def get_vectorstore_stats(self, vectorstore: Chroma) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            collection = vectorstore.get()
            count = len(collection['documents']) if collection and 'documents' in collection else 0
            return {
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "text-embedding-3-small"
            }
        except Exception as e:
            return {"error": str(e)}
