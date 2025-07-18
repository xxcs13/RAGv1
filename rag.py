import os
import re
import json
import logging
import time
import tiktoken
from typing import List, Any, Dict, Tuple, Optional, Union, Literal, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import inspect

import pandas as pd
from pptx import Presentation
from pypdf import PdfReader  # type: ignore
import pdfplumber

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.document import Document
from langchain_core.messages import BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in your .env file."

# Configure environment for stability and privacy
# Set ENABLE_TELEMETRY=true in .env to allow telemetry data collection
enable_telemetry = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"

if not enable_telemetry:
    # CHROMA_TELEMETRY: Disable ChromaDB telemetry for privacy protection
    # TOKENIZERS_PARALLELISM: Disable to prevent multiprocessing conflicts (HuggingFace recommendation)  
    # ANONYMIZED_TELEMETRY: Disable anonymous data collection for enterprise compliance
    os.environ.setdefault("CHROMA_TELEMETRY", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    print("Telemetry disabled for privacy and stability. Set ENABLE_TELEMETRY=true in .env to enable.")
else:
    print("Telemetry enabled - data may be sent to service providers.")

_log = logging.getLogger(__name__)

# =============================================================================
# STRUCTURED PROMPT SYSTEM
# =============================================================================

def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    """Build structured system prompt with instruction, schema and example"""
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class EnhancedRAGAnswerPrompt:
    """Structured prompt system for RAG answer generation"""
    instruction = """
You are an advanced RAG (Retrieval-Augmented Generation) answering system that responds in Traditional Chinese.
Your task is to answer questions based on information from the provided documents, synthesizing insights across multiple sources when relevant.

IMPORTANT: The input documents may contain content in various languages (Traditional Chinese, Simplified Chinese, English, Japanese, etc.). You MUST carefully verify that the content is actually relevant to the user's question regardless of language barriers.

CRITICAL FINANCIAL DATA PRIORITY:
When dealing with MONEY-RELATED questions (revenue, profit, costs, financial performance, pricing, investment, etc.), you MUST prioritize examining Excel-type documents first. Excel files typically contain the most comprehensive numerical and financial data, including:
- Detailed financial statements and reports
- Numerical breakdowns and calculations
- Time-series financial data
- Quantitative metrics and KPIs
- Precise monetary amounts and figures
Always check Excel sources thoroughly for money-related queries before considering other document types.

QUESTION TYPE ANALYSIS & RESPONSE STRATEGY:
Before answering, classify the question type and adjust your response style accordingly:

FACTUAL/SPECIFIC QUERIES (direct, concise responses):
- Financial figures, dates, quantities, technical specifications
- "What is the revenue?", "How many employees?", "When did X happen?"
- Response style: Direct, precise, minimal interpretation
- Focus: Exact data extraction and clear presentation

ANALYTICAL/STRATEGIC QUERIES (detailed insights with business acumen):
- Trend analysis, competitive positioning, strategic implications
- "How is the company performing?", "What are the growth drivers?", "What's the competitive landscape?"
- Response style: Comprehensive analysis with business insights
- Focus: Strategic thinking, market dynamics, implications

CORE ANSWERING PRINCIPLES:
1. COMPREHENSION: Read the question carefully and identify all its components and requirements
2. LANGUAGE-AGNOSTIC RELEVANCE: Examine content in ALL languages for relevance - don't skip content just because it's in a different language
3. CONTEXT ANALYSIS: Examine each piece of context for direct relevance, accuracy, and completeness
4. CROSS-VALIDATION: When multiple sources discuss the same topic, cross-reference for consistency across languages
5. SYNTHESIS: Combine information from different sources to provide comprehensive insights
6. TRANSPARENCY: Clearly distinguish between what is explicitly stated vs. what is inferred
7. LIMITATION AWARENESS: Acknowledge when information is incomplete, conflicting, or missing

ROBUST ANALYSIS FRAMEWORK:
- Parse the question's explicit requirements and implicit expectations
- Identify key entities, time periods, metrics, and relationships mentioned
- Map each piece of context to specific parts of the question
- Note any contradictions or gaps between sources
- Distinguish between factual data, interpretations, and projections
- Consider temporal context (when data was collected vs. when question asks about)
- Evaluate data quality and source credibility indicators
- Account for potential biases in source materials

ANSWER QUALITY STANDARDS:
- Provide direct answers to all parts of multi-part questions
- Support claims with specific evidence from the context (regardless of source language)
- Use precise language that reflects the confidence level of available data
- ADAPT RESPONSE DEPTH based on question type:
  * Factual queries: Concise, direct answers with minimal elaboration
  * Analytical queries: Comprehensive insights with business context
- Maintain logical flow and clear reasoning chains
- Address potential counterarguments or alternative interpretations when relevant to question type
- Always respond in Traditional Chinese for the final answer
- CROSS-LANGUAGE VERIFICATION: Ensure relevance across all language sources

BUSINESS INTELLIGENCE & ANALYTICAL DEPTH (For Analytical/Strategic Questions ONLY):
- Apply strategic business thinking to interpret data beyond surface-level facts
- Identify underlying business drivers, competitive dynamics, and market forces
- Assess potential risks, opportunities, and strategic implications
- Connect dots between different business metrics and their interdependencies
- Evaluate timing considerations and market context that could affect outcomes
- Consider stakeholder perspectives (investors, customers, competitors, suppliers)
- Provide forward-looking insights based on historical patterns and current trends
- Identify potential catalysts or headwinds that could accelerate/decelerate trends
- Assess the sustainability and scalability of business developments
- Consider broader industry dynamics and macroeconomic factors when relevant

NOTE: For factual/specific queries, focus on data extraction and presentation rather than deep business analysis.

ANALYTICAL SOPHISTICATION (Adjust based on question type):

For FACTUAL queries:
- Focus on precise data extraction and clear presentation
- Verify accuracy across multiple language sources
- Present numbers in standardized format

For ANALYTICAL queries:
- Move beyond "what" to explore "why" and "so what"
- Identify cause-and-effect relationships and feedback loops
- Recognize leading vs. lagging indicators
- Assess the quality and sustainability of business performance
- Consider cyclical vs. structural factors
- Evaluate competitive positioning and moat strength
- Analyze resource allocation efficiency and strategic priorities
- Consider regulatory, technological, and market disruption potential

HANDLING EDGE CASES:
- If information is insufficient: State exactly what is missing and why
- If sources contradict: Present both perspectives and note the discrepancy
- If question asks about future trends: Clearly distinguish between historical data and projections
- If context is tangentially related: Explain the connection and acknowledge limitations
- If multiple interpretations exist: Present the most supported interpretation while noting alternatives

ENHANCED NUMBER FORMATTING RULES:
When presenting numbers, especially large figures, use this comprehensive format:

FOR NUMBERS IN BILLIONS:
- English: "289.43 billion" → Chinese: "2,894,300,000,000(2兆8千9百43億)" 
- Always include "billion" in English context and "兆" for Chinese context when appropriate
- For values ≥ 1 billion: Use both billion notation and 兆/億 conversion

FOR STANDARD LARGE NUMBERS:
- Format: "2,894,307,699(28億9千4百30萬7千6百99)"
- Always include both comma-separated number and Chinese numerical description in parentheses

SCALE CONVERSIONS:
- Thousands: "千" (1,000 = 1千)
- Ten thousands: "萬" (10,000 = 1萬)
- Millions: "百萬" (1,000,000 = 100萬)
- Ten millions: "千萬" (10,000,000 = 1千萬)
- Hundred millions: "億" (100,000,000 = 1億)
- Billions: "十億" (1,000,000,000 = 10億)
- Ten billions: "百億" (10,000,000,000 = 100億)
- Hundred billions: "千億" (100,000,000,000 = 1千億)
- Trillions: "兆" (1,000,000,000,000 = 1兆)

SPECIAL CASES:
- Percentages: Include both decimal and whole number forms when relevant
- Growth rates: Specify time periods and base comparisons
- Financial metrics: Include currency and reporting period context
- Technical specifications: Maintain precision appropriate to the field

Apply this enhanced formatting to ALL significant numbers in your final answer, including revenue, profit, market share, quantities, technical specifications, dates, and statistical measures.
"""

    user_prompt = """
Here is the context:
\"\"\"
{context}
\"\"\"

---

Here is the question:
"{question}"

---

CRITICAL REMINDERS:
1. MULTI-LANGUAGE PROCESSING: The context may contain documents in various languages (Traditional Chinese, Simplified Chinese, English, Japanese, etc.). You MUST examine ALL content regardless of language to find relevant information.

2. FINANCIAL DATA PRIORITY: For any MONEY-RELATED questions (revenue, profit, costs, financial metrics, etc.), prioritize examining Excel-type sources first as they typically contain the most accurate and detailed financial data.

3. QUESTION TYPE CLASSIFICATION: Determine if this is:
   - FACTUAL/SPECIFIC query (e.g., "What is the revenue?", "How many employees?") → Provide direct, concise answers
   - ANALYTICAL/STRATEGIC query (e.g., "How is performance?", "What are growth drivers?") → Apply comprehensive business analysis

4. RELEVANCE VERIFICATION: Only include information that is actually relevant to the question, regardless of the source language.

5. RESPONSE DEPTH: Match your response complexity to the question type - avoid over-analyzing simple factual queries, but provide deep insights for strategic questions.
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
Analysis following the robust framework (adjust depth based on question type):

For FACTUAL questions (concise, 3-4 steps, ~100 words):
1. Question type identification: Recognize as factual/specific query + identify if money-related
2. Multi-language source scanning: Check all language sources for relevant data (prioritize Excel files for financial questions)
3. Data verification: Confirm accuracy and consistency across sources
4. Direct extraction: Present the specific information requested

For ANALYTICAL questions (comprehensive, 5+ steps, 150+ words):
1. Question parsing: Identify all explicit and implicit requirements + financial data needs
2. Context mapping: Map each context piece to question components across all languages (prioritize Excel for financial metrics)
3. Cross-validation: Check for consistency/contradictions across sources
4. Evidence evaluation: Assess data quality, credibility, and temporal relevance  
5. Synthesis: Combine insights while noting limitations and confidence levels

Pay special attention to cross-language relevance verification and question type appropriateness.
""")

        reasoning_summary: str = Field(description="Concise synthesis summary highlighting key evidence, confidence level rationale, and any significant limitations (50-80 words).")

        relevant_sources: List[str] = Field(description="""
Source IDs containing information DIRECTLY used in the answer. Apply strict criteria:
- Include ONLY sources with explicit data/statements that directly answer the question
- Sources providing key supporting evidence or critical context
- Exclude tangentially related or weak-connection sources
- Use exact source IDs as shown in context (format: filetype_filename_page_X)
- Quality over quantity: prefer fewer highly relevant sources
""")

        confidence_level: Literal["high", "medium", "low"] = Field(description="""
Confidence assessment based on:
- HIGH: Multiple consistent sources, complete data, direct answers to all question parts
- MEDIUM: Adequate sources with minor gaps, mostly direct answers, some interpretation needed
- LOW: Limited sources, significant gaps, indirect answers, or conflicting information
""")

        final_answer: str = Field(description="""
Answer in Traditional Chinese with appropriate depth based on question type:

UNIVERSAL REQUIREMENTS:
- Address ALL parts of the question systematically with supporting evidence from all language sources
- For money-related questions, prioritize and emphasize data from Excel-type sources
- Use enhanced number formatting with billion/兆 conversions as specified
- Distinguish clearly between facts, interpretations, and projections
- Verify cross-language source relevance

FOR FACTUAL/SPECIFIC QUESTIONS (concise, direct approach):
- Provide direct, precise answers with minimal elaboration
- Focus on exact data extraction and clear presentation
- Include specific figures, dates, quantities as requested
- Cite relevant sources briefly
- Avoid unnecessary business analysis unless directly relevant

FOR ANALYTICAL/STRATEGIC QUESTIONS (comprehensive insights):
- Go beyond surface facts to explain underlying business dynamics and drivers
- Identify strategic implications, competitive advantages, and market positioning
- Assess sustainability of trends and potential risk factors
- Connect financial metrics to operational performance and market context
- Evaluate timing considerations and market cycle positioning
- Consider stakeholder impact (investors, customers, industry ecosystem)
- Explain "why" behind the numbers and "so what" for stakeholders
- Identify cause-and-effect relationships and interdependencies
- Assess quality of growth/performance (organic vs. inorganic, sustainable vs. cyclical)
- Evaluate competitive moats and differentiation factors

PROFESSIONAL PRESENTATION:
- Maintain analytical rigor while avoiding mechanical, formulaic responses
- Use natural, engaging language appropriate to question complexity
- Structure insights logically from facts → analysis → implications (for analytical questions)
- Acknowledge limitations and uncertainties transparently
- Provide balanced perspective considering multiple viewpoints when appropriate
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Examples:

FACTUAL QUESTION Example:
Question: "台積電 2024 年第三季的營收是多少？"

Answer:
```
{
  "step_by_step_analysis": "1. 問題類型識別：這是一個事實性查詢，需要直接提取特定的財務數據。\n2. 多語言源掃描：檢查所有語言的文件中關於台積電 2024 Q3 營收的資訊。\n3. 數據驗證：確認不同來源中的數據一致性和準確性。\n4. 直接提取：從財務報告中提取準確的營收數字。",
  "reasoning_summary": "直接從官方財務報告提取 2024 年第三季營收數據，數據來源可靠。",
  "relevant_sources": ["pdf_tsmc_2024_q3_earnings_pdf_page_2"],
  "confidence_level": "high",
  "final_answer": "根據台積電 2024 年第三季財務報告，該季營收為 759.69 billion 新台幣(7千5百96億9千萬新台幣)。"
}
```

ANALYTICAL QUESTION Example:
Question: "結合台積電 69% 先進製程營收占比及 DIGITIMES AI 伺服器需求預估，說明台積電 2025 年主要營收成長動力。"

Answer:
```
{
  "step_by_step_analysis": "1. 戰略情境分析：台積電面臨 AI 運算需求爆發的歷史性機遇，69% 先進製程營收占比顯示公司已成功轉型為技術密集型業務模式，與傳統代工業者形成差異化。\n2. 競爭動態解讀：先進製程技術門檻的指數級提升創造了寡頭壟斷格局，台積電在 3nm/5nm 領域的領先地位構築了高難度複製的護城河，客戶別無選擇只能依賴其產能。\n3. 需求結構分析：AI 伺服器市場的雙位數成長不同於過往週期性需求，其背後是數位轉型和 AI 應用的結構性趨勢，需求具有剛性特徵且不易替代。\n4. 商業模式演進：從單純代工轉向「製程+封裝」完整解決方案，CoWoS 技術不僅提升附加價值，更加深客戶依賴度，形成生態系統鎖定效應。\n5. 風險機會評估：雖然 AI 浪潮提供成長動能，但需留意客戶集中風險、地緣政治衝擊，以及新興技術可能帶來的顛覆性挑戰。成功關鍵在於維持技術領先與多元化平衡。",
  "reasoning_summary": "台積電憑藉技術護城河與 AI 結構性需求的戰略契合，正經歷從週期性代工向高附加價值平台的商業模式升級，成長動力具備可持續性但需注意風險平衡。",
  "relevant_sources": ["pdf_tsmc_2024_yearly_report_pdf_page_8", "0.pptx_slide_4"],
  "confidence_level": "medium",
  "final_answer": "從戰略角度分析，台積電正處於 AI 革命浪潮的核心位置，其 2025 年營收成長將體現三重競爭優勢的協同效應：\n\n技術護城河深化：69% 先進製程營收占比(約 690 billion 新台幣/6千9百億新台幣)不僅反映當前領導地位，更重要的是展現了技術密集型商業模式的獲利能力。3nm/5nm 製程的技術門檻極高，競爭對手追趕困難，形成結構性競爭優勢。\n\n市場需求結構性轉變：AI 伺服器雙位數成長背後反映的是運算架構的根本性變革。與過往週期性需求不同，AI 驅動的需求具有持續性和剛性特徵，為台積電提供了更可預期的營收基礎，降低了傳統半導體週期性波動的風險。\n\n生態系統主導權：CoWoS® 先進封裝技術形成了「製程+封裝」的完整解決方案，這種垂直整合能力讓台積電不僅是代工廠，更成為 AI 晶片生態的關鍵使能者。客戶轉換成本高，黏著度強。\n\n戰略意涵：台積電正從週期性代工業務轉向結構性成長模式，AI 浪潮為其創造了罕見的「量價齊升」機會。然而需關注地緣政治風險和客戶集中度問題，多元化策略將是長期競爭力的關鍵。"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

# =============================================================================
# VECTOR DATABASE MANAGEMENT
# =============================================================================

class VectorStoreManager:
    """Vector database persistence and loading"""
    
    def __init__(self, persist_directory: str = "chromadb_new"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    def vectorstore_exists(self) -> bool:
        """Check if vector database exists in persist directory"""
        if not os.path.exists(self.persist_directory):
            return False
        
        required_files = ['chroma.sqlite3']
        for file in required_files:
            if not os.path.exists(os.path.join(self.persist_directory, file)):
                return False
        
        return True
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector database from persist directory"""
        try:
            if not self.vectorstore_exists():
                return None
            
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Test if vectorstore is functional
            test_results = vectorstore.similarity_search("test", k=1)
            print(f"Loaded existing vector database with {vectorstore._collection.count()} documents")
            return vectorstore
            
        except Exception as e:
            print(f"Error loading existing vector database: {e}")
            return None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create new vector database from documents"""
        vectorstore = Chroma.from_documents(
            documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Created new vector database with {len(documents)} documents")
        return vectorstore
    
    def get_vectorstore_stats(self, vectorstore: Chroma) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = vectorstore._collection.count()
            return {
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "text-embedding-3-small"
            }
        except Exception as e:
            return {"error": str(e)}

# =============================================================================
# DOCUMENT PARSING SYSTEM
# =============================================================================

class PDFParser:
    """PDF text extraction with number formatting correction"""
    
    def __init__(self):
        pass
    
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text and tables from PDF file"""
        try:
            filename = Path(file_path).name
            pages = []
            
            # Try multiple extraction methods for best results
            try:
                pages = self._extract_with_pdfplumber(file_path)
                print(f"Successfully parsed PDF with pdfplumber: {filename}")
            except Exception as e:
                print(f"pdfplumber failed, trying pypdf: {e}")
                try:
                    pages = self._extract_with_pypdf(file_path)
                    print(f"Successfully parsed PDF with pypdf: {filename}")
                except Exception as e2:
                    print(f"Both PDF extraction methods failed: {e2}")
                    return self._create_fallback_report(file_path)
            
            # Post-process all pages for better number handling
            processed_pages = []
            for page_data in pages:
                processed_text = self._post_process_text(page_data['text'])
                if processed_text.strip():
                    processed_pages.append({
                        'page': page_data['page'],
                        'text': processed_text.strip()
                    })
            
            report = {
                'metainfo': {
                    'sha1_name': filename.rsplit('.', 1)[0],
                    'filename': filename,
                    'pages_amount': len(processed_pages),
                    'text_blocks_amount': len(processed_pages),
                    'tables_amount': 0,
                    'pictures_amount': 0,
                    'document_type': 'pdf'
                },
                'content': {'pages': processed_pages},
                'tables': [],
                'pictures': []
            }
            
            print(f"Successfully parsed PDF (enhanced): {filename} ({len(processed_pages)} pages)")
            return report
            
        except Exception as e:
            print(f"Error parsing PDF file {file_path}: {e}")
            return self._create_fallback_report(file_path)
    
    def _extract_with_pdfplumber(self, file_path: str) -> List[Dict]:
        """Extract text and tables using pdfplumber"""
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Try multiple extraction strategies
                text_parts = []
                
                # Strategy 1: Standard text extraction
                standard_text = page.extract_text()
                if standard_text:
                    text_parts.append(standard_text)
                
                # Strategy 2: Extract with layout preservation
                try:
                    layout_text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=1)
                    if layout_text and layout_text != standard_text:
                        text_parts.append("=== Layout Preserved ===")
                        text_parts.append(layout_text)
                except:
                    pass
                
                # Strategy 3: Extract tables if present
                try:
                    tables = page.extract_tables()
                    if tables:
                        text_parts.append("=== Tables ===")
                        for i, table in enumerate(tables):
                            if table:
                                table_text = self._format_table_text(table)
                                text_parts.append(f"Table {i+1}:\n{table_text}")
                except:
                    pass
                
                combined_text = '\n\n'.join(text_parts)
                if combined_text.strip():
                    pages.append({
                        'page': page_num,
                        'text': combined_text.strip()
                    })
        
        return pages
    
    def _extract_with_pypdf(self, file_path: str) -> List[Dict]:
        """Extract text using pypdf as fallback"""
        pages = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        'page': page_num,
                        'text': text.strip()
                    })
        
        return pages
    
    def _format_table_text(self, table: List[List]) -> str:
        """Format extracted table data into readable text"""
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            if row:
                # Clean and join cells
                clean_cells = []
                for cell in row:
                    if cell is not None:
                        cell_text = str(cell).strip()
                        # Fix common number formatting issues
                        cell_text = self._fix_number_formatting(cell_text)
                        clean_cells.append(cell_text)
                    else:
                        clean_cells.append("")
                formatted_rows.append(" | ".join(clean_cells))
        
        return "\n".join(formatted_rows)
    
    def _post_process_text(self, text: str) -> str:
        """Fix number formatting and spacing issues in extracted text"""
        if not text:
            return text
        
        # Fix common number formatting issues
        processed_text = self._fix_number_formatting(text)
        
        # Fix line breaks and spacing
        processed_text = self._fix_spacing_issues(processed_text)
        
        # Fix currency and percentage symbols
        processed_text = self._fix_symbols(processed_text)
        
        return processed_text
    
    def _fix_number_formatting(self, text: str) -> str:
        """Correct decimal points, thousand separators, and currency symbols"""
        # Fix decimal points that might be extracted as other characters
        text = re.sub(r'(\d)\s*[,，]\s*(\d{3})', r'\1,\2', text)  # Fix thousand separators
        text = re.sub(r'(\d)\s*[.．]\s*(\d)', r'\1.\2', text)     # Fix decimal points
        text = re.sub(r'(\d)\s*[oO]\s*(\d)', r'\1.0\2', text)     # Fix common OCR errors (o/O as 0)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)              # Remove spaces within numbers
        
        # Fix percentage signs
        text = re.sub(r'(\d)\s*[%％]', r'\1%', text)
        
        # Fix currency symbols
        text = re.sub(r'[$＄]\s*(\d)', r'$\1', text)
        text = re.sub(r'([NT$]+)\s*(\d)', r'\1\2', text)
        
        # Fix negative numbers
        text = re.sub(r'[-－—]\s*(\d)', r'-\1', text)
        
        return text
    
    def _fix_spacing_issues(self, text: str) -> str:
        """Fix spacing and line break issues"""
        # Fix broken words across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _fix_symbols(self, text: str) -> str:
        """Fix currency and other symbols"""
        # Standardize currency symbols
        text = re.sub(r'[＄$]', '$', text)
        text = re.sub(r'[％%]', '%', text)
        
        # Fix common symbol OCR errors
        text = re.sub(r'[（(]', '(', text)
        text = re.sub(r'[）)]', ')', text)
        
        return text
    
    def _create_fallback_report(self, file_path: str) -> Dict[str, Any]:
        """Create minimal report when PDF parsing fails"""
        filename = Path(file_path).name
        return {
            'metainfo': {
                'sha1_name': filename.rsplit('.', 1)[0],
                'filename': filename,
                'pages_amount': 0,
                'text_blocks_amount': 0,
                'tables_amount': 0,
                'pictures_amount': 0,
                'document_type': 'failed'
            },
            'content': {'pages': []},
            'tables': [],
            'pictures': []
        }

class PPTXParser:
    """PPTX content extraction including tables, charts, and images"""
    
    def __init__(self):
        pass
    
    def parse_pptx(self, file_path: str) -> Dict[str, Any]:
        """Parse PPTX file for all types of content"""
        try:
            filename = Path(file_path).name
            prs = Presentation(file_path)
            
            pages = []
            tables_found = 0
            charts_found = 0
            images_found = 0
            other_objects_found = 0
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_content = self._extract_slide_content(slide)
                
                # Count different object types
                tables_found += slide_content['stats']['tables']
                charts_found += slide_content['stats']['charts'] 
                images_found += slide_content['stats']['images']
                other_objects_found += slide_content['stats']['other_objects']
                
                # Combine all extracted content
                if slide_content['combined_text'].strip():
                    pages.append({
                        'page': slide_num,
                        'text': slide_content['combined_text'].strip()
                    })
            
            report = {
                'metainfo': {
                    'sha1_name': filename.rsplit('.', 1)[0],
                    'filename': filename,
                    'pages_amount': len(pages),
                    'text_blocks_amount': len(pages),
                    'tables_amount': tables_found,
                    'pictures_amount': images_found,
                    'document_type': 'pptx'
                },
                'content': {'pages': pages},
                'tables': [],
                'pictures': []
            }
            
            print(f"Successfully parsed PPTX: {filename}")
            print(f"  - {len(pages)} slides with content")
            print(f"  - {tables_found} tables, {charts_found} charts")
            print(f"  - {images_found} images, {other_objects_found} other objects")
            return report
            
        except Exception as e:
            print(f"Error parsing PPTX file {file_path}: {e}")
            return self._create_fallback_report(file_path)
    
    def _extract_slide_content(self, slide) -> Dict[str, Any]:
        """Extract all types of content from a single slide"""
        content_parts = []
        stats = {'tables': 0, 'charts': 0, 'images': 0, 'other_objects': 0}
        
        # Process slide title if exists
        if hasattr(slide, 'shapes'):
            for shape in slide.shapes:
                try:
                    shape_content = self._process_shape(shape, stats)
                    if shape_content:
                        content_parts.append(shape_content)
                except Exception as e:
                    print(f"Warning: Error processing shape: {e}")
                    stats['other_objects'] += 1
                    continue
        
        return {
            'combined_text': '\n\n'.join(content_parts),
            'stats': stats
        }
    
    def _process_shape(self, shape, stats: Dict[str, int]) -> str:
        """Process individual shape and extract relevant content"""
        content_parts = []
        
        try:
            # 1. Handle Tables (highest priority for structured content)
            if self._has_table(shape):
                try:
                    table_text = self._extract_table_text(shape.table)
                    if table_text:
                        content_parts.append(f"Table:\n{table_text}")
                        stats['tables'] += 1
                except Exception:
                    pass  # Skip table extraction if it fails
            
            # 2. Handle Charts
            elif self._has_chart(shape):
                try:
                    chart_text = self._extract_chart_text(shape.chart)
                    if chart_text:
                        content_parts.append(f"Chart:\n{chart_text}")
                        stats['charts'] += 1
                except Exception:
                    pass  # Skip chart extraction if it fails
            
            # 3. Handle Images and Pictures (check by shape type)
            elif self._is_image_shape(shape):
                image_info = self._extract_image_info(shape)
                if image_info:
                    content_parts.append(f"Image: {image_info}")
                    stats['images'] += 1
            
            # 4. Handle Group Shapes (recursive processing)
            elif self._is_group_shape(shape):
                group_content = self._extract_group_content(shape, stats)
                if group_content:
                    content_parts.append(f"Group: {group_content}")
            
            # 5. Handle Text Boxes and Shapes with text frames
            elif hasattr(shape, 'text_frame') and shape.text_frame:
                text_frame_content = self._extract_text_frame(shape.text_frame)
                if text_frame_content:
                    content_parts.append(f"Text Frame: {text_frame_content}")
            
            # 6. Handle SmartArt and other special shapes
            elif self._is_special_shape(shape):
                special_content = self._extract_special_shape_content(shape)
                if special_content:
                    content_parts.append(f"Special Object: {special_content}")
                    stats['other_objects'] += 1
            
            # 7. Handle Regular Text Content (fallback)
            elif hasattr(shape, 'text') and shape.text and shape.text.strip():
                content_parts.append(f"Text: {shape.text.strip()}")
            
            # 8. Handle Other Shape Types
            else:
                other_content = self._extract_other_shape_content(shape)
                if other_content:
                    content_parts.append(f"Shape: {other_content}")
                    stats['other_objects'] += 1
        
        except Exception as e:
            print(f"Warning: Error in shape processing: {e}")
            stats['other_objects'] += 1
        
        return '\n'.join(content_parts) if content_parts else ""
    
    def _is_image_shape(self, shape) -> bool:
        """Check if shape is an image/picture"""
        try:
            # Method 1: Check for image attribute
            if hasattr(shape, 'image'):
                return True
            
            # Method 2: Check shape type
            if hasattr(shape, 'shape_type'):
                # Use numeric constants instead of enum to avoid import issues
                PICTURE_TYPE = 13  # MSO_SHAPE_TYPE.PICTURE
                return shape.shape_type == PICTURE_TYPE
            
            return False
        except Exception:
            return False
    
    def _is_group_shape(self, shape) -> bool:
        """Check if shape is a group"""
        try:
            if hasattr(shape, 'shape_type'):
                GROUP_TYPE = 6  # MSO_SHAPE_TYPE.GROUP
                return shape.shape_type == GROUP_TYPE
            return False
        except Exception:
            return False
    
    def _is_special_shape(self, shape) -> bool:
        """Check if shape is SmartArt or other special object"""
        try:
            if hasattr(shape, 'shape_type'):
                # Common special shape types (using numeric constants)
                SMART_ART_TYPE = 15  # Approximate value for SmartArt
                MEDIA_TYPE = 16      # Media objects
                OLE_OBJECT_TYPE = 7  # Embedded objects
                
                return shape.shape_type in [SMART_ART_TYPE, MEDIA_TYPE, OLE_OBJECT_TYPE]
            return False
        except Exception:
            return False
    
    def _has_table(self, shape) -> bool:
        """Safely check if shape contains a table"""
        try:
            return hasattr(shape, 'table') and shape.table is not None
        except Exception:
            return False
    
    def _has_chart(self, shape) -> bool:
        """Safely check if shape contains a chart"""
        try:
            return hasattr(shape, 'chart') and shape.chart is not None
        except Exception:
            return False
    
    def _extract_special_shape_content(self, shape) -> str:
        """Extract content from special shapes like SmartArt"""
        try:
            content_parts = []
            
            # Try to get any text content
            if hasattr(shape, 'text') and shape.text:
                content_parts.append(f"Text: {shape.text.strip()}")
            
            # Try text frame
            if hasattr(shape, 'text_frame') and shape.text_frame:
                text_frame_content = self._extract_text_frame(shape.text_frame)
                if text_frame_content:
                    content_parts.append(f"Content: {text_frame_content}")
            
            # Get shape type info
            if hasattr(shape, 'shape_type'):
                content_parts.append(f"Type: {shape.shape_type}")
            
            return ', '.join(content_parts) if content_parts else "Special object"
        except Exception as e:
            print(f"Warning: Error extracting special shape: {e}")
            return "Special object"
    
    def _extract_text_frame(self, text_frame) -> str:
        """Extract text from text frame"""
        try:
            if hasattr(text_frame, 'text') and text_frame.text:
                return text_frame.text.strip()
            elif hasattr(text_frame, 'paragraphs'):
                paragraphs = []
                for paragraph in text_frame.paragraphs:
                    if hasattr(paragraph, 'text') and paragraph.text:
                        paragraphs.append(paragraph.text.strip())
                return '\n'.join(paragraphs) if paragraphs else ""
        except Exception as e:
            print(f"Warning: Error extracting text frame: {e}")
        return ""
    
    def _extract_image_info(self, shape) -> str:
        """Extract basic information about images"""
        try:
            info_parts = []
            if hasattr(shape, 'name') and shape.name:
                info_parts.append(f"Name: {shape.name}")
            
            # Try to get alt text or description
            if hasattr(shape, 'element') and hasattr(shape.element, 'get'):
                alt_text = shape.element.get('alt', '')
                if alt_text:
                    info_parts.append(f"Alt text: {alt_text}")
            
            return ', '.join(info_parts) if info_parts else "Image object"
        except Exception as e:
            print(f"Warning: Error extracting image info: {e}")
            return "Image object"
    

    
    def _extract_group_content(self, group_shape, stats: Dict[str, int]) -> str:
        """Extract content from grouped shapes"""
        try:
            group_parts = []
            if hasattr(group_shape, 'shapes'):
                for shape in group_shape.shapes:
                    shape_content = self._process_shape(shape, stats)
                    if shape_content:
                        group_parts.append(shape_content)
            return '\n'.join(group_parts) if group_parts else ""
        except Exception as e:
            print(f"Warning: Error extracting group content: {e}")
            return "Group object"
    
    def _extract_other_shape_content(self, shape) -> str:
        """Extract content from other shape types"""
        try:
            content_parts = []
            
            # Try to get shape name or any text content
            if hasattr(shape, 'name') and shape.name:
                content_parts.append(f"Shape name: {shape.name}")
            
            # Try to get any text content from the shape
            if hasattr(shape, 'text') and shape.text and shape.text.strip():
                content_parts.append(f"Text: {shape.text.strip()}")
            
            # Try to get shape type information
            if hasattr(shape, 'shape_type'):
                content_parts.append(f"Type: {shape.shape_type}")
            
            return ', '.join(content_parts) if content_parts else ""
        except Exception as e:
            print(f"Warning: Error extracting other shape content: {e}")
            return ""
    
    def _extract_table_text(self, table) -> str:
        """Extract text content from PPTX table"""
        try:
            table_rows = []
            for row in table.rows:
                row_cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip() if cell.text else ""
                    row_cells.append(cell_text)
                table_rows.append(" | ".join(row_cells))
            return "\n".join(table_rows)
        except Exception as e:
            print(f"Error extracting table text: {e}")
            return ""
    
    def _extract_chart_text(self, chart) -> str:
        """Extract text content from PPTX chart"""
        try:
            chart_parts = []
            
            # Try to extract chart title
            if hasattr(chart, 'chart_title') and chart.chart_title and hasattr(chart.chart_title, 'text_frame'):
                try:
                    chart_parts.append(f"Title: {chart.chart_title.text_frame.text}")
                except:
                    pass
            
            # Try to extract series data
            if hasattr(chart, 'plots'):
                for plot in chart.plots:
                    if hasattr(plot, 'series'):
                        for series in plot.series:
                            try:
                                if hasattr(series, 'name') and series.name:
                                    chart_parts.append(f"Series: {series.name}")
                            except:
                                pass
            
            return "\n".join(chart_parts) if chart_parts else ""
        except Exception as e:
            print(f"Error extracting chart text: {e}")
            return ""
    
    def _create_fallback_report(self, file_path: str) -> Dict[str, Any]:
        """Create minimal report when PPTX parsing fails"""
        filename = Path(file_path).name
        return {
            'metainfo': {
                'sha1_name': filename.rsplit('.', 1)[0],
                'filename': filename,
                'pages_amount': 0,
                'text_blocks_amount': 0,
                'tables_amount': 0,
                'pictures_amount': 0,
                'document_type': 'failed'
            },
            'content': {'pages': []},
            'tables': [],
            'pictures': []
        }

class ExcelParser:
    """Excel file parsing for all sheets and data"""
    
    def __init__(self):
        pass
    
    def parse_excel(self, file_path: str) -> Dict[str, Any]:
        """Parse Excel file and return structured report"""
        try:
            file_path_obj = Path(file_path)
            filename = file_path_obj.name
            
            # Read Excel file with appropriate engine
            if file_path_obj.suffix.lower() == '.xls':
                excel_data = pd.read_excel(file_path, sheet_name=None, engine='xlrd')
            else:
                excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            
            # Process sheets
            pages = []
            page_num = 1
            total_text_blocks = 0
            
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                
                # Convert DataFrame to readable text
                sheet_text = f"Sheet: {sheet_name}\n\n"
                
                # Add column headers
                if not df.columns.empty:
                    headers = " | ".join([str(col) for col in df.columns])
                    sheet_text += f"Headers: {headers}\n\n"
                
                # Add rows
                for row_idx, (idx, row) in enumerate(df.iterrows(), start=1):
                    row_text = " | ".join([str(val) if pd.notna(val) else "" for val in row.values])
                    sheet_text += f"Row {row_idx}: {row_text}\n"
                
                pages.append({
                    'page': page_num,
                    'text': sheet_text.strip()
                })
                
                page_num += 1
                total_text_blocks += len(df)
            
            # Create structured report
            report = {
                'metainfo': {
                    'sha1_name': filename.rsplit('.', 1)[0],
                    'filename': filename,
                    'pages_amount': len(pages),
                    'text_blocks_amount': total_text_blocks,
                    'tables_amount': len(excel_data),
                    'pictures_amount': 0,
                    'document_type': 'excel'
                },
                'content': {'pages': pages},
                'tables': [],
                'pictures': []
            }
            
            print(f"Successfully parsed Excel file: {filename} ({len(pages)} sheets)")
            return report
            
        except Exception as e:
            print(f"Error parsing Excel file {file_path}: {e}")
            return self._create_fallback_report(file_path)
    
    def _create_fallback_report(self, file_path: str) -> Dict[str, Any]:
        """Create minimal report when Excel parsing fails"""
        filename = Path(file_path).name
        return {
            'metainfo': {
                'sha1_name': filename.rsplit('.', 1)[0],
                'filename': filename,
                'pages_amount': 0,
                'text_blocks_amount': 0,
                'tables_amount': 0,
                'pictures_amount': 0,
                'document_type': 'failed'
            },
            'content': {'pages': []},
            'tables': [],
            'pictures': []
        }

class UnifiedDocumentParser:
    """Route documents to appropriate parser by file extension"""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.pptx_parser = PPTXParser()
        self.excel_parser = ExcelParser()
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse document using appropriate parser based on file extension"""
        try:
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix.lower()
            
            if file_extension == '.pdf':
                print(f"Parsing PDF : {file_path}")
                return self.pdf_parser.parse_pdf(file_path)
            elif file_extension in ['.pptx', '.ppt']:
                print(f"Parsing PPTX : {file_path}")
                return self.pptx_parser.parse_pptx(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                print(f"Parsing Excel : {file_path}")
                return self.excel_parser.parse_excel(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                return self._create_fallback_report(file_path)
            
        except Exception as e:
            print(f"Error parsing document {file_path}: {e}")
            return self._create_fallback_report(file_path)
    
    def _create_fallback_report(self, file_path: str) -> Dict[str, Any]:
        """Create minimal report when document parsing fails"""
        filename = Path(file_path).name
        return {
            'metainfo': {
                'sha1_name': filename.rsplit('.', 1)[0],
                'filename': filename,
                'pages_amount': 0,
                'text_blocks_amount': 0,
                'tables_amount': 0,
                'pictures_amount': 0,
                'document_type': 'failed'
            },
            'content': {'pages': []},
            'tables': [],
            'pictures': []
        }

class TextSplitter:
    """Document chunking with format-specific strategies"""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_document(self, document_data: Dict) -> List[Document]:
        """Split document using format-specific strategies"""
        doc_type = document_data['metainfo']['document_type']
        
        if doc_type == 'excel':
            return self._split_excel_document(document_data)
        elif doc_type == 'pptx':
            return self._split_pptx_document(document_data)
        else:
            # Use default splitting for PDF and other formats
            return self._split_default_document(document_data)
    
    def _split_excel_document(self, document_data: Dict) -> List[Document]:
        """Excel-specific chunking: preserve table structure and relationships"""
        chunks = []
        
        for page in document_data['content']['pages']:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # Parse sheet content
            sheet_chunks = self._split_excel_sheet(page_text, page['page'])
            chunks.extend(sheet_chunks)
        
        return chunks
    
    def _split_excel_sheet(self, sheet_text: str, page_num: int) -> List[Document]:
        """Split Excel sheet while preserving table structure"""
        documents = []
        lines = sheet_text.split('\n')
        
        if not lines:
            return documents
        
        # Extract sheet name and headers
        sheet_info = []
        data_rows = []
        headers = ""
        
        for i, line in enumerate(lines):
            if line.startswith('Sheet:'):
                sheet_info.append(line)
            elif line.startswith('Headers:'):
                headers = line
                sheet_info.append(line)
            elif line.startswith('Row'):
                data_rows.append(line)
        
        # Create chunks with preserved context
        context_header = '\n'.join(sheet_info)
        
        # Group rows into meaningful chunks (preserve relationships)
        row_groups = []
        current_group = []
        current_size = len(context_header)
        
        for row in data_rows:
            row_size = len(row)
            if current_size + row_size > self.chunk_size and current_group:
                row_groups.append(current_group.copy())
                current_group = [row]
                current_size = len(context_header) + row_size
            else:
                current_group.append(row)
                current_size += row_size
        
        if current_group:
            row_groups.append(current_group)
        
        # Create documents with full context
        for i, group in enumerate(row_groups):
            chunk_content = context_header + '\n\n' + '\n'.join(group)
            
            doc = Document(
                page_content=chunk_content.strip(),
                metadata={
                    "page": page_num,
                    "chunk": i + 1,
                    "total_chunks": len(row_groups),
                    "content_type": "excel_table",
                    "has_headers": bool(headers)
                }
            )
            documents.append(doc)
        
        return documents
    
    def _split_pptx_document(self, document_data: Dict) -> List[Document]:
        """PPTX-specific chunking: preserve slide structure and object relationships"""
        chunks = []
        
        for page in document_data['content']['pages']:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            slide_chunks = self._split_pptx_slide(page_text, page['page'])
            chunks.extend(slide_chunks)
        
        return chunks
    
    def _split_pptx_slide(self, slide_text: str, page_num: int) -> List[Document]:
        """Split PPTX slide while preserving object relationships"""
        documents = []
        
        # Parse slide content by object types
        content_blocks = self._parse_pptx_content_blocks(slide_text)
        
        if not content_blocks:
            return documents
        
        # Group related content blocks
        grouped_blocks = self._group_pptx_content(content_blocks)
        
        # Create chunks from grouped content
        for i, group in enumerate(grouped_blocks):
            chunk_content = '\n\n'.join(group['content'])
            
            if chunk_content.strip():
                doc = Document(
                    page_content=chunk_content.strip(),
                    metadata={
                        "page": page_num,
                        "chunk": i + 1,
                        "total_chunks": len(grouped_blocks),
                        "content_type": "pptx_slide",
                        "object_types": ",".join(group['types']) if group['types'] else ""
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _parse_pptx_content_blocks(self, slide_text: str) -> List[Dict]:
        """Parse PPTX slide into content blocks by type"""
        blocks = []
        lines = slide_text.split('\n\n')
        
        for line_group in lines:
            if not line_group.strip():
                continue
            
            # Identify content type
            content_type = 'text'
            if line_group.startswith('Table:'):
                content_type = 'table'
            elif line_group.startswith('Chart:'):
                content_type = 'chart'
            elif line_group.startswith('Image:'):
                content_type = 'image'
            elif line_group.startswith('Text Frame:'):
                content_type = 'text_frame'
            elif line_group.startswith('Group:'):
                content_type = 'group'
            
            blocks.append({
                'content': line_group.strip(),
                'type': content_type,
                'size': len(line_group)
            })
        
        return blocks
    
    def _group_pptx_content(self, content_blocks: List[Dict]) -> List[Dict]:
        """Group PPTX content blocks intelligently"""
        groups = []
        current_group = {'content': [], 'types': set(), 'size': 0}
        
        for block in content_blocks:
            # Keep tables and charts as separate chunks for better retrieval
            if block['type'] in ['table', 'chart'] and current_group['content']:
                groups.append({
                    'content': current_group['content'].copy(),
                    'types': list(current_group['types'])
                })
                current_group = {'content': [], 'types': set(), 'size': 0}
            
            # Add block to current group
            current_group['content'].append(block['content'])
            current_group['types'].add(block['type'])
            current_group['size'] += block['size']
            
            # If it's a table or chart, create separate chunk
            if block['type'] in ['table', 'chart']:
                groups.append({
                    'content': current_group['content'].copy(),
                    'types': list(current_group['types'])
                })
                current_group = {'content': [], 'types': set(), 'size': 0}
            # If group is getting too large, split it
            elif current_group['size'] > self.chunk_size:
                groups.append({
                    'content': current_group['content'].copy(),
                    'types': list(current_group['types'])
                })
                current_group = {'content': [], 'types': set(), 'size': 0}
        
        # Add remaining content
        if current_group['content']:
            groups.append({
                'content': current_group['content'],
                'types': list(current_group['types'])
            })
        
        return groups
    
    def _split_default_document(self, document_data: Dict) -> List[Document]:
        """Default chunking strategy for PDF and other formats"""
        chunks = []
        
        for page in document_data['content']['pages']:
            page_text = page.get('text', '')
            if page_text.strip():
                page_chunks = self._split_page_text(page_text, page['page'])
                chunks.extend(page_chunks)
        
        return chunks
    
    def _split_page_text(self, page_text: str, page_num: int) -> List[Document]:
        """Default page text splitting"""
        text_chunks = self.text_splitter.split_text(page_text)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "page": page_num,
                        "chunk": i + 1,
                        "total_chunks": len(text_chunks)
                    }
                )
                documents.append(doc)
        
        return documents

# =============================================================================
# FILE INPUT SYSTEM
# =============================================================================

def get_user_files() -> List[str]:
    """Get file paths from user input"""
    print("Please enter file paths for processing.")
    print("Supported formats: PDF , PPTX , XLS/XLSX ")
    print("Enter file paths one by one. Type 'done' when finished.....")
    
    files = []
    while True:
        file_path = input(f"File {len(files) + 1} (or 'done'): ").strip()
        
        if file_path.lower() == 'done':
            break
        
        if not file_path:
            continue
        
        if os.path.exists(file_path):
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.pdf', '.pptx', '.ppt', '.xls', '.xlsx']:
                files.append(file_path)
                print(f"Added: {file_path}")
            else:
                print(f"Unsupported file type: {file_ext}. Supported: .pdf, .pptx, .ppt, .xls, .xlsx")
        else:
            print(f"File not found: {file_path}")
    
    return files

# =============================================================================
# INGESTION SYSTEM
# =============================================================================

@dataclass
class GraphState:
    """State management for the RAG workflow graph"""
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
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    throughput_tokens_per_second: float = 0.0



def ingest_node(state: GraphState) -> GraphState:
    """Parse and ingest documents using unified parsing system"""
    print("Starting document ingestion...")
    
    parsed_reports = []
    parser = UnifiedDocumentParser()
    text_splitter = TextSplitter()
    
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
                    
                    # Add metadata to chunks
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
    
    # Flatten all chunks
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
    """Create vector embeddings for document chunks"""
    print("Creating vector embeddings...")
    
    if state.vectorstore is None and state.docs:
        vs_manager = VectorStoreManager()
        document_list = [doc for doc in state.docs if isinstance(doc, Document)]
        
        if document_list:
            vectorstore = vs_manager.create_vectorstore(document_list)
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

# =============================================================================
# PERFORMANCE TRACKING UTILITIES
# =============================================================================

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using OpenAI's official tiktoken library.
    This is the exact same method used by OpenAI for billing and API limits.
    """
    try:
        # Use model-specific encoding (most accurate)
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # If model not found, use the encoding that gpt-4o-mini uses
        try:
            encoding = tiktoken.get_encoding("o200k_base")  # GPT-4o encoding
            return len(encoding.encode(text))
        except Exception:
            # Last resort: use cl100k_base (GPT-4/ChatGPT encoding)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

def calculate_throughput(tokens: int, time_seconds: float) -> float:
    """Calculate tokens per second throughput"""
    if time_seconds <= 0:
        return 0.0
    return round(tokens / time_seconds, 2)

# =============================================================================
# RETRIEVAL SYSTEM
# =============================================================================

class VectorRetriever:
    """Vector-based document retrieval using embedding model and vector database"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, top_k: int = 30) -> List[Dict]:
        """Retrieve chunks using vector similarity search"""
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

class ParentPageAggregator:
    """Parent page retrieval with deduplication from chunk metadata"""
    
    def __init__(self, parsed_reports: List[Dict]):
        self.parsed_reports = parsed_reports
        self.page_content_map = self._build_page_content_map()
    
    def _build_page_content_map(self) -> Dict[int, str]:
        """Build mapping from page numbers to full page content"""
        page_map = {}
        for report in self.parsed_reports:
            for page_data in report['report']['content']['pages']:
                page_num = page_data['page']
                page_map[page_num] = page_data['text']
        return page_map
    
    def aggregate_to_parent_pages(self, chunk_results: List[Dict]) -> List[Dict]:
        """Extract parent pages from chunks and deduplicate"""
        seen_pages = set()
        parent_results = []
        
        for chunk_result in chunk_results:
            page_num = chunk_result['page']
            
            if page_num not in seen_pages:
                seen_pages.add(page_num)
                
                page_content = self.page_content_map.get(page_num, chunk_result['text'])
                
                parent_result = {
                    'text': page_content,
                    'page': page_num,
                    'distance': chunk_result['distance'],
                    'source_file': chunk_result['source_file'],
                    'document_type': chunk_result['document_type'],
                    'metadata': chunk_result['metadata']
                }
                parent_results.append(parent_result)
        
        return parent_results

class LLMReranker:
    """LLM-based reranking to adjust relevance scores for pages"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def rerank_documents(self, query: str, documents: List[Dict], 
                        documents_batch_size: int = 3, llm_weight: float = 0.7) -> List[Dict]:
        """Rerank pages using LLM with relevance score adjustment"""
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
                
                vector_similarity = max(0, 1 - doc.get('distance', 0.5))
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
        """Rerank a batch of texts using LLM"""
        if not texts:
            return []
        
        if len(texts) == 1:
            passages_text = f"\nPassage 1:\n{texts[0][:800]}...\n"
        else:
            passages_text = ""
            for i, text in enumerate(texts, 1):
                passages_text += f"\nPassage {i}:\n{text[:800]}...\n"
        
        prompt = f"""Rate each passage's relevance to answering the question on a scale of 0.0 to 1.0.

Question: {question}
{passages_text}

Consider:
- How directly the passage answers the question
- The quality and specificity of information
- Whether the passage provides the exact information needed

Respond with ONLY a JSON array of scores, e.g., [0.8, 0.2, 0.9]
"""
        
        try:
            response = self.llm.invoke(prompt)
            response_content = response.content if isinstance(response, BaseMessage) else str(response)
            
            if isinstance(response_content, str):
                json_match = re.search(r'\[[\d\.,\s]+\]', response_content)
                if json_match:
                    scores = json.loads(json_match.group())
                    while len(scores) < len(texts):
                        scores.append(0.3)
                    return scores[:len(texts)]
                else:
                    try:
                        scores = json.loads(response_content)
                        while len(scores) < len(texts):
                            scores.append(0.3)
                        return scores[:len(texts)]
                    except:
                        return [0.5] * len(texts)
            else:
                return [0.5] * len(texts)
        except Exception as e:
            _log.warning(f"Error in LLM reranking: {e}")
            return [0.5] * len(texts)

class HybridRetriever:
    """Complete retrieval system following the six-stage pipeline"""
    
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
        """Complete retrieval pipeline with vector search, parent aggregation and LLM reranking"""
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

def retrieval_node(state: GraphState) -> GraphState:
    """Execute complete retrieval pipeline"""
    print(f"Starting retrieval for question: {state.question[:100]}...")
    
    # Record retrieval start time
    retrieval_start = time.time()
    
    retriever = HybridRetriever(state.vectorstore, state.parsed_reports)
    
    reranked_results = retriever.retrieve(
        query=state.question,
        llm_reranking_sample_size=30,
        documents_batch_size=2,
        top_n=10,
        llm_weight=0.7
    )
    
    # Calculate retrieval time
    retrieval_time = time.time() - retrieval_start
    print(f"Retrieval completed with {len(reranked_results)} results in {retrieval_time:.3f}s")
    
    final_context = _assemble_context(reranked_results)
    
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
        retrieval_time=retrieval_time,
        generation_time=state.generation_time,
        total_time=state.total_time,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        throughput_tokens_per_second=state.throughput_tokens_per_second
    )

def _assemble_context(results: List[Dict]) -> str:
    """Format top pages into final context string with source identification"""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        page_number = result.get('page', '?')
        text = result['text']
        source_file = result.get('source_file', 'unknown')
        document_type = result.get('document_type', 'unknown')
        
        source_id = f"{document_type}_{source_file.replace('.', '_')}_page_{page_number}"
        
        header = f"[Source: {source_id}] - {document_type.upper()} file '{source_file}', Page {page_number}"
        context_parts.append(f"{header}\n\"\"\"\n{text}\n\"\"\"")
    
    return "\n\n---\n\n".join(context_parts)

# =============================================================================
# ANSWERING SYSTEM
# =============================================================================

def rag_node(state: GraphState) -> GraphState:
    """Generate structured answers using enhanced RAG system"""
    print("Generating structured answer...")
    
    # Record generation start time
    generation_start = time.time()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    enhanced_prompt = EnhancedRAGAnswerPrompt()
    context = state.final_context
    
    user_message = enhanced_prompt.user_prompt.format(
        context=context,
        question=state.question
    )
    
    full_prompt = f"{enhanced_prompt.system_prompt_with_schema}\n\n{user_message}"
    
    # Count input tokens
    input_tokens = count_tokens(full_prompt)
    
    try:
        response = llm.invoke(full_prompt)
        response_content = response.content if isinstance(response, BaseMessage) else str(response)
        response_str = str(response_content) if not isinstance(response_content, str) else response_content
        
        structured_answer = _parse_json_response(response_str, state.question)
        
        final_answer = str(structured_answer.get('final_answer', response_str))
        
        print(f"Answer confidence: {structured_answer.get('confidence_level', 'unknown')}")
        print(f"Sources used: {len(structured_answer.get('relevant_sources', []))}")
        
    except Exception as e:
        print(f"Warning: Error in structured answer generation: {e}")
        simple_prompt = f"Answer this question based on the context:\n\nContext: {context}\n\nQuestion: {state.question}"
        response = llm.invoke(simple_prompt)
        fallback_response = response.content if isinstance(response, BaseMessage) else str(response)
        final_answer = str(fallback_response)
        structured_answer = {
            "step_by_step_analysis": "Fallback answer due to parsing error",
            "reasoning_summary": "Error in structured processing",
            "relevant_sources": [],
            "confidence_level": "low",
            "final_answer": final_answer
        }
    
    # Calculate generation time and throughput
    generation_time = time.time() - generation_start
    output_tokens = count_tokens(final_answer)
    total_tokens = input_tokens + output_tokens
    total_time = state.retrieval_time + generation_time
    throughput = calculate_throughput(total_tokens, total_time)
    
    print(f"Generation completed in {generation_time:.3f}s")
    print(f"Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total")
    print(f"Throughput: {throughput} tokens/second")
    
    return GraphState(
        docs=state.docs,
        vectorstore=state.vectorstore,
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=final_answer,
        parsed_reports=state.parsed_reports,
        vector_results=state.vector_results,
        reranked_results=state.reranked_results,
        final_context=state.final_context,
        structured_answer=structured_answer,
        start_time=state.start_time,
        retrieval_time=state.retrieval_time,
        generation_time=generation_time,
        total_time=total_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        throughput_tokens_per_second=throughput
    )

def _parse_json_response(response_text: str, question: str) -> Dict:
    """Parse JSON response with multiple fallback strategies"""
    try:
        parsed = json.loads(response_text)
        
        required_fields = {
            "step_by_step_analysis": "Analysis not available",
            "reasoning_summary": "Summary not available", 
            "relevant_sources": [],
            "confidence_level": "medium",
            "final_answer": "Answer not available"
        }
        
        for field, default in required_fields.items():
            if field not in parsed:
                parsed[field] = default
        
        return parsed
        
    except (json.JSONDecodeError, ValueError):
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'\{.*?\}'
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, response_text, re.DOTALL)
            if json_match:
                try:
                    extracted = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                    parsed = json.loads(extracted)
                    
                    required_fields = {
                        "step_by_step_analysis": "Analysis extracted from response",
                        "reasoning_summary": "Extracted response", 
                        "relevant_sources": [],
                        "confidence_level": "low",
                        "final_answer": parsed.get("final_answer", response_text[:500])
                    }
                    
                    for field, default in required_fields.items():
                        if field not in parsed:
                            parsed[field] = default
                    
                    return parsed
                except:
                    continue
        
        return {
            "step_by_step_analysis": f"Unable to parse structured analysis. Raw response: {response_text[:500]}...",
            "reasoning_summary": "Fallback parsing used due to malformed JSON",
            "relevant_sources": [],
            "confidence_level": "low",
            "final_answer": response_text[:1000]
        }

def log_node(state: GraphState) -> GraphState:
    """Log detailed metrics and results including performance data"""
    print("Logging enhanced metrics...")
    
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
    
    # Performance metrics
    performance_metrics = {
        "retrieval_time_s": round(state.retrieval_time, 3),
        "generation_time_s": round(state.generation_time, 3),
        "total_time_s": round(state.total_time, 3),
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "total_tokens": state.input_tokens + state.output_tokens,
        "throughput_tokens_per_second": state.throughput_tokens_per_second
    }
    
    log_entry = {
        "question": state.question,
        "answer": state.answer,
        "confidence_level": state.structured_answer.get('confidence_level', 'unknown'),
        "relevant_sources": state.structured_answer.get('relevant_sources', []),
        "reasoning_summary": state.structured_answer.get('reasoning_summary', ''),
        "retrieval_metrics": metrics,
        "performance_metrics": performance_metrics,
        "used_existing_vectordb": False  # Simplified: this info not tracked in new flow
    }
    
    df = pd.DataFrame([log_entry])
    log_file = "enhanced_rag_qa_log.csv"
    
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
        print(f"Created new log file: {log_file}")
    else:
        df.to_csv(log_file, mode="a", header=False, index=False)
        print(f"Appended to log file: {log_file}")
    
    # Print performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Retrieval Time: {performance_metrics['retrieval_time_s']}s")
    print(f"Generation Time: {performance_metrics['generation_time_s']}s")
    print(f"Total Processing Time: {performance_metrics['total_time_s']}s")
    print(f"Input Tokens: {performance_metrics['input_tokens']:,}")
    print(f"Output Tokens: {performance_metrics['output_tokens']:,}")
    print(f"Total Tokens: {performance_metrics['total_tokens']:,}")
    print(f"Throughput: {performance_metrics['throughput_tokens_per_second']} tokens/second")
    print("=" * 50)
    
    return state

# =============================================================================
# WORKFLOW ORCHESTRATION
# =============================================================================

# Simplified workflow without redundant vectorstore checking
init_workflow = StateGraph(GraphState)
init_workflow.add_node("ingest", ingest_node)
init_workflow.add_node("embed", embed_node)
init_workflow.set_entry_point("ingest")
init_workflow.add_edge("ingest", "embed")
init_workflow.add_edge("embed", END)
init_graph = init_workflow.compile()

# Query workflow remains the same
query_workflow = StateGraph(GraphState)
query_workflow.add_node("retrieval", retrieval_node)
query_workflow.add_node("rag", rag_node)
query_workflow.add_node("log", log_node)
query_workflow.set_entry_point("retrieval")
query_workflow.add_edge("retrieval", "rag")
query_workflow.add_edge("rag", "log")
query_workflow.add_edge("log", END)
query_graph = query_workflow.compile()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Enhanced RAG system with streamlined database management...")
    print("Initializing system...")
    
    # First, check if existing database is available
    print("Checking for existing vector database...")
    vs_manager = VectorStoreManager()
    
    if vs_manager.vectorstore_exists():
        # Use existing database
        try:
            vectorstore = vs_manager.load_existing_vectorstore()
            if vectorstore is not None:
                stats = vs_manager.get_vectorstore_stats(vectorstore)
                print(f" Found existing vector database: {stats}")
                docs = []  # Empty since we're using existing database
                parsed_reports = []  # Empty for existing database
                print(" Ready for questions........\n")
            else:
                raise Exception("Failed to load existing database")
        except Exception as e:
            print(f" Error loading existing database: {e}")
            print("Will create new database from documents...")
            vectorstore = None
    else:
        print(" No existing vector database found.")
        vectorstore = None
    
    # If no existing database, get files from user and create new database
    if vectorstore is None:
        print("\nPlease provide documents for processing:")
        user_files = get_user_files()
        
        if not user_files:
            print("No files provided. Exiting.")
            exit(1)
        
        print(f"\nProcessing {len(user_files)} documents...")
        
        # Create initial state and process documents
        initial_state = GraphState(
            docs=user_files,
            question="",
            skip_parsing=False  # Force processing since no existing database
        )
        
        try:
            state_after_embed = init_graph.invoke(initial_state)
            print(" Documents parsed and embedded successfully!")
            
            docs = state_after_embed["docs"]
            vectorstore = state_after_embed["vectorstore"]
            parsed_reports = state_after_embed["parsed_reports"]
            
            # Show parsing summary
            if parsed_reports:
                total_pages = sum(len(report['report']['content']['pages']) for report in parsed_reports)
                total_chunks = len(docs)
                print(f" Documents processed: {len(parsed_reports)}")
                print(f" Total pages/slides: {total_pages}")
                print(f" Text chunks created: {total_chunks}")
            
            print(" Ready for questions!\n")
            
        except Exception as e:
            print(f" Error during document processing: {e}")
            print("This might be due to:")
            print("  - Missing dependencies")
            print("  - Corrupted document files")
            print("  - File format issues")
            exit(1)
    
    # Main question-answer loop
    print("Enter your questions below. Type 'quit' to exit.")
    
    while True:
        question = input("\n" + "=" * 50 + "\nQuestion (or 'quit' to exit): ")
        
        if question.strip().lower() == "quit":
            print("Exiting RAG system...")
            break
        
        # Record start time for the question
        start_time = time.time()
        
        state = GraphState(
            docs=docs,
            vectorstore=vectorstore,
            question=question,
            parsed_reports=parsed_reports,
            start_time=start_time
        )
        
        result = query_graph.invoke(state)
        
        print("\n" + "=" * 50)
        print("ENHANCED RAG RESULTS")
        print("=" * 50)
        print(f"Confidence: {result['structured_answer'].get('confidence_level', 'unknown')}")
        print(f"Documents Retrieved: {len(result.get('reranked_results', []))}")
        
        if result.get('reranked_results'):
            scores = [r.get('combined_score', 0) for r in result['reranked_results']]
            if scores:
                print(f"Score Range: {min(scores):.3f} - {max(scores):.3f}")
        
        print(f"\n--- Reasoning Summary ---")
        print(result['structured_answer'].get('reasoning_summary', 'No summary available'))
        
        print(f"\n--- Step-by-Step Analysis ---")
        analysis = result['structured_answer'].get('step_by_step_analysis', 'No analysis available')
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
        
        print(f"\n--- Relevant Sources ---")
        sources = result['structured_answer'].get('relevant_sources', [])
        if sources:
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")
        else:
            print("  No specific sources identified")
        
        print(f"\n" + "=" * 20 + " FINAL ANSWER " + "=" * 20)
        print(result["answer"])
        print("=" * 55)
