"""
Answer generation system for RAG.
"""
import json
import re
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema.messages import BaseMessage
from prompts import RAGAnswerPrompt
from utils import count_tokens, calculate_throughput


class AnswerGenerator:
    """Structured answer generation using LLM."""
    
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = RAGAnswerPrompt()
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate structured answer with metrics."""
        generation_start = time.time()
        
        user_message = self.prompt.user_prompt.format(
            context=context,
            question=question
        )
        
        full_prompt = f"{self.prompt.system_prompt_with_schema}\n\n{user_message}"
        input_tokens = count_tokens(full_prompt)
        
        try:
            response = self.llm.invoke(full_prompt)
            response_content = response.content if isinstance(response, BaseMessage) else str(response)
            response_str = str(response_content) if not isinstance(response_content, str) else response_content
            
            structured_answer = self._parse_json_response(response_str, question)
            final_answer = str(structured_answer.get('final_answer', response_str))
            
        except Exception as e:
            print(f"Warning: Error in structured answer generation: {e}")
            final_answer = self._generate_fallback_answer(question, context)
            structured_answer = self._create_fallback_structure(final_answer)
        
        generation_time = time.time() - generation_start
        output_tokens = count_tokens(final_answer)
        total_tokens = input_tokens + output_tokens
        throughput = calculate_throughput(total_tokens, generation_time)
        
        return {
            'final_answer': final_answer,
            'structured_answer': structured_answer,
            'generation_time': generation_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'throughput': throughput
        }
    
    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate simple fallback answer when structured generation fails."""
        simple_prompt = f"Answer this question based on the context:\n\nContext: {context}\n\nQuestion: {question}"
        response = self.llm.invoke(simple_prompt)
        return response.content if isinstance(response, BaseMessage) else str(response)
    
    def _create_fallback_structure(self, answer: str) -> Dict[str, Any]:
        """Create fallback structured answer when parsing fails."""
        return {
            "step_by_step_analysis": "Fallback answer due to parsing error",
            "reasoning_summary": "Error in structured processing",
            "relevant_sources": [],
            "confidence_level": "low",
            "final_answer": answer
        }
    
    def _parse_json_response(self, response_text: str, question: str) -> Dict:
        """Parse JSON response with multiple fallback strategies."""
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
