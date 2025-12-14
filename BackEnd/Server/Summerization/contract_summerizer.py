import re
from typing import List, Dict
from langchain_ollama import OllamaLLM
from .risk_analyzer import RiskAnalyzer
import torch

class ContinuousContractSummarizer:
    def __init__(self, model: str = "llama3.3"):
        self.llm = OllamaLLM(model=model)
        self.risk_analyzer = RiskAnalyzer()
        print(f"Risk Analyzer device: CPU")
        if torch.cuda.is_available():
            print(f"LLM will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("LLM will use CPU")

    def split_contract(self, text: str) -> List[str]:
        sections = re.split(r'\n\s*(ARTICLE|SECTION)\s+[IVXLCDM]+\s*\n', text)
        if len(sections) > 1:
            return sections
        return re.split(r'\n\d+\.\s+[A-Z][A-Z\s]+\n', text) or [text]

    def generate_prompt(self, text: str, risk_info: Dict[str, str] = None, is_final: bool = False) -> str:
        risk_context = ""
        if risk_info:
            risk_context = f"""
            RISK ANALYSIS RESULTS:
            - Risk Level: {risk_info['level']}
            - Risk Category: {risk_info['category']}
            
            IMPORTANT: Pay special attention to high-risk sections and highlight any terms related to the identified risk category.
            """
        
        if is_final:
            return f"""
            {risk_context}
            SUMMARIZE THIS CONTRACT SECTION CLEARLY AND CONCISELY:
            RULES:
            - Keep the summary in natural language (no bullet points).
            - Highlight IMPORTANT INFO using bold (e.g., Party A, $10,000, terminate, 30 days).
            - Pay special attention to risk-related terms based on the risk analysis above.
            - Focus on:
            • Who is involved  
            • What must be done  
            • When deadlines/payments happen  
            • How to exit/terminate  
            • Any identified risks
            - Include only essential clauses (financial, legal obligations, dates, risks, termination).
            - Remove boilerplate, generic text, and formalities.
            INPUT:
            {text}
            Output: A short, readable paragraph summary with bold keywords.
            IF THE TEXT WAS NOT A CONTRACT OR AGREEMENT, RETURN:"I DO NOT UNDERSTAND THIS TEXT. IT IS NOT A CONTRACT OR AGREEMENT."
            """
        else:
            return f"""
            {risk_context}
            EXTRACT KEY LEGAL ELEMENTS FROM THE FOLLOWING CONTRACT TEXT:
            RULES:
            - Identify and capture:
            • Involved parties  
            • Payment details (amounts, due dates)  
            • Termination conditions  
            • Key obligations and responsibilities  
            • Risks or conditions (especially those matching the identified risk category)
            - Keep monetary values, deadlines, durations, and clause names exactly as-is
            - Highlight the intent without including generic legal boilerplate
            - Return a CLEAN, minimal version of the contract's core logic
            TEXT TO ANALYZE:
            {text}
            Output: Clean, reduced terms ready for summary.
            IF THE TEXT WAS NOT A CONTRACT OR AGREEMENT, RETURN:"I DO NOT UNDERSTAND THIS TEXT. IT IS NOT A CONTRACT OR AGREEMENT."
            """

    def summarize(self, contract_text: str):
        chunks = self.split_contract(contract_text)
        extracted_terms = []
        
        for chunk in chunks:
            risk_level, risk_category = self.risk_analyzer.analyze_risk(chunk)
            print(f"\n[Risk Analysis] Level: {risk_level}, Category: {risk_category}")
            risk_info = {'level': risk_level, 'category': risk_category}
            prompt = self.generate_prompt(chunk, risk_info)
            
            print("\n[Processing Chunk]")
            chunk_extract = ""
            for chunk_output in self.llm.stream(prompt):
                print(chunk_output, end="", flush=True)
                chunk_extract += chunk_output
            
            extracted_terms.append({
                'text': chunk_extract,
                'risk_level': risk_level,
                'risk_category': risk_category
            })
        
        combined_terms = "\n".join([f"[Risk: {item['risk_level']} - {item['risk_category']}]\n{item['text']}" 
                                for item in extracted_terms])
        
        overall_risk_level, overall_risk_category = self.risk_analyzer.analyze_risk(contract_text)
        final_risk_info = {
            'level': overall_risk_level,
            'category': overall_risk_category
        }
        
        print("\n\n================ FINAL SUMMARY ==================")
        print(f"[Overall Risk Assessment] Level: {overall_risk_level}, Category: {overall_risk_category}\n")
        
        final_prompt = self.generate_prompt(combined_terms, final_risk_info, is_final=True)
        for chunk in self.llm.stream(final_prompt):
            print(chunk, end="", flush=True)