import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from config import BASE_MODEL

class LLMLoader:
    def __init__(self, system_prompt, query_wrapper_prompt):
        self.system_prompt = system_prompt
        self.query_wrapper_prompt = query_wrapper_prompt
        self.base_model = BASE_MODEL
        self.tokenizer = None
        self.model = None

    def load_llm(self, device=None):
        print("Loading LLM...")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            #quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # 파이프라인 생성
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        # LangChain용 HuggingFacePipeline 생성
        llm = HuggingFacePipeline(pipeline=pipe)

        return llm