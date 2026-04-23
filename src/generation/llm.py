"""
LLM wrapper — supports OpenAI and Ollama (local).

Why support both:
- OpenAI (GPT-4o-mini): best quality, costs money, data leaves your machine
- Ollama (Llama 3.1 8B): free, runs locally, private, good enough for most RAG
- Having both means the project works with OR without an API key
"""

import os
from typing import Optional

import yaml

from src.generation.prompt import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


class LLMEngine:
    """Unified LLM interface for RAG generation."""

    def __init__(self, config_path: str = "configs/local.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.llm_config = config["llm"]
        # Use the centralised prompt; fall back to config if overridden there
        self.system_prompt = config["generation"].get("system_prompt", RAG_SYSTEM_PROMPT)
        self.provider = self.llm_config["provider"]

        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either:\n"
                "  1. export OPENAI_API_KEY='your-key'\n"
                "  2. Use local config: --config configs/local.yaml"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = self.llm_config["model"]
        print(f"  LLM: OpenAI {self.model}")

    def _init_ollama(self):
        """Initialize Ollama client."""
        from openai import OpenAI

        base_url = self.llm_config.get("base_url", "http://localhost:11434")
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama doesn't need a real key
        )
        self.model = self.llm_config["model"]
        print(f"  LLM: Ollama {self.model} at {base_url}")

    def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[list] = None,
    ) -> str:
        """
        Generate an answer using retrieved context.

        Args:
            query: user's question
            context: formatted context from retrieval
            conversation_history: optional list of previous messages

        Returns:
            LLM's answer as string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        user_message = RAG_USER_TEMPLATE.format(context=context, question=query)

        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.llm_config.get("temperature", 0.1),
            max_tokens=self.llm_config.get("max_tokens", 1024),
        )

        return response.choices[0].message.content
