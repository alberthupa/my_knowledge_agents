import os
import argparse
import re
from datetime import datetime
from azure.cosmos import exceptions
from src.vectors.cosmos_client import SimpleCosmosClient
from src.llms.basic_agent import BasicAgent

'''
https://chatgpt.com/c/6824f32a-2c70-800a-8080-e55fd2007674
'''


# Configuration variables
COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"  # Replace with your database name if different
CONTAINER_NAME = "knowledge-chunks"  # Replace with your container name if different
PARTITION_KEY_PATH = "/id"  # Replace with your partition key path if different


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    container_name=CONTAINER_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()

if cosmos_client.container_client:
    last_date = cosmos_client.get_last_newsletter_date()
    last_notes = cosmos_client.get_notes_from_day(last_date)
    last_notes_str = str(last_notes)

    message = f"""This is a content of last newslettters about AI: {last_notes_str}.
        What are the main topics in the text? Topics that can be presented as objects in ontology in knowledge graph about AI.
    """

    # Summarize it in bullers, telling what news is the most important: . Apply nice markdown formatting.


    llm_agent = BasicAgent()
    ai_response = llm_agent.get_text_response_from_llm(
        llm_model_input="gemini-2.0-flash-exp",  # Changed keyword argument
        messages=message,
        code_tag=None,
    )
    ai_response = ai_response["text_response"]
    print(ai_response)



'''
**Core AI Concepts & Technologies:**

*   **Large Language Models (LLMs):**  The central focus.  Sub-objects include:
    *   Model Architectures (MoE, Transformers, etc.)
    *   Model Families (Qwen, Llama, Gemini, Claude, DeepSeek, Grok, Sonar)
    *   Model Sizes (8B, 30B, 235B, etc.)
    *   Quantization Techniques (GGUF, AWQ, GPTQ, Dynamic Quantization)
    *   Training Data & Techniques (RL, Supervised Learning, Post-Training)
    *   Inference Optimization (Speculators, vLLM, Inference Endpoints)
    *   Tokenization (Byte-level, Traditional)
*   **Vision Language Models (VLMs):**  Models that process both images and text.
    *   GUI Agents
    *   Multimodal RAG
    *   Video LMs
    *   Smol Models
*   **AI Agents:** Autonomous systems that can perform tasks.
    *   Ambient Agents
    *   Code Agents
*   **AI Infrastructure:** The hardware and software needed to run AI.
    *   GPUs (NVIDIA, Intel Arc)
    *   TPUs
    *   Memory (VRAM, KV Cache)
    *   CUDA
    *   IPEX
    *   SYCL
*   **AI Engineering & Tooling:** Tools and techniques for building AI systems.
    *   DSPy
    *   KerasRS
    *   LangChain
    *   LlamaIndex
    *   Aider
    *   Cursor
    *   Windsurf
    *   CUTLASS
    *   CuTe DSL
    *   Torchtune
    *   Mojo
*   **AI Applications:** Specific uses of AI.
    *   Coding Assistants
    *   Medical Diagnosis
    *   Image Generation
    *   Video Generation
    *   Robotics
    *   Recommender Systems
    *   Document Structuring
    *   Gaslighting (as an experimental application)

**Evaluation & Benchmarking:**

*   **Benchmarks:** Standardized tests for evaluating AI performance.
    *   LMArena
    *   HealthBench
    *   GPQA
    *   MATH-500
    *   AIME24
    *   BrowseComp
    *   HLE
*   **Metrics:**  Specific measurements of AI performance (e.g., accuracy, speed).

**Industry & Community:**

*   **AI Companies:** Organizations involved in AI research and development.
    *   OpenAI
    *   Anthropic
    *   Google
    *   Meta
    *   Tencent
    *   DeepSeek
    *   Perplexity AI
    *   Cartesia
    *   PrimeIntellect
    *   Alibaba
    *   HuggingFace
    *   Unsloth AI
    *   Nous Research AI
*   **AI Researchers & Engineers:** People working in the field.
*   **AI Communities:** Online groups and forums for discussing AI.
    *   Reddit (various subreddits)
    *   Discord (various servers)
*   **AI Events:** Conferences, workshops, and meetups.
    *   AI Engineer World's Fair
    *   LangChain Interrupt
    *   Sequoia AI Ascent

**Ethical & Societal Implications:**

*   **AI Governance:**  The regulation and oversight of AI.
*   **AI Ethics:**  Moral considerations related to AI development and use.
*   **AI Bias:**  Unfair or discriminatory outcomes from AI systems.
*   **AI Impact on Work:**  How AI is changing the nature of jobs.
*   **Copyright:** Legal issues related to AI-generated content.

**Data & Resources:**

*   **Datasets:** Collections of data used to train AI models.
*   **Model Weights:** The parameters of a trained AI model.
*   **Technical Reports:** Documents describing the details of AI models and research.
*   **Open Source Projects:** AI-related software that is freely available.

This list provides a good starting point for building a comprehensive knowledge graph or ontology about AI based on the provided newsletter content.
'''