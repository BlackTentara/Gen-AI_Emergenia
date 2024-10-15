import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr 
from llama_index.core import PromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
import httpx

import nest_asyncio

DEFAULT_EMBED_BATCH_SIZE=64

class OllamaEmbedding(BaseEmbedding):
    """Class for Ollama embeddings."""

    base_url: str = Field(description="Base url the model is hosted by Ollama")
    model_name: str = Field(description="The Ollama model to use.")
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    ollama_additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Ollama API."
    )

    _client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        ollama_additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            ollama_additional_kwargs=ollama_additional_kwargs or {},
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = httpx.Client(base_url=self.base_url)
        self._async_client = httpx.AsyncClient(base_url=self.base_url)

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self.aget_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self.aget_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await asyncio.gather(
            *[self.aget_general_text_embedding(text) for text in texts]
        )

    def get_general_text_embedding(self, texts: str) -> List[float]:
        """Get Ollama embedding."""
        result = self._client.embeddings(
            model=self.model_name, prompt=texts, options=self.ollama_additional_kwargs
        )
        return result["embedding"]

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Ollama embedding."""
        result = await self._async_client.embeddings(
            model=self.model_name, prompt=prompt, options=self.ollama_additional_kwargs
        )
        return result["embedding"]

nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

system_prompt = """
You are a multi-lingual advisor expert specializing in emergency situations. You will assist users in finding information about different types of emergencies. If you don't know the answer, say that you DON'T KNOW.

Your primary job is to show users information about emergency incidents by querying the emergency database based on the date given

When a user asks about emergencies on a date, you should provide every information about incident on the date given.

When a user asks about emergencies, you should provide the following information:
1. Name of the Incident (Nama)
2. Quantity (Jumlah Insiden)

Here is a short example:
User: Can you find information on emergency incidents on 14 October 2024?
Assistant: Sure! Here are the results for 2024-10-14 emergency incidents:
- Name: Flood
- Quantity: 13

Feel free to ask about specific emergency types or incidents!
"""


react_system_header_str = """\

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
```

Please ALWAYS start with a Thought.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)

import sys

import logging
import requests

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", system_prompt=system_prompt, temperature=0)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

# Main Program
st.title("RAG Test")

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Mau tahu apa tentang kejadian darurat hari ini?"}
    ]

# Declare Tools
# function tools

from datetime import datetime

async def search_informasi_kejadian_darurat(tanggal: str = None) -> str:
    """Searches the Keadaan Darurat database for matching keadaan darurat entries."""

    if not tanggal:
        tanggal = datetime.now().strftime('%Y-%m-%d')
    else:
        # Validasi format tanggal input dari pengguna (harus YYYY-MM-DD)
        try:
            # Mencoba mengonversi input string ke datetime untuk memeriksa validitas formatnya
            datetime.strptime(tanggal, '%Y-%m-%d')
        except ValueError:
            return "Error: Format tanggal tidak valid. Harus dalam format YYYY-MM-DD."
        
    r = requests.get("https://layanan112.kominfo.go.id/get/emergency_lists/{today}")
    
    data = r.json()
    output = f"# Data Search Results for {tanggal}\n"

    for d in data:
        output += f"""
Tipe  : {d['id']}
Nama  : {d['name']}
Jumlah : {d['y']}

"""
    return output


# async def get_studi_independen_activity_detail(date: str) -> str:
#     """Provides detailed information regarding the studi independen activity."""
#     r = requests.get(f"https://layanan112.kominfo.go.id/get/emergency_lists/{date}")

#     data = r.json()["data"]
#     return f"""
# Activity Name: {data["name"]}
# Activity Type: {data["activity_type"]}
# Location: {data["location"]}

# Description:
# {data["description"]}

# Requirements:
# {data["requirement"]}
#     """


search_informasi_kejadian_darurat = FunctionTool.from_defaults(async_fn=search_informasi_kejadian_darurat)
# get_studi_independen_activity_detail_tool = FunctionTool.from_defaults(async_fn=get_studi_independen_activity_detail)

tools = [search_informasi_kejadian_darurat]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Mau tahu apa tentang Breaking News hari ini?"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
    st.session_state.chat_engine = ReActAgent.from_tools(
        tools,
        chat_mode="react",
        verbose=True,
        memory=memory,
        react_system_prompt=react_system_prompt,
        # retriever=retriever,
        llm=Settings.llm
    )

    print(st.session_state.chat_engine.get_prompts())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream.response})
