import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from pathlib import Path
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding  # Contoh impor, sesuaikan dengan pustaka Anda


from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from qdrant_client import QdrantClient
from llama_index.core.tools import BaseTool, FunctionTool


import sys
import logging
import requests
from typing import Optional
import nest_asyncio


from mixedbread_ai.client import MixedbreadAI
from sentence_transformers.util import cos_sim

import sys

import logging
import requests
from typing import Optional

nest_asyncio.apply()

# CONTEXT_PROMPT = """You are an expert system with knowledge of interview questions.
# These are documents that may be relevant to user question:\n\n
# {context_str}
# If you deem this piece of information is relevant, you may use it to answer user. 
# Else then you can say that you DON'T KNOW."""

# CONDENSE_PROMPT = """
# """

class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="mxbai-embed-large:latest", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = self.load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(_arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434",
            model_name=embedding_model)
        Settings.system_prompt = """
                                You are a multi-lingual expert system who has knowledge, based on 
                                real-time data. You will always try to be helpful and try to help them 
                                answering their question. If you don't know the answer, say that you DON'T
                                KNOW. Your main purpose is to answer questions about layanan keadaan darurat. 
                                Give full explanations according to ur knowledge, pdf, and web scrapping information
                                provided. Dont ever say where you got the information from though such as the pdf, web, or anything, just act as if u know.
                                """

        return Settings

    @st.cache_resource(show_spinner=False)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            # Read & load document from folder
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            client = QdrantClient(
                url="https://44efc1d5-e7fd-44bb-826e-7bdd3fe4745b.europe-west3-0.gcp.cloud.qdrant.io:6333", 
                api_key="DJ-djE9fDppvXGzZR_VuTWxW4AgYn6iYTgwVY4hBODPoiRg1UvlLGg" ,
            )
            vector_store = QdrantVectorStore(client=client, collection_name="Documents")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )
        # return index.as_chat_engine(chat_mode="condense_plus_context", chat_store_key="chat_history", memory=self.memory, verbose=True)




# Main Program
st.title("Layanan Keadaan Darurat")
chatbot = Chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Layanan Keadaan Darurat siap melayani segala pertanyaan!😁"}
    ]

print(chatbot.chat_store.store)



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chatbot.set_chat_history(st.session_state.messages)

# React to user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = chatbot.chat_engine.chat(prompt)
        st.markdown(response.response)

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response.response})
    
# emb_60296f7fa631ac1c42a4dd98be85e3f1e473faeb90fff0e3