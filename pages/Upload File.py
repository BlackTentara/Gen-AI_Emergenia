import os
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from typing import Optional, Dict, List
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader
import json
import re


class JSONReader(BaseReader):
    """JSON reader.

    Reads JSON documents with options to help us out relationships between nodes.

    Args:
        levels_back (int): the number of levels to go back in the JSON tree, 0
          if you want all levels. If levels_back is None, then we just format the
          JSON and make each line an embedding

        collapse_length (int): the maximum number of characters a JSON fragment
          would be collapsed in the output (levels_back needs to be not None)
          ex: if collapse_length = 10, and
          input is {a: [1, 2, 3], b: {"hello": "world", "foo": "bar"}}
          then a would be collapsed into one line, while b would not.
          Recommend starting around 100 and then adjusting from there.

        is_jsonl (Optional[bool]): If True, indicates that the file is in JSONL format.
        Defaults to False.

        clean_json (Optional[bool]): If True, lines containing only JSON structure are removed.
        This removes lines that are not as useful. If False, no lines are removed and the document maintains a valid JSON object structure.
        If levels_back is set the json is not cleaned and this option is ignored.
        Defaults to True.
    """

    def __init__(
        self,
        levels_back: Optional[int] = None,
        collapse_length: Optional[int] = None,
        ensure_ascii: bool = False,
        is_jsonl: Optional[bool] = False,
        clean_json: Optional[bool] = True,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.levels_back = levels_back
        self.collapse_length = collapse_length
        self.ensure_ascii = ensure_ascii
        self.is_jsonl = is_jsonl
        self.clean_json = clean_json

    def load_data(
        self, input_file: str, extra_info: Optional[Dict] = {}
    ) -> List[Document]:
        """Load data from the input file."""
        with open(input_file, encoding="utf-8") as f:
            load_data = []
            if self.is_jsonl:
                for line in f:
                    load_data.append(json.loads(line.strip()))
            else:
                load_data = [json.load(f)]

            documents = []
            for data in load_data:
                if self.levels_back is None and self.clean_json is True:
                    # If levels_back isn't set and clean json is set,
                    # remove lines containing only formatting, we just format and make each
                    # line an embedding
                    json_output = json.dumps(
                        data, indent=0, ensure_ascii=self.ensure_ascii
                    )
                    lines = json_output.split("\n")
                    useful_lines = [
                        line for line in lines if not re.match(r"^[{}\[\],]*$", line)
                    ]
                    documents.append(
                        Document(text="\n".join(useful_lines), metadata=extra_info)
                    )

                elif self.levels_back is None and self.clean_json is False:
                    # If levels_back isn't set  and clean json is False, create documents without cleaning
                    json_output = json.dumps(data, ensure_ascii=self.ensure_ascii)
                    documents.append(Document(text=json_output, metadata=extra_info))

                elif self.levels_back is not None:
                    # If levels_back is set, we make the embeddings contain the labels
                    # from further up the JSON tree
                    lines = [
                        *_depth_first_yield(
                            data,
                            self.levels_back,
                            self.collapse_length,
                            [],
                            self.ensure_ascii,
                        )
                    ]
                    documents.append(
                        Document(text="\n".join(lines), metadata=extra_info)
                    )
            return documents


def upload_files(files, path):
    files_path = []
    for file in files:
        try:
            save_path = os.path.join(path, file.name)
            if os.path.exists(save_path):
                st.warning("{} already exists!".format(file.name), icon="⚠")
            else:
                with open(save_path, "wb") as f:
                    f.write(file.getvalue())
                    files_path.append(file.name)
                    indexing_data(path, file.name)
                f.close()
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None

    files_path = ", ".join(files_path)
    if files_path:
        st.success("Successfully uploaded {}".format(files_path))
        st.rerun()


def display_files(path):
    file_list = [file for file in os.listdir(path) if not file.startswith(".")]
    delete_button = []

    for i, file in enumerate(file_list):
        with st.container(border=True):
            col1, col2, col3 = st.columns([9, 1.5, 1])
            with col1:
                st.write(f"📄 {file}")
            with col2:
                size = os.stat(os.path.join(path, file)).st_size
                st.write(f"{round(size / (1024 * 1024), 2)} MB")
            with col3:
                delete = st.button("🗑️", key="delete"+str(i))
                delete_button.append(delete)

    if True in delete_button:
        index = delete_button.index(True)
        os.remove(os.path.join(path, file_list[index]))
        st.toast(f"Successfully deleted {file_list[index]}", icon="❌")
        del file_list[index]
        st.rerun()

def indexing_data(path, file_name):
    file_path = os.path.join(path, file_name)
    print(file_path)
    with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes, don't turn off or switch pages!"):
        # Read & load document
        reader = SimpleDirectoryReader(input_files=[file_path], file_extractor={
            ".json": JSONReader(),
        })
        documents = reader.load_data()

        # Create Collection
        create_collection(documents, "All Documents", st.session_state.client)


def create_collection(documents, collection_name, client):
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index


def reindex(client, path):
    # Delete collection
    client.delete_collection(collection_name="All Documents")

    # Reindexing
    file_list = [file for file in os.listdir(path) if not file.startswith(".")]
    for file in file_list:
        indexing_data(path, file)
    st.success("Successfully reset index")


path = "docs/"

# Create Qdrant client & store
if "chatbot" not in st.session_state:
    st.session_state.client = QdrantClient(url="https://44efc1d5-e7fd-44bb-826e-7bdd3fe4745b.europe-west3-0.gcp.cloud.qdrant.io:6333", api_key="DJ-djE9fDppvXGzZR_VuTWxW4AgYn6iYTgwVY4hBODPoiRg1UvlLGg")
else:
    chatbot = st.session_state.chatbot
    st.session_state.client = chatbot.client
Settings.embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large", cache_dir="../fastembed_cache")

tab1, tab2 = st.tabs(["Upload", "Management"])
with tab1:
    st.header("Upload Files")
    with st.form("Upload", clear_on_submit=True):
        files = st.file_uploader("Document:", accept_multiple_files=True)
        upload_button = st.form_submit_button("Upload")

    if upload_button:
        if files is not None:
            upload_files(files, path)
        else:
            st.warning("Make sure your file is uploaded before submitting!")

with tab2:
    st.header("File List")
    reset_vector = st.button("🔄 Re-index")
    if reset_vector:
        reindex(st.session_state.client, path)
    st.warning("Changes will take effect immediately!")
    display_files(path)


