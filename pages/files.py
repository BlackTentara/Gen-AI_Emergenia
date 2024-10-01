import os
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.readers.json import JSONReader
from llama_index.readers.file import UnstructuredReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding


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
            ".pdf":UnstructuredReader(),
            ".json":JSONReader(),
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
    st.session_state.client = QdrantClient(url=st.secrets["qdrant"]["connection_url"], api_key=st.secrets["qdrant"]["api_key"])
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