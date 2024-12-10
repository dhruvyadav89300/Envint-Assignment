import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.set_page_config(page_title="LLM PDF QA Tool", page_icon="/image.png")
st.title("PDF QA Tool")

################################################################################
# Load environment
################################################################################

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found")
    st.stop()

if not openai_api_key:
    st.error("OPENAI_API_KEY not found")
    st.stop()

@st.cache_resource
def initialize_llm(api_key: str):
    return ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")

llm = initialize_llm(groq_api_key)

PROJECTS_DIR = "projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

################################################################################
# Projects State
################################################################################

if "projects_state" not in st.session_state:
    st.session_state.projects_state = {}

################################################################################
# Project Management Related Functions
################################################################################

def list_projects():
    projects = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    return projects

def create_project(project_name: str):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path):
        os.makedirs(os.path.join(project_path, "pdfs"), exist_ok=True)

        # This helps achieve project isolation
        st.session_state.projects_state[project_name] = {
            "loader": None,
            "embeddings": None,
            "docs": None,
            "splitter": None,
            "documents": None,
            "vectorstore": None,
            "retrieval_chain": None
        }

        # I am setting the new project as the current project
        set_current_project(project_name)
        return True
    return False

def delete_project(project_name: str):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if os.path.exists(project_path):
        shutil.rmtree(project_path)

        # Remove that project's state
        if project_name in st.session_state.projects_state:
            del st.session_state.projects_state[project_name]

        # If the deleted project was the current project, unset it
        if "current_project" in st.session_state and st.session_state.current_project == project_name:
            del st.session_state.current_project
        return True
    return False

def set_current_project(project_name):
    st.session_state.current_project = project_name

################################################################################
# RAG
################################################################################

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
"""
)

def initialize_vector_store(project_name):
    # Check if the project state exists
    if project_name not in st.session_state.projects_state:
        st.error(f"Project '{project_name}' does not exist in session state.")
        return None

    project_path = os.path.join(PROJECTS_DIR, project_name, "pdfs")
    vectorstore_path = os.path.join(PROJECTS_DIR, project_name, "vectorstore")

    if os.path.exists(vectorstore_path):
        # Load existing vectorstore
        try:

            ####### ----------- I have set allow_dangerous_deserialization as true to load pickle file ----------- #######
            vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            st.session_state.projects_state[project_name]["vectorstore"] = vectorstore
            st.write(f"Loaded existing vector store for project '{project_name}'.")
        except Exception as e:
            st.error(f"Failed to load existing vector store: {e}")
            return None
    else:
        loader = PyPDFDirectoryLoader(project_path)
        st.session_state.projects_state[project_name]["loader"] = loader

        embeddings = OpenAIEmbeddings()
        st.session_state.projects_state[project_name]["embeddings"] = embeddings

        try:
            docs = loader.load()
            st.session_state.projects_state[project_name]["docs"] = docs
            st.write(f"Loaded {len(docs)} documents for project '{project_name}'.")
        except Exception as e:
            st.error(f"Failed to load documents: {e}")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.projects_state[project_name]["splitter"] = splitter
        try:
            documents = splitter.split_documents(docs)
            st.session_state.projects_state[project_name]["documents"] = documents
            st.write(f"Split documents into {len(documents)} chunks.")
        except Exception as e:
            st.error(f"Failed to split documents: {e}")
            return None

        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            # Save vectorstore locally
            vectorstore.save_local(vectorstore_path)
            st.session_state.projects_state[project_name]["vectorstore"] = vectorstore
            st.write(f"Created and saved vector store for project '{project_name}'.")

        except Exception as e:
            st.error(f"Failed to create or save vector store: {e}")
            return None

    # Retrieval Chain
    try:
        documents_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, documents_chain)
        st.session_state.projects_state[project_name]["retrieval_chain"] = retrieval_chain
        st.write(f"Created retrieval chain for project '{project_name}'.")
    except Exception as e:
        st.error(f"Failed to create retrieval chain: {e}")
        return None

    return vectorstore


def get_current_project_state():
    project_name = st.session_state.current_project
    return st.session_state.projects_state.get(project_name, None)

def generate_answer(user_input):
    project_state = get_current_project_state()
    if not project_state or not project_state["retrieval_chain"]:
        return "The vector store is not initialized for this project. Please upload PDFs first."
    try:
        response = project_state["retrieval_chain"].invoke({"input": user_input})
        return response
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

################################################################################
# Sidebar 
################################################################################

with st.sidebar:
    st.header("Project Management")

    # New Project xreation

    st.subheader("Create a New Project")
    new_project_name = st.text_input("New Project Name", "")
    if st.button("Create Project"):
        project_name = new_project_name.strip()
        if project_name:
            success = create_project(project_name)
            if success:
                st.success(f"Project '{project_name}' created and set as current project.")
            else:
                st.warning("Project already exists.")
        else:
            st.error("Please enter a valid project name.")

    projects = list_projects()

    # Project Selection

    if projects:
        st.subheader("Select a Project")
        try:
            selected_project = st.selectbox(
                "Projects",
                projects,
                index=projects.index(st.session_state.get("current_project", projects[0])) 
                if "current_project" in st.session_state else 0
            )
        except ValueError:
            selected_project = projects[0]
        
        if st.button("Use Project"):
            set_current_project(selected_project)
            st.success(f"Switched to project: {selected_project}")

    # Project Deletion

    if projects:
        st.subheader("Delete a Project")
        del_project = st.selectbox("Select a Project to Delete", projects)
        if st.button("Delete Project"):
            if "current_project" in st.session_state and del_project == st.session_state.current_project:
                st.warning("Cannot delete the currently selected project. Please switch to another project first.")
            else:
                success = delete_project(del_project)
                if success:
                    st.success(f"Project '{del_project}' deleted.")
                else:
                    st.error("Failed to delete the project.")
    else:
        st.write("No projects available. Create one above.")

################################################################################
# Main Page
################################################################################

if "current_project" in st.session_state:
    project_name = st.session_state.current_project
    st.subheader(f"Current Project: {project_name}")

    st.write("### Upload PDFs")
    pdf_files = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True, key=f"pdf_uploader_{project_name}")
    if pdf_files:
        project_pdf_path = os.path.join(PROJECTS_DIR, project_name, "pdfs")
        # Ensure directory exists
        os.makedirs(project_pdf_path, exist_ok=True)

        for pdf_file in pdf_files:
            # Save the file locally
            save_path = os.path.join(project_pdf_path, pdf_file.name)
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            st.write(f"Saved '{pdf_file.name}' to {save_path}")

        # Initialize vector store 
        
        with st.spinner("Processing PDFs and initializing vector store..."):
            vectorstore = initialize_vector_store(project_name)
        if vectorstore:
            st.success("PDFs uploaded and vector store initialized.")

    st.write("### Ask a Question")
    user_question = st.text_input("Enter your question here:", key=f"user_question_{project_name}")
    if st.button("Ask"):
        if user_question.strip():
            with st.spinner("Generating answer..."):
                answer = generate_answer(user_question)
            st.write("**Answer:**")
            st.write(answer if isinstance(answer, str) else answer.get('answer', "No answer found."))
        else:
            st.error("Please enter a question before asking.")
else:
    st.write("No project selected. Please create or select a project from the sidebar.")
