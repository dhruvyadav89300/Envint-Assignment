# Envint-Assignment


---

## **Introduction**

This project implements a web application that utilizes a Large Language Model (LLM) to extract key information from PDFs. The application enables users to manage multiple projects, each with its own set of PDFs, ensuring that questions are answered solely based on the PDFs uploaded for the active project. Key features include:

- **Project Management**: Create, list, and delete projects.
- **PDF Handling**: Upload and process PDFs for retrieval-based question answering.
- **Contextual Q&A**: Provide accurate answers specific to the active project's PDFs.

The application is built with **Streamlit**, **FAISS**, and **Groq AI**, optimized for performance with caching mechanisms and designed for usability with a simple interface.

---

## **Libraries Used**

- **Streamlit**: User interface framework.
- **LangChain**: Backend for building LLM-powered applications.
- **Groq**: For fast model inferencing.
- **OpenAI**: Generates embeddings for retrieval.
- **OS**: Manages project directories.
- **FAISS**: Vector storage and retrieval.

---

## **Core Features**

### **1. Environment Setup**
- Loads API keys for **Groq** (`GROQ_API_KEY`) and **OpenAI** (`OPENAI_API_KEY`) from environment variables.
- Initializes the LLM for fast and accurate inference.

### **2. Project Management**
- Create, delete, and switch projects dynamically.
- Maintain project-specific directories for PDFs and vector stores.
- Use `st.session_state` for seamless interactivity.

### **3. PDF Handling**
- Upload and store PDFs in project-specific directories.
- Process PDFs into manageable chunks using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.

### **4. Vector Store**
- Generate embeddings with OpenAI and store them in **FAISS** for efficient retrieval.
- Reuse previously saved vector stores to optimize performance.

### **5. Retrieval Augmented Generation (RAG)**
- Use FAISS-based retrieval with Groq AI's LLM for contextual question answering.
- Ensure responses are based on uploaded PDFs through a custom `ChatPromptTemplate`.

### **6. Streamlit UI**
- Sidebar for project management: create, delete, switch projects.
- Main interface for PDF upload, vector store initialization, and Q&A.
- Dynamic session updates for a smooth user experience.

---

## **Application Workflow**

1. **Project Management**
   - Create a new project from the sidebar.
   - Initialize a directory for the project and update session state.
2. **PDF Upload**
   - Upload PDFs, parse text content, and split into chunks.
   - Embed documents into a vector store for efficient retrieval.
3. **Question Answering**
   - Retrieve relevant chunks from the vector store based on user questions.
   - Generate project-specific responses using the LLM.
4. **Project Switching**
   - Switch between projects dynamically, reloading relevant vector stores without requiring re-upload.

---

## **Structure**

1. **Environment Initialization**
   - Sets up the LLM with Groq AI.
   - Creates a `projects` directory for managing data.
2. **Project Management**
   - Functions like `create_project`, `delete_project`, and `set_current_project` handle lifecycle management.
3. **Retrieval-Augmented Generation (RAG)**
   - PDFs are processed, split into chunks, and embedded.
   - The LLM retrieves and answers questions contextually.
4. **User Interface**
   - Streamlit UI provides an intuitive experience with dynamic session updates.

---

## **Demo**

Watch the demo of the application [here](https://www.youtube.com/watch?v=mOlA6L104-M).

---

## **How to Run**

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up API keys for OpenAI and Groq in your environment.
4. Run the application using `streamlit run app.py`.
