# Use command prompt : streamlit run Phase2.py
import os
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import os
import sys
import subprocess
import base64

def install_missing_packages():
    """Install required packages if missing"""
    packages = ["streamlit", "transformers", "langchain-huggingface", "langchain-core", 
                "langchain", "langchain-community", "sentence-transformers", "faiss-cpu", "torch"]
    
    python_exe = sys.executable
    pip_path = os.path.join(os.path.dirname(python_exe), "pip.exe")
    
    for package in packages:
        try:
            if package == "streamlit":
                import streamlit
            elif package == "transformers":
                import transformers
            elif package == "torch":
                import torch
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([pip_path, "install", package])
                print(f"✅ {package} installed successfully!")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")

install_missing_packages()
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="UIUC Student Assist",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_image(image_path):
    """Convert image to base64 for embedding in CSS"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading background image: {e}")
        return ""

# Convert background image to base64
image_path = r"C:\Users\91940\Downloads\Database_Dump\Medibot - Extension\wp2374096.jpg"
image_base64 = get_base64_image(image_path)

st.markdown(f"""
<style>
    div[data-testid="InputInstructions"] {{
        display: none !important;
    }}

    /* Main app background: replaced with custom image */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        transition: background 0.5s;

        
    }}

    /* REMOVED the big translucent box from main content area */
    .main .block-container {{
        background: transparent !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: none !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }}

    /* Add background only to specific containers */
    

    .main-header {{
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}

    .sub-header {{
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }}

    .user-message {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}

    .bot-message {{
        background: white;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }}

    .source-badge {{
        background: #FF6B35;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}

    /* Smaller, more compact sidebar menu */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #2c3e50, #34495e);
        color: white;
        padding: 0.5rem 0.75rem !important;
        min-width: 150px !important;
        max-width: 150px !important;
        font-size: 0.9rem !important;
    }}
    .sidebar .element-container {{
        padding: 0.2rem 0.3rem !important;
        font-size: 0.9rem !important;
    }}

    .stButton button {{
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
        outline: none;
        filter: none;
        margin-bottom: 0.2rem;
        cursor: pointer;
        font-size: 0.9rem;
    }}

    .stButton button:hover {{
        background: linear-gradient(135deg, #E55A2B, #E58217);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
    }}

    .quick-question {{
        background: white;
        border: 2px solid #FF6B35;
        border-radius: 10px;
        padding: 0.6rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-weight: 500;
        color: #FF6B35;
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.09);
        font-size: 0.85rem;
    }}
    .quick-question:hover {{
        background: #FF6B35;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.15);
    }}

    .stForm {{
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }}

    .stTextInput input {{
        color: black !important;
        background: white !important;
        border: 2px solid white !important;   /* make border invisible */
        border-radius: 25px !important;
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    .stTextInput input:focus {{
        border-color: white !important;       /* keep border camouflaged on focus */
        box-shadow: none !important;
        outline: none !important; 
    }}

    /* Remove the info box styling */
    .stInfo {{
        display: none !important;
    }}
    

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #FF6B35;
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #E55A2B;
    }}
    
    /* Developer name styling */
    .developer-name {{
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }}

    /* Style the main content area to be transparent */
    .main {{
        background: transparent !important;
    }}

    /* Ensure the chat area has proper spacing */
    .css-1d391kg {{
        padding: 2rem !important;
    }}
</style>
""", unsafe_allow_html=True)

def load_llm():
    try:
        pipe = pipeline(
            "text2text-generation",
            model="./fine_tuned_flan_uiuc_20min",  # Use your fine-tuned model
            max_new_tokens=250,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.1,
            device=-1
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to a simple LLM
        from langchain_community.llms import FakeListLLM
        responses = ["I specialize in UIUC student information. Please ask questions about campus life, academics, or university services."]
        return FakeListLLM(responses=responses)

# Updated prompt for UIUC student assistance
uiuc_prompt_template = """Using the UIUC context below, provide a helpful and informative answer to the student's question.

Context: {context}

Question: {question}

Please provide a clear, detailed answer based on the available university information. Focus on being helpful to UIUC students with accurate, practical information.

Helpful UIUC answer:"""

prompt = PromptTemplate(
    template=uiuc_prompt_template,
    input_variables=["context", "question"]
)

# Load database
@st.cache_resource
def load_database():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_path = r"C:\Users\91940\Downloads\Database_Dump\Medibot - Extension\Embeddings"
        
        # Check if database exists
        if not os.path.exists(db_path):
            st.error(f"Database path does not exist: {db_path}")
            return None
            
        db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        st.success("✅ Database loaded successfully!")
        return db
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# UIUC domain filter
UIUC_TOPICS = [
    'admissions', 'courses', 'professors', 'majors', 'campus', 'housing', 'dining',
    'library', 'academic', 'registration', 'financial', 'tuition', 'scholarships',
    'career', 'internships', 'research', 'graduate', 'undergraduate', 'illinois',
    'uiuc', 'university', 'student', 'faculty', 'campuslife', 'clubs', 'sports',
    'grades', 'gpa', 'deadlines', 'calendar', 'advising', 'department'
]

def is_uiuc_question(question):
    """Check if question seems related to UIUC"""
    if not question:
        return False
    question_lower = question.lower()
    return any(term in question_lower for term in UIUC_TOPICS)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False

# Quick questions for users
QUICK_QUESTIONS = [
    "What are the admission requirements for UIUC?",
    "How do I register for classes?",
    "What housing options are available?",
    "Tell me about computer science majors",
    "Where is the main library located?",
    "How to apply for financial aid?",
    "What career services are available?"
]

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 UIUC Student Assist</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="color: white; margin-bottom: 0.2rem;">Your AI assistant for University of Illinois Urbana-Champaign</p>', unsafe_allow_html=True)
    
    # Add developer name
    st.markdown('<p class="developer-name" style="color: white; font-weight: bold; margin-top: 0.2rem;">Developed by Jeet Thakore</p>', unsafe_allow_html=True)

    # Sidebar - Reduced size
    with st.sidebar:
        st.markdown("### 🏫 About")
        st.markdown("""
        <div style='font-size: 0.85rem;'>
        I can help you with:
        - Admissions & Requirements
        - Academic Programs
        - Campus Life
        - Housing & Dining
        - Career Services
        - Student Resources
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Quick Questions")
        for q in QUICK_QUESTIONS:
            if st.button(q, key=f"quick_{q}"):
                st.session_state.messages.append({"role": "user", "content": q})
                process_question(q)
        
        st.markdown("---")
        st.markdown("### 📊 Session Info")
        st.write(f"Messages: {len(st.session_state.messages)}")
        
        # Database status
        if st.session_state.db_loaded:
            st.success("✅ Database loaded")
        else:
            st.warning("⚠️ Database not loaded")
            
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
        # Load database button
        if st.button("📂 Load Database"):
            with st.spinner("Loading database..."):
                db = load_database()
                if db is not None:
                    st.session_state.db_loaded = True
                    st.rerun()
    
    # Main chat area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        if not st.session_state.messages:
            st.markdown('<div class="bot-message">🤖 Hello! I\'m your UIUC Student Assistant. How can I help you today?</div>', unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        st.markdown("**Sources:**")
                        for source in message["sources"]:
                            st.markdown(f'<div class="source-badge">📚 {source}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input form with rounded edges
        with st.form("question_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_input("Ask about UIUC:", placeholder="Type your question here...", label_visibility="collapsed")
            with col2:
                submitted = st.form_submit_button("🚀 Ask")
            if submitted and question:
                if not st.session_state.db_loaded:
                    st.warning("⚠️ Please load the database first using the button in the sidebar.")
                else:
                    st.session_state.messages.append({"role": "user", "content": question})
                    process_question(question)
                    st.rerun()

def process_question(question):
    """Process the user's question and generate response"""
    # Initialize QA chain if not exists
    if st.session_state.qa_chain is None:
        db = load_database()
        if db is None:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "⚠️ Database not available. Please ensure the UIUC embeddings are properly set up.",
                "sources": []
            })
            return
            
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
    
    # Check if question is UIUC-related
    if not is_uiuc_question(question):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🎓 I specialize in UIUC student information. Please ask about admissions, academics, campus life, or university services at the University of Illinois Urbana-Champaign.",
            "sources": []
        })
        return
    
    try:
        with st.spinner("🔍 Searching UIUC resources..."):
            response = st.session_state.qa_chain.invoke({'query': question})
            answer = response["result"].strip()
            
            # Extract source information
            sources = []
            if response["source_documents"]:
                for doc in response["source_documents"]:
                    source_content = ' '.join(doc.page_content.split()[:50])
                    sources.append(f"{source_content}...")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources[:3]  # Show top 3 sources
            })
            
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ I encountered an error: {str(e)}. Please try rephrasing your question about UIUC.",
            "sources": []
        })

if __name__ == "__main__":
    main()