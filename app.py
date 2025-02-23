import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as generativeai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import tempfile

# Konfigurasi Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key is missing. Please set GOOGLE_API_KEY in your .env file or environment variables.")
    st.stop()

generativeai.configure(api_key=api_key)

def extract_pdf_content(pdf_files):
    combined_text = ""

    for pdf in pdf_files:
        # Simpan file PDF yang diunggah ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        try:
            reader = PdfReader(tmp_file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                if text and text.strip():  
                    combined_text += text + "\n"
                else:
                    # Jika teks tidak ditemukan, gunakan OCR pada halaman tersebut
                    images = convert_from_path(tmp_file_path, first_page=page_num+1, last_page=page_num+1)
                    
                    for image in images:
                        ocr_text = pytesseract.image_to_string(image)
                        combined_text += ocr_text + "\n"
        
        finally:
           
            os.unlink(tmp_file_path)
    
    return combined_text

def split_text_into_chunks(content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    return splitter.split_text(content)

def generate_vector_index(text_segments):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_db = FAISS.from_texts(text_segments, embedding=embed_model)
    vector_db.save_local("faiss_index_store")

def create_qa_chain():
    custom_prompt = """
    Please provide a detailed answer based on the context given. If the context does not contain the answer, simply state, 
    "The answer is not available in the provided context." Please do not guess.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt_template = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

def handle_user_query(query):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    faiss_db = FAISS.load_local("faiss_index_store", embedding_model, allow_dangerous_deserialization=True)
    matched_docs = faiss_db.similarity_search(query)
    qa_chain = create_qa_chain()
    result = qa_chain({"input_documents": matched_docs, "question": query}, return_only_outputs=True)
    st.write("AI Response:", result.get("output_text", "No response generated."))

def main():
    st.set_page_config(page_title="PDF Q&A with Gemini")
    st.header("Chat with Gemini AI 🤖 Who Consumed Knowledge from your PDF")
    user_query = st.text_input("Enter a question based on your uploaded PDFs:")
    if user_query:
        handle_user_query(user_query)
    with st.sidebar:
        st.title("Upload Section:")
        uploaded_pdfs = st.file_uploader("Select PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing your files..."):
                extracted_text = extract_pdf_content(uploaded_pdfs)
                if extracted_text.strip():
                    text_segments = split_text_into_chunks(extracted_text)
                    generate_vector_index(text_segments)
                    st.success("PDFs processed successfully!")
                else:
                    st.error("No valid text extracted from PDFs. Please try another file.")

if __name__ == "__main__":
    main()