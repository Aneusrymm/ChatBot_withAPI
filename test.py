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
import hashlib
import shutil

# Konfigurasi Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load API Key dari .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key is missing. Please set GOOGLE_API_KEY in your .env file or environment variables.")
    st.stop()

generativeai.configure(api_key=api_key)

# Direktori penyimpanan teks hasil ekstraksi dan input manual
CACHE_DIR = "hasil_txt"
MANUAL_INPUT_FILE = "manual_input.txt"
os.makedirs(CACHE_DIR, exist_ok=True)

# Fungsi untuk menyimpan input manual ke file
def save_manual_input(text):
    with open(MANUAL_INPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Fungsi untuk membaca input manual dari file
def load_manual_input():
    if os.path.exists(MANUAL_INPUT_FILE):
        with open(MANUAL_INPUT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Fungsi untuk menyimpan dan membaca teks hasil ekstraksi PDF
def get_cache_file_path(pdf_hash):
    return os.path.join(CACHE_DIR, f"{pdf_hash}.txt")

def save_extracted_text(text, pdf_hash):
    cache_path = get_cache_file_path(pdf_hash)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_extracted_text(pdf_hash):
    cache_path = get_cache_file_path(pdf_hash)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def generate_pdf_hash(pdf_files):
    hash_md5 = hashlib.md5()
    for pdf in pdf_files:
        hash_md5.update(pdf.getvalue())
    return hash_md5.hexdigest()

# Fungsi untuk ekstraksi teks dari PDF dengan OCR jika diperlukan
def extract_pdf_content(pdf_files):
    combined_text = ""
    for pdf in pdf_files:
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
                    images = convert_from_path(tmp_file_path, first_page=page_num+1, last_page=page_num+1)
                    for image in images:
                        ocr_text = pytesseract.image_to_string(image)
                        combined_text += ocr_text + "\n"
        finally:
            os.unlink(tmp_file_path)
    return combined_text

# Fungsi untuk membagi teks menjadi bagian kecil
def split_text_into_chunks(content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    return splitter.split_text(content)

# Fungsi untuk memuat indeks FAISS yang sudah ada
def load_existing_faiss_index():
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    index_path = "faiss_index_store"
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
    return None

# Fungsi untuk menambahkan data baru ke indeks FAISS
def generate_vector_index(text_segments):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    existing_index = load_existing_faiss_index()
    new_index = FAISS.from_texts(text_segments, embedding=embed_model)
    if existing_index:
        existing_index.merge_from(new_index)
        existing_index.save_local("faiss_index_store")
    else:
        new_index.save_local("faiss_index_store")

# Fungsi untuk membuat chain Q&A
def create_qa_chain():
    
    # custom_prompt = """
    # Please provide a detailed answer based on the context given. If the context does not contain the answer, simply state, 
    # "The answer is not available in the provided context." Please do not guess.\n\n
    # Context:\n {context}\n
    # Question:\n {question}\n
    # Answer:
    # """
    custom_prompt = """
    berikan jawaban rinci,jelas dan tepat berdasarkan konteks yang diberikan. Jika konteks tersebut mengandung referensi seperti URL, nama,gambar, atau informasi khusus, jika konteks tidak berisi jawaban nyatakan dengan "maaf jawaban tidak ada , kirimkan pertanyaan yang lebih spesifik " ,jawab pertanyaan dengan menggunakan bahasa indonesia\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt_template = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

# Fungsi untuk menangani pertanyaan pengguna
def handle_user_query(query):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    faiss_db = load_existing_faiss_index()
    if not faiss_db:
        st.warning("No indexed documents found. Please upload and process PDFs first.")
        return
    matched_docs = faiss_db.similarity_search(query)
    qa_chain = create_qa_chain()
    result = qa_chain({"input_documents": matched_docs, "question": query}, return_only_outputs=True)
    st.write( result.get("output_text", "No response generated."))

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Q&A with Gemini")
    st.header("Chat with Gemini AI 🤖 Who Consumed Knowledge from your PDF")

    existing_faiss_index = load_existing_faiss_index()
    if existing_faiss_index:
        st.success("Indeks FAISS kebaca.")

    user_query = st.text_input("Masukan pertanyaan:", key="user_query")
    if user_query:
        handle_user_query(user_query)

    with st.sidebar:
        st.title("Upload Section:")
        uploaded_pdfs = st.file_uploader("Select PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not uploaded_pdfs:
                st.warning("Silakan upload file PDF terlebih dahulu.")
            else:
                pdf_hash = generate_pdf_hash(uploaded_pdfs)
                cached_text = load_extracted_text(pdf_hash)
                manual_text = load_manual_input()
                combined_text = manual_text + "\n" + (cached_text if cached_text else extract_pdf_content(uploaded_pdfs))
                save_extracted_text(combined_text, pdf_hash)
                text_segments = split_text_into_chunks(combined_text)
                generate_vector_index(text_segments)
                st.success("All PDFs processed successfully!")

        st.subheader("Tambahkan Informasi Manual:")
        user_input_text = st.text_area("Masukkan teks tambahan:")
        if st.button("Tambahkan ke Database"):
            if user_input_text.strip():
                save_manual_input(user_input_text)
                text_segments = split_text_into_chunks(user_input_text)
                generate_vector_index(text_segments)
                st.success("Informasi berhasil ditambahkan!")
            else:
                st.warning("Harap masukkan teks sebelum menambahkan.")

if __name__ == "__main__":
    main()
