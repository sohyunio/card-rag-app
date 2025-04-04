import requests
import json
import os  # 폴더 생성용
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
import torch
_ = torch.classes

# API Key 입력
api_key = "up_P5UsiWxIdS9sAyUCQmc9R5krA9dQr"
url = "https://api.upstage.ai/v1/document-ai/document-parse"

headers = {
    "Authorization": f"Bearer {api_key}"
}

pdf_files = ["삼성.pdf", "하나.pdf", "토스.pdf", "우리.pdf"]

# 저장할 폴더 경로 (card-rag-app/parsed_html)
save_dir = "parsed_html"
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 자동 생성

# 여러 개의 파일을 변환하는 반복문
for filename in pdf_files:
    with open(filename, "rb") as file:
        files = {"document": file}

        data = {
            "ocr": "force",
            "coordinates": True,
            "chart_recognition": True,
            "output_formats": ["html"],
            "base64_encoding": [],
            "model": "document-parse"
        }

        response = requests.post(url, headers=headers, files=files, json=data)
        result = response.json()

        html_text = result['content']['html']
        card_name = filename.split(".")[0]
        output_path = os.path.join(save_dir, f"{card_name}.html")
        print(f"Saving {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_text)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# HTML 파일이 저장된 폴더
html_dir = "parsed_html"
chunks = []

# HTML 파일 읽고 텍스트 분할
for filename in os.listdir(html_dir):
    if filename.endswith(".html"):
        path = os.path.join(html_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50,
                                                  separators=["\n\n", "\n", ".", " ", ""])
        card_chunks = splitter.split_text(text)
        print(f"✅ {filename}: {len(card_chunks)}개 chunk 생성됨")
        chunks.extend(card_chunks)

chunks = chunks[:20]

if not chunks:
    raise ValueError("❗ HTML에서 추출된 텍스트가 없습니다.")

vector_db = FAISS.from_texts(chunks, embedding_model)
vector_db.save_local("vector_db")
print("✅ 벡터 DB 저장 완료! → vector_db/")

# 임베딩 모델 및 벡터 DB 로드
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)

# HuggingFace LLM 파이프라인 (flan-t5-base)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    return_source_documents=True
)

query = "배달앱 할인되는 카드 뭐 있어?"
result = qa.invoke(query)

print("💬 응답:\n", result["result"])
print("\n📄 참조된 카드 혜택 문서:\n")
for doc in result["source_documents"]:
    print("———")
    print(doc.page_content)

# --- Streamlit 앱 시작 ---
st.set_page_config(page_title="카드 추천 챗봇")
st.title("💳 대학생 대상 카드 추천 AI 챗봇")
st.markdown("카드 약관 기반으로 혜택을 분석해주는 AI 챗봇입니다.")

user_query = st.text_input("❓ 궁금한 점을 입력하세요", placeholder="예: 배달앱 할인 카드 뭐 있어?")

if user_query:
    with st.spinner("🤖 AI가 분석 중입니다..."):

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("vector_db", embedding, allow_dangerous_deserialization=True)

        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            truncation=True
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 1})
        docs = retriever.get_relevant_documents(user_query)

        context_raw = docs[0].page_content
        question_raw = user_query

        question_tokens = tokenizer.encode(f"질문:\n{question_raw}", add_special_tokens=False)
        max_input_tokens = 512
        buffer_tokens = 10
        max_context_tokens = max_input_tokens - len(question_tokens) - buffer_tokens

        context_tokens = tokenizer.encode(context_raw, add_special_tokens=False)[:max_context_tokens]
        context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)

        final_prompt = f"카드 혜택 설명:\n{context_text}\n\n질문:\n{question_raw}"
        input_ids = tokenizer.encode(final_prompt, return_tensors="pt")

        output = model.generate(input_ids, max_new_tokens=128)
        final_answer = tokenizer.decode(output[0], skip_special_tokens=True)

    st.markdown("### 💬 응답")
    st.write(final_answer or "⚠️ 답변이 비어있습니다. 입력을 확인해주세요.")

    with st.expander("📄 참조된 문서"):
        for doc in docs:
            st.markdown("———")
            st.write(doc.page_content)
