# 💳 카드 추천 챗봇 (Streamlit + LangChain + HuggingFace)

이 프로젝트는 **대학생 대상 카드 추천 챗봇**으로, PDF로 된 카드 약관 파일을 분석하고, 사용자의 질문에 대해 **LLM 기반으로 카드 혜택 정보를 안내**합니다.

---

## 🔧 주요 기술 스택

- **Streamlit**: 사용자 인터페이스 구현
- **LangChain**: RAG (Retrieval-Augmented Generation) 구조
- **FAISS**: 카드 약관 임베딩을 위한 벡터 저장소
- **HuggingFace Transformers**: flan-T5 모델로 질문 응답 처리
- **Upstage Document AI API**: PDF → HTML 변환

---

## 📁 폴더 구조

```
card-rag-app/
├── parsed_html/        # HTML로 변환된 카드 약관 저장소
├── vector_db/          # FAISS 벡터 DB 저장소
├── 삼성.pdf, ...        # 카드 약관 PDF 파일들
├── app.py              # Streamlit 메인 앱
├── README.md
```

---

## 실행 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

> requirements.txt는 HuggingFace, LangChain, FAISS, Streamlit 등 포함해야 함

### 2. Streamlit 앱 실행

```bash
streamlit run app.py --server.runOnSave=false
```

> `runOnSave=false`는 PyTorch와 Streamlit 충돌 방지용 설정입니다.

---

## 📌 핵심 기능

### ✅ PDF → HTML 변환 (Upstage API 사용)

- `삼성.pdf`, `하나.pdf` 등 카드 약관 파일들을 HTML로 변환
- HTML 내용은 텍스트로 파싱되어 chunk 단위로 벡터화

### ✅ FAISS 벡터 생성 및 검색

- 약관 내용을 문단 단위로 자른 뒤 임베딩
- 유사한 chunk를 검색하여 LLM에 제공

### ✅ LLM 응답 처리

- 사용자 질문과 관련된 카드 혜택 정보를 찾아 prompt 구성
- flan-T5 모델을 통해 자연어로 응답 생성

---

## ❗ 유의사항

- 모델 입력 길이 초과 방지를 위해 **토큰 기준으로 context 잘림**
- `torch.classes` 관련 에러 방지를 위해 앱 시작 시 초기화 처리
- 무료 모델인 `flan-t5-small`, `flan-t5-base` 사용 중
- `get_relevant_documents()`는 향후 `invoke()`로 변경 예정 (LangChain 1.0 대응)

---

## 향후 개선 아이디어

-  여러 문서 chunk 통합 검색 (k>1)
-  카드사/카드명 필터 기능 추가
-  사용자 질문 로그 및 분석

---

## 👨‍💻 개발자

- **작성자**: @sohyunio, @nam-hyewon
- **기반 모델**: `google/flan-t5-small`, `sentence-transformers/all-MiniLM-L6-v2`
- **LLM 응답 엔진**: HuggingFace + LangChain RAG


