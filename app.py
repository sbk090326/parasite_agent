import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from Bio import Entrez
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Set up Entrez Email
Entrez.email = os.getenv("ENTREZ_EMAIL", "your_email@example.com")
env_gemini_key = os.getenv("GEMINI_API_KEY", "")

# ----------------- Core Functions -----------------

def extract_text_from_file(uploaded_file):
    """
    Extract text content from uploaded .txt, .pdf, or .docx file.
    """
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif file_name.endswith(".pdf"):
        try:
            import pypdf
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except ImportError:
            st.warning("PDF 텍스트 추출을 위해 pypdf 라이브러리가 필요합니다.")
            return "PDF reader library missing."
    elif file_name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        except ImportError:
            st.warning("Word 문서(.docx) 텍스트 추출을 위해 python-docx 라이브러리가 필요합니다.")
            return "Word reader library missing."
    return ""

def fetch_pubmed_papers(keywords, journal, max_results=5):
    """
    Search PubMed for papers matching keywords and a specific target journal.
    """
    journal_queries = {
        "PLOS Pathogens": '"PLoS Pathog"[Journal] OR "PLOS Pathogens"[Journal]',
        "International Journal for Parasitology (IJP)": '"Int J Parasitol"[Journal] OR "International Journal for Parasitology"[Journal]'
    }
    
    jq = journal_queries.get(journal, f'"{journal}"[Journal]')
    query = f"({keywords}) AND ({jq})"
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        if not id_list:
            return []
        
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
        
        papers = []
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            
            title = article_data.get("ArticleTitle", "No Title Available")
            
            abstract_dict = article_data.get("Abstract", {})
            abstract_texts = abstract_dict.get("AbstractText", [])
            abstract = " ".join([str(text) for text in abstract_texts]) if abstract_texts else "No Abstract Available"
            
            authors = []
            for author in article_data.get("AuthorList", []):
                last_name = author.get("LastName", "")
                initials = author.get("Initials", "")
                if last_name:
                    authors.append(f"{last_name} {initials}")
            authors_str = ", ".join(authors) if authors else "Unknown Authors"
            
            pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", "N/A")
            
            papers.append({
                "title": title,
                "authors": authors_str,
                "year": year,
                "abstract": abstract
            })
            
        return papers
    except Exception as e:
        st.error(f"PubMed API Error: {str(e)}")
        return []

def analyze_manuscript(abstract_text, target_journal, keywords, matching_papers, api_key_valid, api_key):
    """
    Run Gemini LLM Agent to analyze manuscript peer-review and calculate success probability.
    """
    if not api_key_valid:
        return {
            "error": "유효한 Google Gemini API Key가 설정되지 않았습니다. 사이드바에 API 키를 입력해 주세요."
        }
        
    try:
        genai.configure(api_key=api_key)
        background_context = ""
        for i, paper in enumerate(matching_papers, 1):
            background_context += f"Paper {i}:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
            
        prompt = f"""
당신은 세계적인 기생충학(Parasitology) 분야의 권위 있는 저널인 **PLOS Pathogens** 및 **International Journal for Parasitology (IJP)**의 시니어 에디터이자 피어 리뷰어입니다.

연구자가 제출한 아래 [대상 논문 원고/초록]을 읽고, [최신 유사 합격 논문 정보]를 참조하여 종합적인 가상 피어 리뷰 리포트를 작성해 주세요.

[대상 저널]
{target_journal}

[연구 키워드]
{keywords}

[대상 논문 원고/초록]
{abstract_text}

[최신 유사 합격 논문 정보 (PubMed 검색 결과)]
{background_context}

---

**[요구사항 및 리포트 작성 가이드라인]**
1. **평가 어조**: 전문적이며 건설적이고 예리하게 지적해 주어야 합니다.
2. **합격 확률**: 0에서 100 사이의 숫자로 합격 가능성을 정량 예측해 주세요.
3. **작성 언어**: 한국어로 출력해야 합니다. 단, 논문의 주요 생물학 용어(예: 유전자명, 경로명, 실험기법 등)는 영어 원문을 함께 기입해 주세요.
4. **리포트 구성 형식**: 아래 지정된 JSON 포맷으로 반드시 응답해야 하며, 그 외의 다른 텍스트는 절대 포함하지 마십시오.

```json
{{
  "score": 75,
  "journal_fit": "저널 적합성에 대한 2-3문장 분석",
  "strengths": [
    "강점 1 (예: 참신한 메커니즘 규명 등)",
    "강점 2"
  ],
  "reject_risks": [
    "데스크 리젝트 또는 심사 시 지적받을 리스크 1 (예: 대조군 미흡, In vivo 검증 부족 등)",
    "리스크 2"
  ],
  "action_plans": [
    "보완 대책 및 추가 실험 권고 사항 1",
    "보완 대책 및 추가 실험 권고 사항 2"
  ]
}}
```
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        
        import json
        result = json.loads(response.text.strip())
        return result
    except Exception as e:
        return {
            "error": f"AI 분석 중 오류가 발생했습니다: {str(e)}"
        }

# ----------------- Streamlit UI Page Setup -----------------

st.set_page_config(
    page_title="Parasitology Journal Acceptance Guide Agent",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for dark-mode premium scientific style
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #2b5c8f;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3b7cb8;
    }
    .card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }
    .card-title {
        color: #58a6ff;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- App Layout & Sidebar -----------------

st.title("🔬 Parasitology Journal Acceptance Guide Agent")
st.markdown("### PLOS Pathogens & International Journal for Parasitology (IJP) 투고 지원 솔루션")
st.write("본 프로그램은 사용자의 초록/전체 논문 데이터를 Google AI Studio 통신으로 보호하여 학습에 반영하지 않고 분석합니다.")

with st.sidebar:
    st.header("🔑 API 설정")
    
    # Input user API Key or fallback to .env
    user_api_key = st.text_input(
        "Google Gemini API Key 입력",
        type="password",
        placeholder="AI Studio에서 발급받은 API 키를 입력하세요...",
        help="입력하지 않으면 기본 서버 환경설정(.env)의 API 키를 사용합니다."
    )
    
    selected_key = user_api_key.strip() if user_api_key.strip() else env_gemini_key.strip()
    is_api_key_valid = False
    
    if selected_key and not selected_key.startswith("your_gemini_api_key") and selected_key != "":
        is_api_key_valid = True
        
    if is_api_key_valid:
        st.success("Google Generative AI 연결 준비 완료")
    else:
        st.error("Gemini API 키를 입력해 주세요. (미설정 상태)")
        
    st.header("⚙️ 검색 옵션")
    target_journal = st.selectbox(
        "타깃 저널 선택",
        ["PLOS Pathogens", "International Journal for Parasitology (IJP)"]
    )
    max_papers = st.slider("PubMed 참고 논문 매칭 수", min_value=3, max_value=10, value=5)

# Main Form
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'><div class='card-title'>✍️ 연구 정보 입력</div>", unsafe_allow_html=True)
    keywords = st.text_input("연구 핵심 키워드 (예: Toxoplasma gondii, autophagy, macrophage)", "")
    
    # File Uploader for multiple formats (.txt, .pdf, .docx)
    uploaded_file = st.file_uploader("초록 또는 전체 논문 파일 업로드 (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    
    default_text = ""
    if uploaded_file is not None:
        default_text = extract_text_from_file(uploaded_file)
        
    abstract_text = st.text_area(
        "원고 내용 직접 입력 또는 수정",
        value=default_text,
        height=300,
        placeholder="이곳에 논문의 Abstract 또는 전체 원고 텍스트를 입력하십시오..."
    )
    
    submit_button = st.button("🚀 투고 적합성 고속 매칭 & AI 피어 리뷰 시작")
    st.markdown("</div>", unsafe_allow_html=True)

# Process logic
if submit_button:
    if not keywords:
        st.warning("PubMed 유사 논문 검색을 위해 키워드를 입력해 주세요.")
    elif not abstract_text:
        st.warning("리뷰할 초록(Abstract) 내용을 입력해 주세요.")
    else:
        with st.spinner("1️⃣ PubMed API에서 최신 합격 논문 매칭 검색 중..."):
            matching_papers = fetch_pubmed_papers(keywords, target_journal, max_results=max_papers)
            
        with col2:
            st.markdown("<div class='card'><div class='card-title'>📚 PubMed 매칭 논문</div>", unsafe_allow_html=True)
            if not matching_papers:
                st.info("해당 키워드와 저널 조합으로 매칭된 최신 논문이 없습니다. (키워드를 더 넓게 조정해 보세요)")
            else:
                for idx, paper in enumerate(matching_papers, 1):
                    with st.expander(f"[{paper['year']}] {paper['title']}"):
                        st.markdown(f"**저자:** {paper['authors']}")
                        st.markdown(f"**Abstract:** {paper['abstract']}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("2️⃣ Gemini 1.5 Pro/Flash 기반 가상 피어 리뷰 및 리포트 생성 중..."):
            analysis = analyze_manuscript(abstract_text, target_journal, keywords, matching_papers, is_api_key_valid, selected_key)
            
        with col2:
            if "error" in analysis:
                st.error(analysis["error"])
            else:
                st.markdown("<div class='card'><div class='card-title'>📊 투고 통과 예측 확률 및 심사 결과</div>", unsafe_allow_html=True)
                
                score = analysis.get("score", 50)
                
                # Gauge Chart Using Plotly
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{target_journal} 예측 합격률"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#2b5c8f"},
                        'steps': [
                            {'range': [0, 50], 'color': "#4a1212"},
                            {'range': [50, 80], 'color': "#3c3d10"},
                            {'range': [80, 100], 'color': "#13381a"}
                        ],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**🧐 저널 적합성 분석:**\n{analysis.get('journal_fit', 'N/A')}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Strengths & Risks
                st.markdown("<div class='card'><div class='card-title'>🌟 논문 주요 강점 (Strengths)</div>", unsafe_allow_html=True)
                for strength in analysis.get("strengths", []):
                    st.markdown(f"✅ {strength}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='card'><div class='card-title'>⚠️ 리젝트 리스크 리스크 (Reject Risks)</div>", unsafe_allow_html=True)
                for risk in analysis.get("reject_risks", []):
                    st.markdown(f"❌ {risk}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Action Plans
                st.markdown("<div class='card'><div class='card-title'>💡 투고 성공률 극대화를 위한 보완 Action Plan</div>", unsafe_allow_html=True)
                for i, plan in enumerate(analysis.get("action_plans", []), 1):
                    st.markdown(f"**{i}. {plan}**")
                st.markdown("</div>", unsafe_allow_html=True)
