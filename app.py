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
        "International Journal for Parasitology (IJP)": '"Int J Parasitol"[Journal] OR "International Journal for Parasitology"[Journal]',
        "TRENDS IN PARASITOLOGY": '"Trends Parasitol"[Journal] OR "Trends in Parasitology"[Journal]',
        "Parasites & Vectors": '"Parasites Vectors"[Journal] OR "Parasites & Vectors"[Journal]',
        "PLoS Neglected Tropical Diseases": '"PLoS Negl Trop Dis"[Journal] OR "PLoS Neglected Tropical Diseases"[Journal]',
        "Frontiers in Microbiology": '"Front Microbiol"[Journal] OR "Frontiers in Microbiology"[Journal]',
        "Journal of Eukaryotic Microbiology": '"J Eukaryot Microbiol"[Journal] OR "Journal of Eukaryotic Microbiology"[Journal]',
        "Frontiers in Cellular and Infection Microbiology": '"Front Cell Infect Microbiol"[Journal] OR "Frontiers in Cellular and Infection Microbiology"[Journal]'
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
            pmid = str(medline.get("PMID", ""))
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
                "abstract": abstract,
                "pmid": pmid
            })
            
        return papers
    except Exception as e:
        st.error(f"PubMed API Error: {str(e)}")
        return []

# 저널별 평가 프로필 정의 (난이도, 중점 심사 기준, 평가 지침)
JOURNAL_PROFILES = {
    "PLOS Pathogens": {
        "difficulty": "매우 높음 (Top-tier 저널, 높은 수준의 병원성 메커니즘 검증 및 In vivo 데이터 필수)",
        "focus": "병원성 기전의 심층 규명, 생체 내(In vivo) 검증의 유무, 기생충-숙주 상호작용의 구체성",
        "instructions": "매우 보수적이고 엄격하게 점수를 매기십시오. 단순 현상 기술이나 데이터 양이 적은 연구는 감점을 크게 하며, 70점 이상을 받기가 극히 어렵습니다."
    },
    "International Journal for Parasitology (IJP)": {
        "difficulty": "높음 (기생충학 분야 최고의 전통 저널)",
        "focus": "분자생물학적/면역학적 분석의 타당성, 기생충 모델의 독창성 및 학술적 깊이",
        "instructions": "학술적 참신함과 논리적 완성도가 높아야 75점 이상을 부여합니다. 생리학적/면역학적 메커니즘을 상세히 다루었는지 엄밀히 평가하십시오."
    },
    "TRENDS IN PARASITOLOGY": {
        "difficulty": "매우 높음 (높은 임팩트의 리뷰 및 트렌드 의견 제시 위주 저널)",
        "focus": "해당 분야를 선도할 수 있는 참신한 통찰력과 학술적 영향력, 명확한 개념적 진보",
        "instructions": "연구 데이터의 참신성뿐만 아니라, 해당 원고가 Parasitology 분야 전체에 미치는 개념적이고 패러다임적인 영향력을 기준으로 엄격하게 평가하십시오."
    },
    "Parasites & Vectors": {
        "difficulty": "보통 (실용적이고 기술적인 연구도 많이 수용)",
        "focus": "매개체-기생충 상호작용 및 역학 연구, 실험 결과의 실무적 적용 가능성 및 데이터 신뢰도",
        "instructions": "기존에 잘 알려진 주제라도 데이터가 견고하고 역학적 가치가 있다면 점수를 합리적으로 부여(70~85점 가능)하십시오. 불필요하게 점수를 깎기보다 데이터 검증성에 초점을 맞추십시오."
    },
    "PLoS Neglected Tropical Diseases": {
        "difficulty": "높음 (소외된 열대 질환 관련 대표 저널)",
        "focus": "NTD 질환에 대한 공중보건학적 의의, 병원성 분석, 역학적 유용성 및 실용성",
        "instructions": "공중보건적 임팩트와 병리 메커니즘을 동시에 균형 있게 평가하십시오. 소외 질환 퇴치에 어떻게 기여하는지 명확해야 높은 점수를 얻습니다."
    },
    "Frontiers in Microbiology": {
        "difficulty": "보통-높음 (넓은 스펙트럼의 미생물/면역 분야 저널)",
        "focus": "미생물학적 기초 연구, 면역학적 분석, 실험 방법론의 타당성과 명확성",
        "instructions": "데이터가 체계적이고 결론을 지지하기에 타당하다면 합리적인 점수대(65~80점)를 유연하게 제공하십시오."
    },
    "Journal of Eukaryotic Microbiology": {
        "difficulty": "보통 (진핵 미생물 전문 저널)",
        "focus": "원생동물의 세포생물학, 분류학, 진화 및 유전학적 분석",
        "instructions": "생물학적 발견의 고유성에 가치를 두되, 데이터가 타당하고 체계적이라면 70점 내외의 긍정적인 점수를 부여하십시오."
    },
    "Frontiers in Cellular and Infection Microbiology": {
        "difficulty": "보통-높음 (감염 및 세포 미생물학 전문 저널)",
        "focus": "숙주-기생충 상호작용 시 세포 수준의 기전 분석, 감염 모델의 정확성",
        "instructions": "감염 세포 수준의 메커니즘이 잘 입증되었다면 비교적 유연한 합격 점수를 수용할 수 있습니다."
    }
}

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
            
        # 저널 프로필 획득
        journal_info = JOURNAL_PROFILES.get(target_journal, {
            "difficulty": "보통",
            "focus": "학술적 타당성 및 연구의 신뢰도",
            "instructions": "일반적인 저널 심사 기준을 따르며, 연구 내용의 데이터 신뢰성을 검증하십시오."
        })
        
        prompt = f"""
당신은 세계적인 기생충학(Parasitology) 및 미생물학 분야의 권위 있는 저널인 **{target_journal}**의 시니어 에디터이자 피어 리뷰어입니다.

연구자가 제출한 아래 [대상 논문 원고/초록]을 읽고, [최신 유사 합격 논문 정보]를 참조하여 종합적인 가상 피어 리뷰 리포트를 작성해 주세요.
특히, 본 저널({target_journal}) 고유의 투고 난이도 및 평가 기준을 절대적으로 적용하여 엄밀하게 스코어링해야 합니다.

[대상 저널 정보]
- 저널 이름: {target_journal}
- 게재 난이도: {journal_info['difficulty']}
- 핵심 평가 요소: {journal_info['focus']}

[스코어 산출 및 심사 기준 안내]
- {journal_info['instructions']}
- 저널의 난이도가 높을수록 더 엄격하고 깐깐하게 점수를 매겨야 합니다. 난이도가 매우 높은 저널의 경우, 본문 내용이나 대조군이 부실하면 50점 이하의 데스크 리젝트(Desk Reject) 위기 점수를 부여해야 합니다.
- 반면, 비교적 유연한 저널의 경우 데이터가 신뢰할 수 있다면 합리적인 패스 점수를 부여할 수 있습니다.

[연구 키워드]
{keywords}

[대상 논문 원고/초록]
{abstract_text}

[최신 유사 합격 논문 정보 (PubMed 검색 결과)]
{background_context}

---

**[요구사항 및 리포트 작성 가이드라인]**
1. **평가 어조**: 전문적이며 건설적이고 예리하게 지적해 주어야 합니다.
2. **합격 확률**: 0에서 100 사이의 숫자로 합격 가능성을 정량 예측해 주세요. 선택하신 저널의 투고 난이도와 연구 원고의 품질을 반영해야 하므로, 저널에 따라 점수가 뚜렷하게 차이 나야 합니다.
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
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .card-title {
        color: #58a6ff;
        font-size: 1.35rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
    }
    p, li, span, div {
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
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
        [
            "PLOS Pathogens", 
            "International Journal for Parasitology (IJP)",
            "TRENDS IN PARASITOLOGY",
            "Parasites & Vectors",
            "PLoS Neglected Tropical Diseases",
            "Frontiers in Microbiology",
            "Journal of Eukaryotic Microbiology",
            "Frontiers in Cellular and Infection Microbiology"
        ]
    )
    max_papers = st.slider("PubMed 참고 논문 매칭 수", min_value=3, max_value=20, value=5)


# Main Form Side-by-Side Column Design
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

with col2:
    st.markdown("<div class='card'><div class='card-title'>ℹ️ 사용 안내 및 매칭 기능</div>", unsafe_allow_html=True)
    st.markdown("""
    본 에이전트는 기생충학 분야 권위지 투고를 위해 두 가지 단계로 작동합니다:
    1. **PubMed 데이터 매칭**: 입력하신 키워드로 실제 게재된 최신 합격 논문 정보와 본문 링크를 가져옵니다.
    2. **가상 피어 리뷰**: 업로드된 원고를 수집된 합격 논문의 깊이와 대조 분석하여 수정 방향을 제시합니다.
    
    *오른쪽 상단 또는 사이드바에 API 키가 설정되어 있어야 AI 피어 리뷰가 작동합니다.*
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Process logic
if submit_button:
    if not keywords:
        st.warning("PubMed 유사 논문 검색을 위해 키워드를 입력해 주세요.")
    elif not abstract_text:
        st.warning("리뷰할 초록(Abstract) 내용을 입력해 주세요.")
    else:
        # 1. PubMed papers fetch
        with st.spinner("1️⃣ PubMed API에서 최신 합격 논문 매칭 검색 중..."):
            matching_papers = fetch_pubmed_papers(keywords, target_journal, max_results=max_papers)
            
        # Draw PubMed Results in Full Width Container
        st.markdown("---")
        st.markdown("<div class='card'><div class='card-title'>📚 PubMed 매칭 논문</div>", unsafe_allow_html=True)
        if not matching_papers:
            st.info("해당 키워드와 저널 조합으로 매칭된 최신 논문이 없습니다. (키워드를 더 넓게 조정해 보세요)")
        else:
            # Render match papers in columns for horizontal space efficiency
            paper_cols = st.columns(len(matching_papers) if len(matching_papers) > 0 else 1)
            for idx, paper in enumerate(matching_papers):
                with paper_cols[idx % len(paper_cols)]:
                    with st.expander(f"[{paper['year']}] {paper['title'][:50]}...", expanded=True):
                        st.markdown(f"**제목:** {paper['title']}")
                        st.markdown(f"**저자:** {paper['authors']}")
                        if paper.get('pmid'):
                            paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                            st.markdown(f"🔗 [본문 링크 (PubMed)]({paper_url})")
                        st.markdown(f"**Abstract:** {paper['abstract'][:200]}...")
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. AI Review
        with st.spinner("2️⃣ Gemini 2.5 Flash 기반 가상 피어 리뷰 및 리포트 생성 중..."):
            analysis = analyze_manuscript(abstract_text, target_journal, keywords, matching_papers, is_api_key_valid, selected_key)
            
        if "error" in analysis:
            st.error(analysis["error"])
        else:
            st.markdown("---")
            st.markdown("### 📊 AI 피어 리뷰 및 투고 적합성 분석 결과 (전체 화면)")
            
            # Row 1: Probability Gauge & Journal Fit (Side-by-side full width)
            row1_col1, row1_col2 = st.columns([1, 2])
            
            with row1_col1:
                st.markdown("<div class='card' style='height: 100%;'><div class='card-title'>🎯 투고 성공 확률</div>", unsafe_allow_html=True)
                score = analysis.get("score", 50)
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{target_journal} 예측 합격률", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#2b5c8f"},
                        'steps': [
                            {'range': [0, 50], 'color': "#4a1212"},
                            {'range': [50, 80], 'color': "#3c3d10"},
                            {'range': [80, 100], 'color': "#13381a"}
                        ],
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with row1_col2:
                st.markdown("<div class='card' style='height: 100%;'><div class='card-title'>🧐 저널 적합성 분석 (Journal Fit)</div>", unsafe_allow_html=True)
                st.write(analysis.get('journal_fit', 'N/A'))
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Space separator
            st.write("")
            
            # Row 2: Strengths & Risks (Side-by-side full width)
            row2_col1, row2_col2 = st.columns([1, 1])
            
            with row2_col1:
                st.markdown("<div class='card' style='height: 100%;'><div class='card-title'>🌟 논문 주요 강점 (Strengths)</div>", unsafe_allow_html=True)
                for strength in analysis.get("strengths", []):
                    st.markdown(f"✅ {strength}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with row2_col2:
                st.markdown("<div class='card' style='height: 100%;'><div class='card-title'>⚠️ 리젝트 리스크 (Reject Risks)</div>", unsafe_allow_html=True)
                for risk in analysis.get("reject_risks", []):
                    st.markdown(f"❌ {risk}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Space separator
            st.write("")
            
            # Row 3: Action Plans (Full width card)
            st.markdown("<div class='card'><div class='card-title'>💡 투고 성공률 극대화를 위한 보완 Action Plan</div>", unsafe_allow_html=True)
            for i, plan in enumerate(analysis.get("action_plans", []), 1):
                st.markdown(f"**{i}. {plan}**")
            st.markdown("</div>", unsafe_allow_html=True)

