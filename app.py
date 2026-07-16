import os
import streamlit as st
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
        "instructions": "매우 보수적이고 엄격하게 점수를 매기십시오. 단순 현상 기술이나 데이터 양이 적은 연구는 감점을 크게 하며, 70점 이상을 받기가 극히 어렵습니다.",
        "tier": "top"
    },
    "International Journal for Parasitology (IJP)": {
        "difficulty": "높음 (기생충학 분야 최고의 전통 저널)",
        "focus": "분자생물학적/면역학적 분석의 타당성, 기생충 모델의 독창성 및 학술적 깊이",
        "instructions": "학술적 참신함과 논리적 완성도가 높아야 75점 이상을 부여합니다. 생리학적/면역학적 메커니즘을 상세히 다루었는지 엄밀히 평가하십시오.",
        "tier": "high"
    },
    "TRENDS IN PARASITOLOGY": {
        "difficulty": "매우 높음 (높은 임팩트의 리뷰 및 트렌드 의견 제시 위주 저널)",
        "focus": "해당 분야를 선도할 수 있는 참신한 통찰력과 학술적 영향력, 명확한 개념적 진보",
        "instructions": "연구 데이터의 참신성뿐만 아니라, 해당 원고가 Parasitology 분야 전체에 미치는 개념적이고 패러다임적인 영향력을 기준으로 엄격하게 평가하십시오.",
        "tier": "top"
    },
    "Parasites & Vectors": {
        "difficulty": "보통 (실용적이고 기술적인 연구도 많이 수용)",
        "focus": "매개체-기생충 상호작용 및 역학 연구, 실험 결과의 실무적 적용 가능성 및 데이터 신뢰도",
        "instructions": "기존에 잘 알려진 주제라도 데이터가 견고하고 역학적 가치가 있다면 점수를 합리적으로 부여(70~85점 가능)하십시오. 불필요하게 점수를 깎기보다 데이터 검증성에 초점을 맞추십시오.",
        "tier": "mid"
    },
    "PLoS Neglected Tropical Diseases": {
        "difficulty": "높음 (소외된 열대 질환 관련 대표 저널)",
        "focus": "NTD 질환에 대한 공중보건학적 의의, 병원성 분석, 역학적 유용성 및 실용성",
        "instructions": "공중보건적 임팩트와 병리 메커니즘을 동시에 균형 있게 평가하십시오. 소외 질환 퇴치에 어떻게 기여하는지 명확해야 높은 점수를 얻습니다.",
        "tier": "high"
    },
    "Frontiers in Microbiology": {
        "difficulty": "보통-높음 (넓은 스펙트럼의 미생물/면역 분야 저널)",
        "focus": "미생물학적 기초 연구, 면역학적 분석, 실험 방법론의 타당성과 명확성",
        "instructions": "데이터가 체계적이고 결론을 지지하기에 타당하다면 합리적인 점수대(65~80점)를 유연하게 제공하십시오.",
        "tier": "mid-high"
    },
    "Journal of Eukaryotic Microbiology": {
        "difficulty": "보통 (진핵 미생물 전문 저널)",
        "focus": "원생동물의 세포생물학, 분류학, 진화 및 유전학적 분석",
        "instructions": "생물학적 발견의 고유성에 가치를 두되, 데이터가 타당하고 체계적이라면 70점 내외의 긍정적인 점수를 부여하십시오.",
        "tier": "mid"
    },
    "Frontiers in Cellular and Infection Microbiology": {
        "difficulty": "보통-높음 (감염 및 세포 미생물학 전문 저널)",
        "focus": "숙주-기생충 상호작용 시 세포 수준의 기전 분석, 감염 모델의 정확성",
        "instructions": "감염 세포 수준의 메커니즘이 잘 입증되었다면 비교적 유연한 합격 점수를 수용할 수 있습니다.",
        "tier": "mid-high"
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

# ═══════════════════════════════════════════════════
#  Streamlit UI — "Specimen Slide" Design (v3)
# ═══════════════════════════════════════════════════

st.set_page_config(
    page_title="Parasitology Journal Acceptance Guide",
    page_icon="🧫",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Source+Sans+3:wght@300;400;500;600&family=Fira+Code:wght@400;500;600;700&display=swap');

    :root {
        --slate: #13151C;
        --frost: #1C1F2E;
        --frost-b: #242738;
        --eosin: #E8627C;
        --haema: #6366F1;
        --viridian: #34D399;
        --buff: #C8BFA9;
        --bone: #EDEDED;
        --dim: #6B7194;
        --rule: rgba(200, 191, 169, 0.1);
        --ff-display: 'DM Sans', sans-serif;
        --ff-body: 'Source Sans 3', sans-serif;
        --ff-data: 'Fira Code', monospace;
    }

    /* ── Page background ── */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: var(--slate) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--slate) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #161822 !important;
        border-right: 1px solid var(--rule) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: var(--ff-display) !important;
        color: var(--bone) !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        font-family: var(--ff-body) !important;
        color: var(--bone) !important;
    }

    p, li {
        font-family: var(--ff-body) !important;
        color: var(--bone) !important;
    }

    /* ── Labels — readable ── */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label {
        font-family: var(--ff-display) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: var(--buff) !important;
    }
    [data-testid="stFileUploader"] label {
        font-family: var(--ff-display) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: var(--buff) !important;
    }

    /* ── Inputs ── */
    [data-testid="stTextInputRootElement"] {
        background-color: var(--frost) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 6px !important;
        width: 100% !important;
    }
    [data-testid="stTextInputRootElement"]:focus-within {
        border-color: var(--haema) !important;
        box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.3) !important;
    }
    [data-testid="stTextInputRootElement"] input {
        background-color: transparent !important;
        border: none !important;
        color: var(--bone) !important;
        font-family: var(--ff-body) !important;
        font-size: 0.92rem !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    [data-testid="stTextInputRootElement"] input:focus {
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stTextInputRootElement"] button {
        background-color: transparent !important;
        border: none !important;
        color: var(--dim) !important;
        box-shadow: none !important;
    }
    [data-testid="stTextInputRootElement"] button:hover {
        background-color: transparent !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: var(--frost) !important;
        color: var(--bone) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 6px !important;
        font-family: var(--ff-body) !important;
        font-size: 0.92rem !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: var(--haema) !important;
        box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.3) !important;
    }
    .stSelectbox > div > div {
        background-color: var(--frost) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 6px !important;
    }

    /* ── File uploader container ── */
    [data-testid="stFileUploader"] > section {
        background-color: var(--frost) !important;
        border: 1px dashed rgba(200, 191, 169, 0.15) !important;
        border-radius: 8px !important;
    }
    /* Enforce hiding native browser input file text */
    [data-testid="stFileUploader"] section input[type="file"] {
        display: none !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        height: 0 !important;
    }
    /* Enforce heading font settings */
    h1, h2, h3, h4 {
        font-family: var(--ff-display) !important;
        color: var(--bone) !important;
    }

    /* ── Submit Button Only (Isolated to prevent styling other buttons) ── */
    .submit-wrap button {
        width: 100% !important;
        background: linear-gradient(135deg, var(--eosin), #D94F6E) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-family: var(--ff-display) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: opacity 0.2s ease, transform 0.15s ease !important;
    }
    .submit-wrap button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Slider ── */
    .stSlider [data-testid="stThumbValue"] {
        font-family: var(--ff-data) !important;
        color: var(--eosin) !important;
    }

    /* ── Alerts ── */
    .stAlert { border-radius: 6px !important; }
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        margin-top: 6px !important;
        margin-bottom: 6px !important;
        background-color: transparent !important;
        padding: 0 !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stAlert"] > div {
        padding: 6px 12px !important;
        min-height: unset !important;
        border-radius: 6px !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stNotificationContent"] {
        font-size: 0.82rem !important;
        line-height: 1.3 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stAlert"] svg {
        width: 14px !important;
        height: 14px !important;
    }
    
    /* ── Sidebar Text Input Align ── */
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] .stTextInput > div {
        width: 100% !important;
    }

    /* ── Containers ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--rule) !important;
        border-radius: 10px !important;
        background-color: var(--frost) !important;
    }

    /* ── Header ── */
    .pg-header {
        display: flex;
        align-items: stretch;
        gap: 1.5rem;
        padding: 1.8rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--rule);
    }
    .pg-header .accent-bar {
        width: 4px;
        flex-shrink: 0;
        border-radius: 2px;
        background: linear-gradient(180deg, var(--eosin), var(--haema));
    }
    .pg-header h1 {
        font-family: var(--ff-display) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: var(--bone) !important;
        margin: 0 0 0.35rem 0 !important;
        letter-spacing: -0.01em !important;
        line-height: 1.3 !important;
    }
    .pg-header .desc {
        font-family: var(--ff-body) !important;
        font-size: 0.9rem !important;
        color: var(--dim) !important;
        line-height: 1.5 !important;
        margin-bottom: 0.6rem;
    }
    .pg-header .meta {
        display: flex;
        gap: 0.5rem;
    }
    .pg-header .chip {
        font-family: var(--ff-data) !important;
        font-size: 0.62rem !important;
        color: var(--buff) !important;
        background: rgba(200, 191, 169, 0.06);
        border: 1px solid rgba(200, 191, 169, 0.1);
        padding: 0.15rem 0.55rem;
        border-radius: 3px;
        letter-spacing: 0.04em;
    }

    /* ── Input section heading ── */
    .input-heading {
        font-family: var(--ff-display) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--bone) !important;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .input-heading .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--eosin);
    }

    /* ── How it works panel ── */
    .how-panel {
        padding: 0.2rem 0;
    }
    .how-step {
        display: flex;
        gap: 0.8rem;
        padding: 0.7rem 0;
    }
    .how-step + .how-step {
        border-top: 1px solid var(--rule);
    }
    .how-step .num {
        font-family: var(--ff-data) !important;
        font-size: 0.65rem !important;
        color: var(--haema) !important;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 0.1rem;
    }
    .how-step .txt {
        font-family: var(--ff-body) !important;
        font-size: 0.85rem !important;
        color: var(--dim) !important;
        line-height: 1.5 !important;
    }
    .how-step .txt strong {
        color: var(--bone) !important;
        font-weight: 500 !important;
    }

    /* ── Section heading ── */
    .sec-heading {
        font-family: var(--ff-display) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--bone) !important;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--rule);
        margin: 2rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sec-heading .label {
        font-family: var(--ff-data) !important;
        font-size: 0.62rem !important;
        color: var(--dim) !important;
        background: rgba(99, 102, 241, 0.1);
        padding: 0.1rem 0.45rem;
        border-radius: 3px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Chromatography strip ── */
    .chroma-container {
        background: var(--frost);
        border: 1px solid var(--rule);
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }
    .chroma-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 1.2rem;
    }
    .chroma-header .journal {
        font-family: var(--ff-display) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--buff) !important;
    }
    .chroma-header .score-num {
        font-family: var(--ff-data) !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        letter-spacing: -0.03em;
    }
    .chroma-header .score-unit {
        font-family: var(--ff-data) !important;
        font-size: 0.75rem !important;
        color: var(--dim) !important;
        margin-left: 0.25rem;
    }
    .chroma-track {
        position: relative;
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg,
            #E8627C 0%,
            #F0A050 35%,
            #E8D44D 55%,
            #34D399 100%
        );
        margin-bottom: 0.6rem;
    }
    .chroma-marker {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 20px;
        background: var(--bone);
        border-radius: 2px;
        box-shadow: 0 0 8px rgba(237, 237, 237, 0.5);
    }
    .chroma-labels {
        display: flex;
        justify-content: space-between;
        font-family: var(--ff-data) !important;
        font-size: 0.62rem !important;
        color: var(--dim) !important;
        letter-spacing: 0.03em;
    }

    /* ── Result card ── */
    .res-card {
        background: var(--frost);
        border: 1px solid var(--rule);
        border-radius: 10px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
    }
    .res-card-title {
        font-family: var(--ff-display) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--buff) !important;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--rule);
    }

    /* ── Review items ── */
    .rv-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        padding: 0.65rem 0;
    }
    .rv-item + .rv-item { border-top: 1px solid var(--rule); }
    .rv-dot {
        flex-shrink: 0;
        width: 5px; height: 5px;
        border-radius: 50%;
        margin-top: 0.5rem;
    }
    .rv-dot.pos { background: var(--viridian); }
    .rv-dot.neg { background: var(--eosin); }
    .rv-text {
        font-family: var(--ff-body) !important;
        font-size: 0.9rem !important;
        color: var(--bone) !important;
        line-height: 1.55 !important;
    }

    /* ── Journal fit blockquote ── */
    .jf-quote {
        border-left: 2px solid var(--haema);
        padding: 0.8rem 1.2rem;
        margin: 0;
    }
    .jf-quote p {
        font-family: var(--ff-body) !important;
        font-size: 0.92rem !important;
        color: var(--dim) !important;
        line-height: 1.65 !important;
        font-style: italic;
    }

    /* ── Paper cards ── */
    .pub-card {
        background: var(--frost);
        border: 1px solid var(--rule);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s ease;
    }
    .pub-card:hover { border-color: rgba(200, 191, 169, 0.25); }
    .pub-year {
        font-family: var(--ff-data) !important;
        font-size: 0.62rem !important;
        color: var(--eosin) !important;
        letter-spacing: 0.06em;
    }
    .pub-title {
        font-family: var(--ff-display) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--bone) !important;
        line-height: 1.35 !important;
        margin: 0.3rem 0 0.4rem 0;
    }
    .pub-authors {
        font-family: var(--ff-body) !important;
        font-size: 0.78rem !important;
        color: var(--dim) !important;
    }
    .pub-abstract {
        font-family: var(--ff-body) !important;
        font-size: 0.8rem !important;
        color: rgba(107, 113, 148, 0.85) !important;
        line-height: 1.45 !important;
        margin-top: 0.5rem;
    }
    .pub-link { margin-top: 0.6rem; }
    .pub-link a {
        font-family: var(--ff-data) !important;
        font-size: 0.7rem !important;
        color: var(--haema) !important;
        text-decoration: none !important;
    }
    .pub-link a:hover { text-decoration: underline !important; }

    /* ── Action steps ── */
    .act-step {
        display: flex;
        gap: 0.9rem;
        padding: 0.7rem 0;
    }
    .act-step + .act-step { border-top: 1px solid var(--rule); }
    .act-num {
        font-family: var(--ff-data) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: var(--haema) !important;
        flex-shrink: 0;
        margin-top: 0.15rem;
    }
    .act-text {
        font-family: var(--ff-body) !important;
        font-size: 0.9rem !important;
        color: var(--bone) !important;
        line-height: 1.55 !important;
    }

    /* ── Tier badge ── */
    .tier-badge {
        display: inline-block;
        font-family: var(--ff-data) !important;
        font-size: 0.65rem !important;
        padding: 0.15rem 0.55rem;
        border-radius: 3px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .tier-top { background: rgba(232, 98, 124, 0.12); color: var(--eosin) !important; }
    .tier-high { background: rgba(240, 160, 80, 0.12); color: #F0A050 !important; }
    .tier-mid-high { background: rgba(99, 102, 241, 0.1); color: var(--haema) !important; }
    .tier-mid { background: rgba(52, 211, 153, 0.1); color: var(--viridian) !important; }

    /* ── Reduced motion ── */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════
#  Header
# ═════════════════════════════════════

st.markdown("""
<div class="pg-header">
    <div class="accent-bar"></div>
    <div class="content">
        <h1>Parasitology Journal<br/>Acceptance Guide</h1>
        <div class="desc">
            원고를 업로드하면 PubMed 합격 논문과 대조 분석 후, AI 가상 피어리뷰를 수행합니다.
        </div>
        <div class="meta">
            <span class="chip">Gemini 2.5 Flash</span>
            <span class="chip">PubMed Entrez API</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════
#  Sidebar
# ═════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔬 검색 설정")

    target_journal = st.selectbox(
        "타깃 저널",
        list(JOURNAL_PROFILES.keys())
    )

    # Tier badge below journal select
    tier = JOURNAL_PROFILES.get(target_journal, {}).get("tier", "mid")
    tier_labels = {"top": "Very High", "high": "High", "mid-high": "Mid-High", "mid": "Moderate"}
    st.markdown(
        f'<span class="tier-badge tier-{tier}">난이도: {tier_labels.get(tier, tier)}</span>',
        unsafe_allow_html=True
    )

    max_papers = st.slider("참고 논문 수", min_value=3, max_value=20, value=5)

    st.markdown("---")
    st.markdown("### 🔑 API 설정")
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
        st.success("API 키 설정 완료")
    else:
        st.error("Gemini API 키를 입력해 주세요. (미설정 상태)")

    st.markdown("---")
    st.markdown("### 🛡️ 데이터 보호")
    st.caption("원고 데이터는 Google AI Studio를 통해 일회성 처리되며, 학습에 반영하지 않고 분석합니다.")


# ═════════════════════════════════════
#  Input Area — 2-column layout
# ═════════════════════════════════════

in_col1, in_col2 = st.columns([3, 2], gap="large")

with in_col1:
    st.markdown(
        '<div class="input-heading"><div class="dot"></div>연구 정보 입력</div>',
        unsafe_allow_html=True
    )

    keywords = st.text_input(
        "연구 키워드",
        "",
        placeholder="예: Toxoplasma gondii, autophagy, macrophage"
    )

    uploaded_file = st.file_uploader(
        "원고 파일 업로드",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed"
    )

    default_text = ""
    if uploaded_file is not None:
        default_text = extract_text_from_file(uploaded_file)

    abstract_text = st.text_area(
        "원고 텍스트",
        value=default_text,
        height=200,
        placeholder="논문의 Abstract 또는 전체 원고를 입력하세요..."
    )

    st.markdown('<div class="submit-wrap">', unsafe_allow_html=True)
    submit_button = st.button("🧬 분석 시작 — PubMed 매칭 & AI 피어리뷰")
    st.markdown('</div>', unsafe_allow_html=True)

with in_col2:
    st.markdown(
        '<div class="input-heading"><div class="dot"></div>사용 방법</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="how-panel">
        <div class="how-step">
            <div class="num">01</div>
            <div class="txt"><strong>PubMed 매칭</strong><br/>키워드로 타깃 저널의 최신 합격 논문을 검색합니다.</div>
        </div>
        <div class="how-step">
            <div class="num">02</div>
            <div class="txt"><strong>AI 피어리뷰</strong><br/>합격 논문과 대조 분석하여 가상 리뷰 리포트를 생성합니다.</div>
        </div>
        <div class="how-step">
            <div class="num">03</div>
            <div class="txt"><strong>보완 전략</strong><br/>투고 성공률을 높이기 위한 구체적 액션 플랜을 제안합니다.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show selected journal info
    profile = JOURNAL_PROFILES.get(target_journal, {})
    st.markdown("---")
    st.markdown(f"**선택된 저널:** {target_journal}")
    st.caption(profile.get("focus", ""))


# ═════════════════════════════════════
#  Results
# ═════════════════════════════════════

if submit_button:
    if not keywords:
        st.warning("키워드를 입력해 주세요.")
    elif not abstract_text:
        st.warning("원고 내용을 입력해 주세요.")
    else:
        # ── PubMed search ──
        with st.spinner("PubMed에서 논문을 검색하고 있습니다..."):
            matching_papers = fetch_pubmed_papers(keywords, target_journal, max_results=max_papers)

        st.markdown(
            '<div class="sec-heading">📚 PubMed 매칭 논문 <span class="label">step 1</span></div>',
            unsafe_allow_html=True
        )

        if not matching_papers:
            st.info("매칭된 논문이 없습니다. 키워드를 조정해 보세요.")
        else:
            cols = st.columns(2)
            for idx, paper in enumerate(matching_papers):
                with cols[idx % 2]:
                    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/" if paper.get("pmid") else "#"
                    abs_text = paper["abstract"][:160] + "..." if len(paper["abstract"]) > 160 else paper["abstract"]
                    st.markdown(f"""
                    <div class="pub-card">
                        <div class="pub-year">{paper['year']}</div>
                        <div class="pub-title">{paper['title']}</div>
                        <div class="pub-authors">{paper['authors']}</div>
                        <div class="pub-abstract">{abs_text}</div>
                        <div class="pub-link"><a href="{paper_url}" target="_blank">PubMed →</a></div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── AI Review ──
        with st.spinner("AI 피어리뷰를 수행하고 있습니다..."):
            analysis = analyze_manuscript(
                abstract_text, target_journal, keywords,
                matching_papers, is_api_key_valid, selected_key
            )

        if "error" in analysis:
            st.error(analysis["error"])
        else:
            st.markdown(
                '<div class="sec-heading">📊 AI 피어리뷰 결과 <span class="label">step 2</span></div>',
                unsafe_allow_html=True
            )

            score = analysis.get("score", 50)
            score_color = (
                "var(--viridian)" if score >= 75
                else "var(--buff)" if score >= 50
                else "var(--eosin)"
            )

            # ── Signature: Chromatography strip ──
            st.markdown(f"""
            <div class="chroma-container">
                <div class="chroma-header">
                    <div class="journal">{target_journal}</div>
                    <div>
                        <span class="score-num" style="color: {score_color};">{score}</span>
                        <span class="score-unit">/ 100</span>
                    </div>
                </div>
                <div class="chroma-track">
                    <div class="chroma-marker" style="left: {score}%;"></div>
                </div>
                <div class="chroma-labels">
                    <span>Desk reject</span>
                    <span>Major revision</span>
                    <span>Minor revision</span>
                    <span>Accept</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Journal fit ──
            journal_fit = analysis.get("journal_fit", "N/A")
            st.markdown(f"""
            <div class="res-card">
                <div class="res-card-title">저널 적합성 분석</div>
                <div class="jf-quote"><p>{journal_fit}</p></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Strengths & Risks ──
            r_col1, r_col2 = st.columns(2)

            with r_col1:
                items_html = ""
                for s in analysis.get("strengths", []):
                    items_html += f'<div class="rv-item"><div class="rv-dot pos"></div><div class="rv-text">{s}</div></div>'
                st.markdown(f"""
                <div class="res-card">
                    <div class="res-card-title">강점 Strengths</div>
                    {items_html}
                </div>
                """, unsafe_allow_html=True)

            with r_col2:
                items_html = ""
                for r in analysis.get("reject_risks", []):
                    items_html += f'<div class="rv-item"><div class="rv-dot neg"></div><div class="rv-text">{r}</div></div>'
                st.markdown(f"""
                <div class="res-card">
                    <div class="res-card-title">리스크 Reject Risks</div>
                    {items_html}
                </div>
                """, unsafe_allow_html=True)

            # ── Action Plans ──
            steps_html = ""
            for i, plan in enumerate(analysis.get("action_plans", []), 1):
                steps_html += f'<div class="act-step"><div class="act-num">{i:02d}</div><div class="act-text">{plan}</div></div>'

            st.markdown(f"""
            <div class="res-card">
                <div class="res-card-title">보완 전략 Action Plan</div>
                {steps_html}
            </div>
            """, unsafe_allow_html=True)
