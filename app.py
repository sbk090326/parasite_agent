import os
import ssl
import streamlit as st
from dotenv import load_dotenv
from Bio import Entrez
from openai import OpenAI

# Bypass SSL certificate verification for NCBI Entrez and other requests
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# Set up Entrez Email
Entrez.email = os.getenv("ENTREZ_EMAIL", "your_email@example.com")
env_nvidia_key = os.getenv("NVIDIA_API_KEY", "")

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

import re
import math
from collections import Counter
import requests

def get_cosine_similarity(text1, text2):
    """
    Calculate the cosine similarity between word frequency vectors of text1 and text2.
    """
    if not text1 or not text2:
        return 0.0
    words1 = re.findall(r'\w+', text1.lower())
    words2 = re.findall(r'\w+', text2.lower())
    vec1 = Counter(words1)
    vec2 = Counter(words2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return float(numerator) / denominator

def enrich_papers_with_pubtator_and_pmc(papers):
    """
    Enrich paper list by:
    1. Converting PMIDs to PMCIDs using the PMC ID converter API.
    2. Fetching biological entities (Gene, Disease, Chemical) from PubTator Central.
    3. Fetching PMC Full Text (if available) from PubTator PMC BioC JSON API.
    """
    if not papers:
        return papers
        
    pmids = [p["pmid"] for p in papers if p.get("pmid")]
    if not pmids:
        return papers
        
    # Step 1: ID Conversion (PMID -> PMCID)
    pmid_to_pmcid = {}
    try:
        conv_url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={','.join(pmids)}&format=json&tool=bio_acceptance_guide&email={Entrez.email}"
        conv_res = requests.get(conv_url, timeout=10)
        if conv_res.status_code == 200:
            conv_data = conv_res.json()
            for record in conv_data.get("records", []):
                curr_pmid = record.get("pmid")
                curr_pmcid = record.get("pmcid")
                if curr_pmid and curr_pmcid:
                    pmid_to_pmcid[curr_pmid] = curr_pmcid
    except Exception as e:
        pass

    # Step 2 & 3: Fetch PubTator data (Abstracts or PMC Full Texts)
    for paper in papers:
        pmid = paper["pmid"]
        pmcid = pmid_to_pmcid.get(pmid)
        
        paper["pmcid"] = pmcid
        paper["full_text_available"] = False
        paper["full_text"] = ""
        paper["genes"] = []
        paper["diseases"] = []
        paper["chemicals"] = []
        paper["species"] = []
        paper["celllines"] = []
        
        try:
            if pmcid:
                url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmcids={pmcid}"
            else:
                url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids={pmid}"
                
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                docs = []
                if isinstance(data, list):
                    docs = data
                elif isinstance(data, dict):
                    docs = data.get("documents", [data]) if "documents" in data else [data]
                    
                if docs and len(docs) > 0:
                    doc = docs[0]
                    passages = doc.get("passages", [])
                    
                    genes = set()
                    diseases = set()
                    chemicals = set()
                    species = set()
                    celllines = set()
                    
                    full_text_parts = []
                    for passage in passages:
                        p_type = passage.get("infons", {}).get("type", "")
                        p_text = passage.get("text", "")
                        if p_text:
                            full_text_parts.append(p_text)
                        
                        # Extract annotations
                        for ann in passage.get("annotations", []):
                            ann_text = ann.get("text", "")
                            ann_type = ann.get("infons", {}).get("type", "")
                            if not ann_type:
                                ann_type = ann.get("type", "")
                                
                            if ann_text and ann_type:
                                ann_text_clean = ann_text.strip()
                                if ann_type.lower() == "gene":
                                    genes.add(ann_text_clean)
                                elif ann_type.lower() == "disease":
                                    diseases.add(ann_text_clean)
                                elif ann_type.lower() in ["chemical", "drug"]:
                                    chemicals.add(ann_text_clean)
                                elif ann_type.lower() == "species":
                                    species.add(ann_text_clean)
                                elif ann_type.lower() in ["cellline", "cell_line"]:
                                    celllines.add(ann_text_clean)
                                    
                    paper["genes"] = sorted(list(genes))[:15]
                    paper["diseases"] = sorted(list(diseases))[:15]
                    paper["chemicals"] = sorted(list(chemicals))[:15]
                    paper["species"] = sorted(list(species))[:15]
                    paper["celllines"] = sorted(list(celllines))[:15]
                    
                    if pmcid and len(full_text_parts) > 1:
                        full_text_raw = "\n\n".join(full_text_parts)
                        paper["full_text"] = full_text_raw[:8000]
                        paper["full_text_available"] = True
        except Exception as e:
            pass
    return papers

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

# 저널별 평가 프로필 정의 (난이도, 중점 심사 기준, 평가 지침, 루브릭 가중치)
JOURNAL_PROFILES = {
    "PLOS Pathogens": {
        "difficulty": "매우 높음 (Top-tier 병원체 분야 대표 저널, Impact Factor ~6.0대, 무조건적인 신규 분자 메커니즘 규명 필수)",
        "focus": "호스트-패스오젠 상호작용(Host-Pathogen Interactions)의 정밀한 세포학적/분자 생물학적 기전 규명 여부, 생체 내(In vivo) 마우스/동물 모델 검증 필수, CRISPR/Cas9 등 유전자 녹아웃(Knockout)/과발현(Overexpression) 대조군 데이터 구비 여부, 통계적 유의성(Biological replicates & power analysis)의 엄밀함.",
        "instructions": "세계적인 피어 리뷰어 수준으로 현미경 수준의 깐깐한 심사를 수행하십시오. 단순한 감염 현상 기술(Descriptive study)이나 데이터의 규모가 적은 연구는 과감히 50점 이하의 Desk Reject 판정을 내립니다. 가설이 명확하게 정립되고 모든 하위 실험들이 이 기전을 입증하기 위해 입체적으로 구성되어야만 70점 이상(Major Revision 이상)을 획득할 수 있습니다.",
        "tier": "top",
        "weights": {
            "novelty": 0.25,
            "methodology": 0.35,      # 방법론과 대조군에 매우 엄격한 가중치
            "data_completeness": 0.20,
            "journal_fit": 0.10,
            "presentation": 0.10
        },
        "weight_description": "분자 기전과 In vivo 동물 검증, CRISPR 대조군 완비 여부(Methodology & Controls)에 35%의 지배적인 가중치를 부여합니다."
    },
    "International Journal for Parasitology (IJP)": {
        "difficulty": "높음 (기생충학 분야 최고의 권위와 역사적인 저널, Impact Factor ~3.5대)",
        "focus": "분자기생충학(Molecular Parasitology), 기생충 면역 생리 메커니즘의 독창성, 기생 생물 모델(In vivo/In vitro)의 생물학적 타당성, 유전적 다양성 및 약물 내성 메커니즘 분석의 엄밀함.",
        "instructions": "학술적 참신함과 논리적 완결성이 극도로 높아야 75점 이상을 부여합니다. 실험 방법론에서 음성/양성 대조군(Negative/Positive Controls)이 확실하게 셋팅되었는지, 기생충 발달 단계(Life stages)별 특이적인 발견이 포함되었는지를 유심히 살피십시오. 단순 모니터링성 논문은 감점 요인입니다.",
        "tier": "high",
        "weights": {
            "novelty": 0.30,          # 참신한 면역 생리 기전에 가중치
            "methodology": 0.25,
            "data_completeness": 0.20,
            "journal_fit": 0.15,
            "presentation": 0.10
        },
        "weight_description": "기생충 특이적 생리/면역 메커니즘의 학술적 신규성(Novelty)에 30%, 대조군 완비 상태(Methodology)에 25%의 가중치를 둡니다."
    },
    "TRENDS IN PARASITOLOGY": {
        "difficulty": "매우 높음 (리뷰 및 트렌드 의견 제시 전문 고임팩트 저널, Impact Factor ~8.0대)",
        "focus": "기생충학 분야의 전반적인 패러다임을 바꿀 수 있는 수준의 개념적 진보(Conceptual Advance), 미래 연구 방향성의 설득력 있는 제시, 최신 발견들의 긴밀한 통합적 분석(Synthesis)과 입체적 시각화 도표(Figures/Models) 제안.",
        "instructions": "이 저널은 오리지널 연구 데이터(Original research data)를 투고하는 곳이 아니라 최신 트렌드를 정리하고 패러다임을 제안하는 리뷰 저널임을 명심하십시오. 따라서 원고가 기생충학계 전체에 유의미한 새로운 시각을 주는 '개념적 기여'가 보이지 않는다면 즉각 Reject하십시오. 매우 혁신적이고 넓은 학술적 통찰력을 보이는 경우에만 80점 이상의 고득점을 부여하십시오.",
        "tier": "top",
        "weights": {
            "novelty": 0.45,          # 패러다임적 개념 진보에 압도적 가중치
            "methodology": 0.05,      # 실험 원 데이터 비중은 극도로 최소화
            "data_completeness": 0.10,
            "journal_fit": 0.30,      # 학술 커뮤니티 트렌드 적합성
            "presentation": 0.10
        },
        "weight_description": "개념적 진보 및 통찰력(Novelty)에 45%, 리뷰 저널로서의 논지 전개 및 적합성(Journal Fit)에 30%의 지배적 가중치를 두며, 오리지널 데이터 방법론(Methodology) 비중은 5%로 제한합니다."
    },
    "Parasites & Vectors": {
        "difficulty": "보통 (매개체 및 기생충 질병 치료/역학 전문 OA 저널, Impact Factor ~3.0대)",
        "focus": "매개곤충(Vector)-기생체(Parasite) 상호작용, 역학적 현장 조사(Field Study) 데이터의 신뢰도 및 표본 크기(Sample size), 살충제 저항성(Insecticide resistance) 유전체 분석의 실무적 유용성.",
        "instructions": "기존에 잘 알려진 이론의 단순 현장 적용(예: 특정 지역 분포 조사)이더라도 표본 분석 규모가 충분하고 현장 데이터가 견고하다면 합리적으로 수용(70~85점 가능)하십시오. 복잡한 유전자 메커니즘 분석보다는 방법론의 투명성과 데이터의 실무적 방제 기여 가치에 엄격한 기준을 들이대십시오.",
        "tier": "mid",
        "weights": {
            "novelty": 0.15,
            "methodology": 0.20,
            "data_completeness": 0.30, # 현장 표본 분석 및 대규모 데이터 중시
            "journal_fit": 0.20,
            "presentation": 0.15
        },
        "weight_description": "대규모 현장 샘플 규모 및 통계적 확실성(Data & Statistics)에 30%, 실무 역학 통제 기여도(Journal Fit)에 20%의 가중치를 둡니다."
    },
    "PLoS Neglected Tropical Diseases": {
        "difficulty": "높음 (소외 열대 질환 분야의 독보적 대표 저널, Impact Factor ~3.8대)",
        "focus": "WHO 지정 소외 열대 질환(NTD)에 대한 공중보건학적 임팩트(Public Health Impact), 병리생태학적 분석의 깊이, 실제 진단법/치료제 개발의 임상적/실용적 유용성 및 질병 부담(Burden of Disease) 경감 기여성.",
        "instructions": "단순 실험실 데이터에 그치지 않고, 임상 현장이나 공중보건 역학 연구에 직접 연결될 수 있는 가치를 가졌는지 평가하십시오. 임상적 의미나 역학적 기여도가 불분명할 경우 점수를 낮게 매기십시오. 논문이 현장의 질병 퇴치 로드맵에 어떻게 기여하는지 요약 내용에 포함해야 합니다.",
        "tier": "high",
        "weights": {
            "novelty": 0.20,
            "methodology": 0.20,
            "data_completeness": 0.25,
            "journal_fit": 0.25,      # 공중보건학적 임팩트 및 질병 부담 기여도 중시
            "presentation": 0.10
        },
        "weight_description": "소외 질환 통제 임상/공중보건 유용성(Journal Fit)에 25%, 역학 데이터 규모 및 신뢰성(Data)에 25%의 균형 있는 가중치를 둡니다."
    },
    "Frontiers in Microbiology": {
        "difficulty": "보통-높음 (미생물학 분야의 거대 대표 저널, Impact Factor ~4.0대)",
        "focus": "미생물학/면역학적 기초 연구 데이터의 체계성, 오믹스(RNA-seq/Metagenomics) 분석 파이프라인의 방법론적 엄밀성, 통계적 검정의 타당성 및 재현성 확보.",
        "instructions": "실험 데이터가 결론을 충분히 뒷받침할 만큼 견고하고 방법론에 맹점이 없다면 비교적 유연하게 65~80점 범주 내에서 게재 가능성을 열어두어 평가하십시오. 다만 생정보학적 분석의 경우 표준 워크플로우를 충실히 준수했는지 비판적으로 보십시오.",
        "tier": "mid-high",
        "weights": {
            "novelty": 0.25,          # 기초 학술성 중시
            "methodology": 0.25,      # 표준 생정보학 파이프라인 엄밀성
            "data_completeness": 0.20,
            "journal_fit": 0.15,
            "presentation": 0.15
        },
        "weight_description": "미생물 학술적 기초 발견(Novelty)과 유전체 오믹스 파이프라인 등 표준 분석 방법론(Methodology)에 각각 25%의 가중치를 고루 적용합니다."
    },
    "Journal of Eukaryotic Microbiology": {
        "difficulty": "보통 (진핵 미생물 전문 정통 저널, Impact Factor ~2.0대)",
        "focus": "원생동물(Protozoa) 및 진핵 단세포 생물의 미세구조(Ultrastructure), 분자계통학적 분류(Phylogeny), 진화생물학적 신규성 및 분류동정의 정확성.",
        "instructions": "기하학적/미세 구조 분석이나 계통 분석 등 형태학적이고 진화적인 데이터의 정확성에 초점을 맞추어 엄격하게 심사하십시오. 고난도의 인비보 기능 분석이 없더라도 분류/동정 학설상의 중요한 발견이면 좋은 평가(70점 내외)를 매길 수 있습니다.",
        "tier": "mid",
        "weights": {
            "novelty": 0.20,
            "methodology": 0.30,      # 형태 구조 분석 및 계통 분류의 정확성에 가장 높은 비중
            "data_completeness": 0.20,
            "journal_fit": 0.15,
            "presentation": 0.15
        },
        "weight_description": "진핵 단세포 미세구조 기하학 분석 및 계통수(Phylogeny) 동정 분류 정확성(Methodology)에 30%의 가장 높은 가중치를 배정합니다."
    },
    "Frontiers in Cellular and Infection Microbiology": {
        "difficulty": "보통-높음 (감염 및 세포 미생물학 전문 저널, Impact Factor ~4.5대)",
        "focus": "세포 수준에서의 감염 및 면역학적 반응 기전(Cellular signaling pathways), 숙주 세포 침입 및 증식(Invasion & Proliferation) 메커니즘 규명 강도, 체외(In vitro) 3D 감염 모델의 진보성.",
        "instructions": "감염 과정 중 호스트 세포 내부의 구체적인 신호 전달 기전이 웨스턴 블롯, ELISA, 형광 이미지 등을 통해 세포 수준에서 입증되는지 엄격히 따지십시오. 입증 메커니즘이 모호하고 정량 분석이 미흡하다면 리젝트를 내리십시오.",
        "tier": "mid-high",
        "weights": {
            "novelty": 0.25,
            "methodology": 0.30,      # 웨스턴블롯/이미징 세포 메커니즘 검증 비중
            "data_completeness": 0.20,
            "journal_fit": 0.15,
            "presentation": 0.10
        },
        "weight_description": "숙주 세포 신호 전달 분석 및 체외 감염 모델 검증(Methodology)에 30%의 가중치를 적용합니다."
    }
}

def summarize_paper(title, full_text, api_key):
    """
    Summarize a single PMC full-text paper using LLM to extract key scientific points
    without dense experimental protocols to prevent safety filter triggers.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    prompt = f"""
당신은 세계적인 의학 및 생물학 전문 학술 에디터입니다. 아래 제공된 학술 논문의 본문 일부(PMC Full-Text)를 읽고, 다른 논문과의 학술적 대조 분석에 필요한 핵심 요약본을 한국어로 작성해 주세요.

[요구사항]
- 핵심 가설, 주요 분자/세포 생물학적 기전(Signaling pathways, genes, proteins), 주요 연구 결과, 그리고 학술적 결론을 중심으로 600자 내외로 조리 있게 작성하십시오.
- AI 안전 필터(Content Filter) 작동을 유발하는 구체적인 세포 배양/감염 프로토콜, 화학적 버퍼 혼합비 등 불필요하게 위험하거나 상세한 실험 프로토콜(Experimental protocols) 세부 사항은 완전히 배제하십시오. 오로지 발견 사실과 데이터 비교에 유용한 연구 사실에 집중하십시오.
- 오직 한국어로 구성된 요약 텍스트만 출력하십시오.

[논문 제목]
{title}

[논문 본문]
{full_text}
"""
    try:
        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
    except Exception:
        pass
    return None

def analyze_manuscript(abstract_text, target_journal, keywords, matching_papers, api_key_valid, api_key):
    """
    Run Llama-3.3-Nemotron-Super-49B LLM Agent to analyze manuscript peer-review and calculate success probability.
    """
    if not api_key_valid:
        return {
            "error": "유효한 NVIDIA API Key가 설정되지 않았습니다. 사이드바에 API 키를 입력해 주세요."
        }
        
    # AI 컨텍스트 윈도우 오버플로우 방지를 위해 업로드된 전체 원고의 길이를 최대 12,000자로 제한합니다.
    if len(abstract_text) > 12000:
        abstract_text = abstract_text[:12000] + "\n\n...[Manuscript truncated for LLM context window optimization]..."
        
    # 저널 프로필 획득
    journal_info = JOURNAL_PROFILES.get(target_journal, {
        "difficulty": "보통",
        "focus": "학술적 타당성 및 연구의 신뢰도",
        "instructions": "일반적인 저널 심사 기준을 따며, 연구 내용의 데이터 신뢰성을 검증하십시오."
    })
    
    # NVIDIA integrate.api.nvidia.com OpenAI-compatible 클라이언트 초기화
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    weights = journal_info.get("weights", {
        "novelty": 0.20,
        "methodology": 0.20,
        "data_completeness": 0.20,
        "journal_fit": 0.20,
        "presentation": 0.20
    })
    weight_desc = journal_info.get("weight_description", "모든 평가 요소에 동일한 20%의 가중치를 고르게 부여합니다.")
    
    use_summaries = True
    for attempt in range(2):
        try:
            background_context = ""
            for i, paper in enumerate(matching_papers, 1):
                sim_pct = int(paper.get("similarity", 0) * 100)
                
                # 본문(PMC)이 존재하고 요약본이 아직 캐싱되지 않은 경우, 요약 시도
                is_ft = paper.get("full_text_available", False) and paper.get("full_text") and use_summaries
                
                if is_ft:
                    if "full_text_summary" not in paper:
                        summary = summarize_paper(paper["title"], paper["full_text"], api_key)
                        if summary:
                            paper["full_text_summary"] = summary
                        else:
                            # 요약 실패 시 (또는 필터 차단 시) Abstract로 임시 대체 표기
                            paper["full_text_summary"] = None
                    
                    if paper.get("full_text_summary"):
                        ft_status = "Full-Text Summary (PMC Map-Reduced)"
                        content_to_use = paper["full_text_summary"]
                    else:
                        ft_status = "Abstract Only (Fallback)"
                        content_to_use = paper["abstract"]
                else:
                    ft_status = "Abstract Only"
                    content_to_use = paper["abstract"]
                
                background_context += f"Paper {i} (Similarity: {sim_pct}%, Source: {ft_status}):\n"
                background_context += f"Title: {paper['title']}\n"
                
                entities_str = []
                if paper.get("genes"):
                    entities_str.append(f"Genes: {', '.join(paper['genes'])}")
                if paper.get("diseases"):
                    entities_str.append(f"Diseases: {', '.join(paper['diseases'])}")
                if paper.get("chemicals"):
                    entities_str.append(f"Chemicals/Drugs: {', '.join(paper['chemicals'])}")
                if paper.get("species"):
                    entities_str.append(f"Species: {', '.join(paper['species'])}")
                if paper.get("celllines"):
                    entities_str.append(f"Cell Lines: {', '.join(paper['celllines'])}")
                    
                if entities_str:
                    background_context += f"Identified Entities: {'; '.join(entities_str)}\n"
                    
                background_context += f"Content:\n{content_to_use}\n\n"
                
            prompt = f"""
당신은 세계적인 기생충학(Parasitology) 및 미생물학 분야의 권위 있는 저널인 **{target_journal}**의 시니어 에디터이자 피어 리뷰어입니다.
귀하는 학술지의 높은 명성과 수준을 유지하기 위해 투고 원고를 **매우 엄격하고 깐깐하며 비판적인 시각**으로 채점하는 심사위원입니다. 조금이라도 실험 대조군이 부실하거나, 생체외/내(In vitro/In vivo) 입증 데이터에 논리적 공백이 있다면 과감하게 점수를 깎아야 합니다.

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

[저널별 맞춤형 다차원 평가 가중치 지침]
본 저널 **{target_journal}**의 중점 기준에 따른 각 심사 지표별 가중치는 다음과 같습니다:
- 학술적 신규성 및 개념적 진보 (novelty): {int(weights['novelty'] * 100)}%
- 방법론적 엄밀성 및 대조군 설계 (methodology): {int(weights['methodology'] * 100)}%
- 데이터 완결성 및 통계적 검정 (data_completeness): {int(weights['data_completeness'] * 100)}%
- 저널 적합성 및 학계 영향력 (journal_fit): {int(weights['journal_fit'] * 100)}%
- 논리 구조 및 작성 품질 (presentation): {int(weights['presentation'] * 100)}%
* 가중치 세부 설명: {weight_desc}

[연구 키워드]
{keywords}

[대상 논문 원고/초록]
{abstract_text}

[최신 유사 합격 논문 정보 (PubMed 검색 결과 및 PubTator/PMC 추출 데이터)]
{background_context}

---

**[요구사항 및 리포트 작성 가이드라인]**
1. **평가 어조**: 전문적이며 건설적이고 예리하게 지적해 주어야 합니다.
2. **합격 확률**: 0에서 100 사이의 숫자로 합격 가능성을 정량 예측해 주세요. (가중치 지침에 따른 sub_scores의 가중평균값에 완벽히 논리적으로 수렴해야 합니다.)
3. **다차원 평가 루브릭 스코어링**: 5가지 차원(novelty, methodology, data_completeness, journal_fit, presentation)에 대해 각각 0~100점 사이의 상세 점수와 구체적인 한국어 근거(rationale)를 제공하십시오.
4. **작성 언어**: 한국어로 출력해야 합니다. 단, 논문의 주요 생물학 용어(예: 유전자명, 경로명, 실험기법 등)는 영어 원문을 함께 기입해 주세요.
5. **리포트 구성 형식**: 아래 지정된 JSON 포맷으로 반드시 응답해야 하며, 그 외의 다른 텍스트는 절대 포함하지 마십시오.

```json
{{
  "score": 73,
  "sub_scores": {{
    "novelty": {{
      "score": 80,
      "rationale": "신규 분자 표적 또는 독창적 메커니즘을 제시하였으나..."
    }},
    "methodology": {{
      "score": 65,
      "rationale": "음성/양성 대조군은 양호하지만 In vivo 마우스 동물 실험이나 특정 유전자 넉아웃 대조군 데이터가 미흡..."
    }},
    "data_completeness": {{
      "score": 70,
      "rationale": "실험 반복 횟수에 대한 통계적 유의성 검정이 부족..."
    }},
    "journal_fit": {{
      "score": 85,
      "rationale": "숙주-기생체 면역 반응을 다루어 저널의 핵심 주제에 부합..."
    }},
    "presentation": {{
      "score": 75,
      "rationale": "논문의 구성은 매끄러우나 특정 용어(예: autophagy)에 대한 통일성이 필요..."
    }}
  }},
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
            response = client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )
            
            choice = response.choices[0]
            content = choice.message.content
            
            # 안전 필터 차단 감지
            if content is None or choice.finish_reason == "content_filter":
                # 요약본을 쓴 상태에서 차단된 경우, 초록 모드로 완전히 변경 후 재시도
                if use_summaries:
                    use_summaries = False
                    continue
                else:
                    prompt_len = len(prompt)
                    return {
                        "error": f"AI 안전 필터(Content Filter) 작동으로 분석 결과를 생성할 수 없습니다. (디버그 정보 - Finish Reason: {choice.finish_reason}, Content: {'None' if content is None else 'Present'}, Prompt Length: {prompt_len} 자). 입력하신 원고 내용에 병원균/면역회피 등 민감 학술 키워드가 다수 포함되어 차단되었을 가능성이 높습니다."
                    }
            
            import json
            import re
            
            clean_content = content.strip()
            # 마크다운 ```json 코드 블록 형태인 경우 내역 추출
            if clean_content.startswith("```"):
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_content, re.DOTALL)
                if match:
                    clean_content = match.group(1)
                else:
                    lines = clean_content.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_content = "\n".join(lines).strip()
            
            result = json.loads(clean_content)
            return result
        except Exception as e:
            # 예외 발생 시 본문 사용 상태였다면 초록 모드로 재시도
            if use_summaries:
                use_summaries = False
                continue
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

    /* ── Biological entity and similarity badges ── */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .ent-badge {
        font-family: var(--ff-data) !important;
        font-size: 0.62rem !important;
        padding: 0.1rem 0.35rem;
        border-radius: 3px;
        letter-spacing: 0.01em;
        font-weight: 500;
        border: 1px solid transparent;
        display: inline-block;
    }
    .badge-sim { background: rgba(200, 191, 169, 0.15); color: var(--buff) !important; border-color: rgba(200, 191, 169, 0.3); }
    .badge-gene { background: rgba(99, 102, 241, 0.1); color: var(--haema) !important; border-color: rgba(99, 102, 241, 0.25); }
    .badge-disease { background: rgba(232, 98, 124, 0.1); color: var(--eosin) !important; border-color: rgba(232, 98, 124, 0.25); }
    .badge-chem { background: rgba(52, 211, 153, 0.1); color: var(--viridian) !important; border-color: rgba(52, 211, 153, 0.25); }
    .badge-full { background: rgba(237, 237, 237, 0.1); color: var(--bone) !important; border-color: rgba(237, 237, 237, 0.25); }
    .badge-spec { background: rgba(240, 160, 80, 0.1); color: #F0A050 !important; border-color: rgba(240, 160, 80, 0.25); }
    .badge-cell { background: rgba(200, 191, 169, 0.08); color: var(--buff) !important; border-color: rgba(200, 191, 169, 0.2); }


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

    /* ── Sub-scores Rubric ── */
    .rubric-container {
        background: var(--frost);
        border: 1px solid var(--rule);
        border-radius: 10px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
    }
    .rubric-title {
        font-family: var(--ff-display) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: var(--buff) !important;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 0.5rem;
    }
    .rubric-item {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: 1rem;
    }
    .rubric-item:last-child {
        margin-bottom: 0;
    }
    .rubric-meta {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .rubric-label {
        font-family: var(--ff-display) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--bone) !important;
    }
    .rubric-score {
        font-family: var(--ff-data) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    .rubric-bar-bg {
        height: 6px;
        background: #1B1E2B;
        border-radius: 3px;
        overflow: hidden;
    }
    .rubric-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .rubric-rationale {
        font-family: var(--ff-body) !important;
        font-size: 0.78rem !important;
        color: var(--dim) !important;
        line-height: 1.45 !important;
        margin-top: 0.15rem;
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
            원고를 업로드하면 PubMed 최신 논문 검색 후 Reranking, PubTator 생물학적 엔티티 분석 및 PMC Full Text 기반으로 대조 분석하여 AI 피어리뷰를 수행합니다.
        </div>
        <div class="meta">
            <span class="chip">NVIDIA Nemotron Super 49B</span>
            <span class="chip">Cosine Reranker</span>
            <span class="chip">PubTator Entity Extractor</span>
            <span class="chip">PubMed & PMC Full Text</span>
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
        "NVIDIA API Key 입력",
        type="password",
        placeholder="build.nvidia.com에서 발급받은 API 키(nvapi-...)를 입력하세요...",
        help="입력하지 않으면 기본 서버 환경설정(.env)의 API 키를 사용합니다."
    )

    selected_key = user_api_key.strip() if user_api_key.strip() else env_nvidia_key.strip()
    is_api_key_valid = False
    if selected_key and not selected_key.startswith("your_nvidia_api_key") and selected_key != "":
        is_api_key_valid = True
        st.success("API 키 설정 완료")
    else:
        st.error("NVIDIA API 키를 입력해 주세요. (미설정 상태)")

    st.markdown("---")
    st.markdown("### 🛡️ 데이터 보호")
    st.caption("원고 데이터는 NVIDIA NIM API를 통해 일회성 처리되며, 학습에 반영하지 않고 분석합니다.")


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
        # ── PubMed search & Rerank ──
        with st.spinner("PubMed에서 관련 논문을 검색 및 Reranking하고 있습니다..."):
            # 1. 2배수 후보 검색 (최소 15개)
            max_candidates = max(max_papers * 2, 15)
            candidates = fetch_pubmed_papers(keywords, target_journal, max_results=max_candidates)
            
            if candidates:
                # 2. Cosine Similarity 계산 및 Rerank
                for paper in candidates:
                    paper_text = f"{paper['title']} {paper['abstract']}"
                    # 사용자 원고와 매칭도 평가
                    paper['similarity'] = get_cosine_similarity(abstract_text + " " + keywords, paper_text)
                
                # 유사도 기준 내림차순 정렬
                candidates.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                
                # 3. Top-N 선별
                selected_papers = candidates[:max_papers]
                
                # 4. PubTator 생물학적 엔티티 & PMC Full Text 추출로 강화
                with st.spinner("PubTator API 및 PMC Full-Text 데이터를 파싱하는 중..."):
                    matching_papers = enrich_papers_with_pubtator_and_pmc(selected_papers)
            else:
                matching_papers = []

        st.markdown(
            '<div class="sec-heading">📚 PubMed 매칭 논문 (Reranked & PubTator 분석) <span class="label">step 1</span></div>',
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
                    
                    # 배지 생성
                    sim_pct = int(paper.get("similarity", 0) * 100)
                    sim_badge = f'<span class="ent-badge badge-sim">유사도: {sim_pct}%</span>'
                    
                    ft_badge = ""
                    if paper.get("full_text_available"):
                        ft_badge = '<span class="ent-badge badge-full">PMC Full-Text</span>'
                        
                    gene_badges = "".join([f'<span class="ent-badge badge-gene">{g}</span>' for g in paper.get("genes", [])[:2]])
                    disease_badges = "".join([f'<span class="ent-badge badge-disease">{d}</span>' for d in paper.get("diseases", [])[:2]])
                    chem_badges = "".join([f'<span class="ent-badge badge-chem">{c}</span>' for c in paper.get("chemicals", [])[:2]])
                    spec_badges = "".join([f'<span class="ent-badge badge-spec">{s}</span>' for s in paper.get("species", [])[:2]])
                    cell_badges = "".join([f'<span class="ent-badge badge-cell">{cl}</span>' for cl in paper.get("celllines", [])[:2]])
                    
                    badges_html = f'<div class="badge-container">{sim_badge}{ft_badge}{spec_badges}{gene_badges}{disease_badges}{chem_badges}{cell_badges}</div>'
                    
                    st.markdown(f"""
                    <div class="pub-card">
                        <div class="pub-year">{paper['year']}</div>
                        <div class="pub-title">{paper['title']}</div>
                        <div class="pub-authors">{paper['authors']}</div>
                        {badges_html}
                        <div class="pub-abstract" style="margin-top:0.4rem;">{abs_text}</div>
                        <div class="pub-link" style="margin-top:0.4rem;"><a href="{paper_url}" target="_blank">PubMed →</a></div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── AI Review ──
        with st.spinner("AI 피어리뷰를 수행하고 있습니다..."):
            # 컨텍스트 오버플로우 방지 및 효율적인 대조 분석을 위해 유사도 기준 상위 10개 논문만 AI 분석용으로 사용
            review_papers = matching_papers[:10]
            analysis = analyze_manuscript(
                abstract_text, target_journal, keywords,
                review_papers, is_api_key_valid, selected_key
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

            # ── Sub-scores Rubric ──
            sub_scores = analysis.get("sub_scores", {})
            rubric_html = ""
            rubric_labels = {
                "novelty": "학술적 신규성 & 개념적 진보 (Novelty & Advance)",
                "methodology": "방법론적 엄밀성 & 대조군 (Methodology & Controls)",
                "data_completeness": "데이터 완결성 & 통계 (Data & Statistics)",
                "journal_fit": "저널 적합성 & 임팩트 (Journal Fit & Impact)",
                "presentation": "논리 구조 & 논문 작성 품질 (Presentation & Writing)"
            }
            
            for key, label in rubric_labels.items():
                score_data = sub_scores.get(key, {"score": 50, "rationale": "N/A"})
                sub_score = score_data.get("score", 50)
                sub_rationale = score_data.get("rationale", "N/A")
                
                # Determine color for sub-score bar
                bar_color = (
                    "var(--viridian)" if sub_score >= 75
                    else "var(--buff)" if sub_score >= 50
                    else "var(--eosin)"
                )
                
                rubric_html += f"""<div class="rubric-item">
<div class="rubric-meta">
<span class="rubric-label">{label}</span>
<span class="rubric-score" style="color: {bar_color};">{sub_score}점</span>
</div>
<div class="rubric-bar-bg">
<div class="rubric-bar-fill" style="width: {sub_score}%; background: {bar_color};"></div>
</div>
<div class="rubric-rationale">{sub_rationale}</div>
</div>"""
            
            st.markdown(f"""<div class="rubric-container">
<div class="rubric-title">📊 다차원 상세 심사 루브릭 (Evaluation Rubric)</div>
{rubric_html}
</div>""", unsafe_allow_html=True)

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
