import os
import tempfile
import urllib.parse
from datetime import date

import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import Tool

# ---- 환경 설정 ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# SerpAPI 웹검색 도구
# -----------------------------
def search_web():
    search = SerpAPIWrapper()

    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:8]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."

    return Tool(
        name="web_search",
        func=run_with_source,
        description=(
            "여행 관련 최신 정보(날씨, 환율, 영업시간, 휴무일, 비자, 현지 소식, 명소/식당/교통 등)를 검색합니다. "
            "제목+출처+링크+간단요약(snippet)으로 반환합니다."
        ),
    )

# -----------------------------
# 구글 지도 링크 생성 도구
# -----------------------------
def map_link_tool():
    def make_map_link(place_query: str) -> str:
        q = urllib.parse.quote_plus(place_query.strip())
        return f"https://www.google.com/maps/search/?api=1&query={q}"
    return Tool(
        name="map_link",
        func=make_map_link,
        description="입력한 장소/키워드에 대한 구글 지도 검색 링크를 생성합니다."
    )

# -----------------------------
# PDF 업로드 → 벡터DB → 검색 도구
# -----------------------------
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="업로드된 여행 관련 PDF(가이드북/티켓 규정 등)에서 정보를 검색합니다."
    )
    return retriever_tool

# -----------------------------
# 에이전트 실행 래퍼
# -----------------------------
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result["output"]

# -----------------------------
# 세션별 히스토리
# -----------------------------
def get_session_history(session_id):
    if "session_history" not in st.session_state:
        st.session_state.session_history = {}
    if session_id not in st.session_state.session_history:
        st.session_state.session_history[session_id] = ChatMessageHistory()
    return st.session_state.session_history[session_id]

# -----------------------------
# 이전 메시지 출력
# -----------------------------
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# 여행 프로필 문자열화
# -----------------------------
def serialize_travel_profile(profile: dict) -> str:
    parts = [
        f"여행 이름: {profile.get('trip_name') or '미정'}",
        f"출발지: {profile.get('origin') or '미정'}",
        f"목적지: {profile.get('destination') or '미정'}",
        f"여행기간: {profile.get('start_date')} ~ {profile.get('end_date')}",
        f"인원: {profile.get('party_size')}명",
        f"예산(일일): {profile.get('budget_per_day')} {profile.get('currency')}",
        f"이동 선호: {', '.join(profile.get('transport_prefs') or []) or '무관'}",
        f"관심사: {', '.join(profile.get('interests') or []) or '무관'}",
        f"식사/알레르기: {profile.get('diet') or '없음'}",
        f"메모: {profile.get('notes') or '없음'}",
    ]
    return "\n".join(parts)

# -----------------------------
# 메인 앱
# -----------------------------
def main():
    st.set_page_config(page_title="여행 도우미 ✈️", layout="wide", page_icon="🌎")

    with st.container():
        st.image('./Bot_Image.png', use_container_width=True)
        st.title("여행 도우미 ✈️ RAG + 실시간 검색")
        st.caption("PDF 가이드+티켓 규정은 RAG로, 최신 정보(날씨/환율/영업시간/휴무/현지뉴스)는 웹검색으로!")

    # 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ---- 사이드바: 키 & 파일 & 여행 프로필 ----
    with st.sidebar:
        st.header("설정")
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", type="password", placeholder="sk-...")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI API 키", type="password", placeholder="serp_api_key")
        st.markdown("---")

        st.subheader("여행 프로필")
        trip_name = st.text_input("여행 이름", value="가을 유럽 여행")
        origin = st.text_input("출발지(도시/공항 코드)", value="")
        destination = st.text_input("목적지(도시/국가/지역)", value="")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("출발일", value=date.today())
        with col2:
            end_date = st.date_input("귀국일", value=date.today())
        party_size = st.number_input("인원", min_value=1, max_value=20, value=2, step=1)
        budget_per_day = st.number_input("예산(1인/1일)", min_value=0, value=150, step=10)
        currency = st.selectbox("통화", ["KRW", "USD", "EUR", "JPY", "GBP"], index=0)
        interests = st.multiselect(
            "관심사",
            ["미식", "카페", "미술/박물관", "야경", "쇼핑", "자연/하이킹", "해변", "액티비티", "가족", "인스타 스팟"],
            default=["미식", "미술/박물관"],
        )
        transport_prefs = st.multiselect("이동 선호", ["도보", "대중교통", "택시/차량", "자전거"], default=["대중교통", "도보"])
        diet = st.text_input("식사 제한/알레르기(선택)", value="")
        notes = st.text_area("메모(선택)", value="", height=80)

        st.markdown("---")
        pdf_docs = st.file_uploader("여행 관련 PDF 업로드 (가이드/티켓/규정 등)", accept_multiple_files=True, key="pdf_uploader")

    # API 키 체크
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
        os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

        # 여행 프로필 구성
        travel_profile = {
            "trip_name": trip_name,
            "origin": origin,
            "destination": destination,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "party_size": int(party_size),
            "budget_per_day": int(budget_per_day),
            "currency": currency,
            "interests": interests,
            "transport_prefs": transport_prefs,
            "diet": diet,
            "notes": notes,
        }
        travel_context_str = serialize_travel_profile(travel_profile)

        # 도구 정의
        tools = []
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            tools.append(pdf_search)

        # 여행 프로필을 반환하는 도구 (에이전트가 호출하여 컨텍스트 확보)
        def get_travel_context(_: str) -> str:
            return travel_context_str

        tools.append(
            Tool(
                name="get_travel_context",
                func=get_travel_context,
                description="현재 사용자의 여행 프로필(출발지, 목적지, 날짜, 인원, 예산, 관심사 등)을 조회합니다."
            )
        )
        tools.append(search_web())
        tools.append(map_link_tool())

        # LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 프롬프트: 여행 특화
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "언어: 반드시 한국어.\n"
                        "당신은 여행 도우미 `AI 비서 톡톡이`입니다. 친근하고 간결하게, 항상 적절한 이모지(과하지 않게)를 포함하세요.\n"
                        "우선 `get_travel_context` 도구로 여행 프로필을 파악한 뒤, 질문 의도에 맞게 답하세요.\n"
                        "PDF에서 찾을 내용(가이드, 티켓 규정, 약관, 전자책 등)은 `pdf_search`를 우선 사용하세요.\n"
                        "날씨, 환율, 영업시간/휴무일, 현지 공휴일, 파업/이슈, 교통, 명소/식당 최신 정보가 필요하거나 "
                        "사용자 질문에 '최신', '현재', '오늘', '날씨', '환율', '영업시간', '휴무', '휴일' 등이 포함되면 반드시 `web_search`를 사용하세요.\n"
                        "장소를 제안할 때는 필요시 `map_link`로 구글지도 링크를 함께 제공하세요.\n"
                        "일정 제안 시 형식 예시:\n"
                        "- Day 1: 오전 … / 오후 … / 저녁 …\n"
                        "- 이동: … / 대략 소요시간: …\n"
                        "예산/이동 동선/휴무일 충돌을 주의하고, 대안 1~2개를 제시하세요.\n"
                        "안전/비자/보험/사전예약 필요 여부는 간단히 알림으로 표시하세요.\n"
                    )
                ),
                ("placeholder", "{chat_history}"),
                (
                    "human",
                    "{input}\n\n답변에는 이모지를 포함하고, 필요한 경우 웹검색/지도링크를 활용하세요."
                ),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 입력창
        user_input = st.chat_input("여행에 대해 무엇이 궁금하신가요?")
        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            # 과거 메시지 문자열 형태로 전달(간단 접근)
            if getattr(session_history, "messages", None):
                prev_msgs = [
                    {"role": m.type if hasattr(m, "type") else m["role"],
                     "content": m.content if hasattr(m, "content") else m["content"]}
                    for m in session_history.messages
                ]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            # 히스토리 저장 (표준 API 사용)
            session_history.add_user_message(user_input)
            session_history.add_ai_message(response)

        print_messages()

    else:
        st.warning("좌측 사이드바에 OpenAI / SerpAPI 키를 입력해주세요.")

if __name__ == "__main__":
    main()
