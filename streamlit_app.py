import streamlit as st
import os
import json
from pydantic import BaseModel, Field # Pydantic 라이브러리, AgentState 정의에 필수
from typing import Literal

# LangChain/LangGraph Components
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# ==============================================================================
# 1. 환경 설정 및 초기화 (Setup & Initialization)
# ==============================================================================

# 환경 변수 설정 (API Key 관리)
# 실제 서비스에서는 보안을 위해 환경 변수로 관리해야 합니다.
# Streamlit secrets 또는 OS 환경 변수를 사용하세요.
# 여기서는 예시를 위해 Mocking합니다.
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Firebase 전역 변수 설정 (Canvas 환경 필수 요소)
# 이 예제에서는 Firestore를 직접 사용하지 않으나, 환경 호환성을 위해 변수를 정의합니다.
try:
    firebaseConfig = json.loads(__firebase_config)
    appId = __app_id
    initialAuthToken = __initial_auth_token
except NameError:
    # 로컬 테스트 환경을 위한 더미 값
    firebaseConfig = {}
    appId = 'default-app-id'
    initialAuthToken = None
    
# LLM 초기화 (GPT-4o-mini 사용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ==============================================================================
# 2. RAG (Retrieval-Augmented Generation) 설정
# ==============================================================================

# 2-1. 임시 여행지 지식 데이터 (RAG Source)
TRAVEL_KNOWLEDGE = [
    "파리(Paris)는 에펠탑, 루브르 박물관, 노트르담 대성당, 몽마르트 언덕으로 유명하며, 예술과 미식의 중심지입니다. 주요 미식은 마카롱, 크루아상, 에스카르고입니다. 평균 예산은 1일당 150~250 유로입니다.",
    "서울(Seoul)은 경복궁, 남산타워, 명동, 홍대, 강남 등 전통과 현대가 공존하는 도시입니다. 한강에서 라이딩을 즐기거나 K-팝 성지 순례도 인기입니다. 주요 음식은 김치찌개, 비빔밥, 치맥이며, 평균 예산은 1일당 10만~15만 원입니다.",
    "도쿄(Tokyo)는 신주쿠, 시부야, 아사쿠사 등 다양한 매력을 지니고 있습니다. 오타쿠 문화의 아키하바라와 미슐랭 레스토랑이 많습니다. 주요 미식은 스시, 라멘, 텐푸라입니다. 평균 예산은 1일당 15,000~25,000 엔입니다.",
    "여행지 선정 시 가장 중요한 요소는 예산과 기간입니다. 예산이 제한적이라면 동남아시아 지역이나 국내 여행을 고려하는 것이 좋고, 기간이 길다면 유럽이나 미주 지역을 추천합니다. 항상 현지 날씨를 확인하세요."
]

# 2-2. Vector Store 생성 (FAISS 활용)
@st.cache_resource
def setup_vector_store():
    """RAG에 사용할 Vector Store를 설정합니다."""
    # Document 객체 생성
    docs = [Document(page_content=t) for t in TRAVEL_KNOWLEDGE]
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # FAISS Vector Store 생성 및 인덱싱
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = setup_vector_store()
retriever = vectorstore.as_retriever()

# RAG Chain 정의
retrieval_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(
        llm,
        ChatPromptTemplate.from_messages([
            ("system", "당신은 세계적인 여행 전문가입니다. 아래 '검색된 정보'와 '사용자의 여행 요청'을 바탕으로 여행 계획 수립에 필요한 핵심 정보를 한국어로 요약하여 제공하세요. 검색된 정보가 없을 경우, 일반적인 지식을 활용하여 답변하세요."),
            ("context", "{context}"),
            ("human", "사용자의 여행 요청: {input}"),
        ])
    )
)

# ==============================================================================
# 3. LangGraph & Multi-Agent Flow 정의 (Planner + Tool)
# ==============================================================================

# 3-1. Tool 정의 (ReAct를 위한 RAG 기능 도구)
@tool
def research_travel_info(query: str) -> str:
    """
    여행 계획에 필요한 특정 도시, 음식, 명소, 예산 등의 정보를 검색합니다.
    예: '파리의 주요 명소는 뭐야?'
    """
    st.session_state.messages.append(SystemMessage(content=f"🤔 **리서치 도구 사용:** '{query}'에 대한 지식을 검색합니다."))
    
    # RAG Chain 실행
    result = retrieval_chain.invoke({"input": query})
    
    # 검색된 정보를 사용자에게 보여주기 위해 세션에 추가
    retrieved_context = result['context']
    
    # 검색된 정보가 없을 경우 처리
    if not retrieved_context:
        return "검색 결과가 없습니다. 일반 지식을 활용하여 답변하세요."

    # context를 문자열로 합치고, 간결하게 요약하여 반환
    context_str = "\n".join([doc.page_content for doc in retrieved_context])

    # 검색된 문서들을 활용하여 답변을 생성하는 LLM 호출
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 검색된 정보를 분석하는 전문가입니다. 아래 '검색된 문서들'을 기반으로 사용자의 요청에 대해 필요한 핵심 내용을 한국어로 요약하여 제공하세요."),
        ("human", f"검색 요청: {query}\n\n검색된 문서들:\n{context_str}"),
    ])
    
    # LLM이 검색된 정보를 요약하도록 한 번 더 호출
    summary = llm.invoke(planner_prompt).content
    st.session_state.messages.append(SystemMessage(content=f"✅ **리서치 결과:** {summary}"))
    return summary


# 3-2. Agent State 정의
class AgentState(BaseModel):
    """
    에이전트 상태를 정의합니다. Pydantic을 사용하여 상태 관리를 명확하게 합니다.
    """
    # 사용자의 원본 요청
    initial_request: str = Field(description="사용자의 최초 여행 계획 요청")
    # 대화 기록 (멀티턴 대화를 위한 Memory)
    chat_history: list[AnyMessage] = Field(description="현재까지의 대화 메시지 리스트")
    # 에이전트가 수집한 핵심 정보
    contextual_data: str = Field(description="리서치와 대화를 통해 수집된 핵심 여행 정보")
    # LangGraph에서 사용할 다음 액션
    next_action: Literal["tool_call", "generate_itinerary", "final_answer"] = Field(description="다음 수행할 액션")
    # ReAct Tool 호출 시 사용할 인보케이션 (LangGraph prebuilt ToolExecutor용)
    tool_invocation: ToolInvocation | None = Field(description="호출할 도구 및 인자")

# 3-3. Agent Nodes 정의

# A. 플래너 노드 (Planner Node - ReAct Logic)
def planner_agent(state: AgentState):
    """
    사용자의 요청을 분석하고, 리서치(Tool)가 필요한지 판단합니다.
    """
    st.session_state.messages.append(SystemMessage(content="🧠 **플래너 에이전트:** 요청 분석 및 도구 사용 여부를 판단합니다."))
    
    # LangChain Runnable을 사용하여 ReAct 로직 구현
    prompt = ChatPromptTemplate.from_messages([
        # Few-shot Prompting 및 Role-playing
        ("system", """
         당신은 여행 계획 수립을 위한 최고 수준의 플래닝 에이전트입니다.
         당신의 임무는 사용자의 요청을 분석하여, 여행 계획을 수립하기 전에 'research_travel_info' 도구의 사용이 필요한지 결정하는 것입니다.
         
         1. 도구 사용이 필요하다고 판단되면, 적절한 'tool_call'을 생성하여 다음 상태로 넘깁니다. 도구에 제공할 인자는 매우 구체적이어야 합니다.
         2. 도구 사용이 필요하지 않거나, 이미 충분한 정보가 수집되었다고 판단되면, 다음 단계인 'generate_itinerary'로 넘깁니다.

         **사용 가능한 도구:** research_travel_info(query: str)
         
         **Chain-of-Thought 예시:**
         요청: 2박 3일 서울 여행 계획을 짜줘.
         Thought: 서울의 주요 명소와 예산 정보가 필요하다.
         Action: research_travel_info(query='서울의 주요 명소, 음식, 예상 경비')
         
         **Output 형식 (반드시 JSON으로 응답):**
         {{ "next_action": "tool_call" | "generate_itinerary", "tool_invocation": ToolInvocation | null, "thought": "당신의 추론 과정" }}
         """),
        ("human", f"사용자의 요청: {state.initial_request}"),
    ]).partial(tools=[research_travel_info])
    
    # LLM 호출 및 JSON 응답 파싱
    parser = JsonOutputParser(pydantic_object=AgentState)
    
    # CoT를 유도하기 위해 `response_format`을 사용하지 않고 LLM에게 JSON 출력을 요청
    response = llm.invoke(prompt)
    
    try:
        # LLM 응답에서 JSON 부분을 추출하여 파싱 시도
        # LangChain의 PydanticOutputParser가 자동 변환을 시도하지만,
        # LLM이 문자열 안에 JSON을 넣는 경우를 대비해 직접 파싱 시도
        json_content = response.content.strip().split("```json")[1].split("```")[0].strip()
        parsed_data = json.loads(json_content)
        
        # Pydantic 모델에 맞게 데이터 정리
        tool_invocation = None
        if parsed_data.get("tool_invocation"):
            # ToolInvocation 객체로 변환
            inv = parsed_data["tool_invocation"]
            tool_invocation = ToolInvocation(
                tool=inv["tool"], 
                tool_input=inv["tool_input"], 
                id=inv.get("id")
            )
        
        # 상태 업데이트
        new_state = state.copy(update={
            "next_action": parsed_data.get("next_action", "generate_itinerary"), # 기본값 설정
            "tool_invocation": tool_invocation,
        })
        return new_state
        
    except Exception as e:
        st.session_state.messages.append(SystemMessage(content=f"⚠️ **JSON 파싱 오류. 강제 진행:** {e}"))
        # 오류 시 강제로 일정 생성 단계로 이동
        return state.copy(update={"next_action": "generate_itinerary"})

# B. 도구 실행 노드 (Tool Execution Node)
def execute_tool(state: AgentState):
    """LangGraph Prebuilt ToolExecutor를 사용하여 도구를 실행합니다."""
    tool_executor = ToolExecutor([research_travel_info])
    
    if state.tool_invocation:
        # 도구 실행
        output = tool_executor.invoke(state.tool_invocation)
        
        # 도구 실행 결과를 컨텍스트에 추가
        new_context = state.contextual_data + "\n\n[검색 결과]\n" + str(output)
        
        # 도구를 실행했으니, 다음은 일정 생성으로 넘어가도록 상태 업데이트
        return state.copy(update={
            "contextual_data": new_context,
            "next_action": "generate_itinerary" # 리서치 후 항상 일정 생성으로 이동
        })
    
    # 도구 호출 정보가 없으면, 플래너가 잘못된 상태를 넘겼으므로 강제 진행
    return state.copy(update={"next_action": "generate_itinerary"})


# C. 일정 생성 노드 (Itinerary Generation Node)
def itinerary_generator(state: AgentState):
    """
    수집된 정보를 바탕으로 최종 여행 일정을 생성하고 사용자에게 응답합니다.
    """
    st.session_state.messages.append(SystemMessage(content="✍️ **일정 생성 에이전트:** 수집된 정보를 바탕으로 최종 일정을 작성합니다."))
    
    # 프롬프트 최적화: 역할 부여 + 수집된 정보 활용
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         당신은 전문 여행 가이드이자 일정 작성 전문가입니다.
         아래 '사용자 요청'과 '수집된 정보'를 종합하여, **마크다운 형식의 상세한 여행 일정을 한국어로 작성하세요.**
         
         일정은 일자별로(예: 1일차, 2일차) 구분하고, 각 활동에 대해 간결한 설명과 함께 예상 시간, 장소, 핵심 팁(예: 예산, 교통 등)을 포함해야 합니다.
         수집된 정보가 부족하더라도, 일반적인 여행 지식을 활용하여 완벽한 일정을 완성하세요.
         """),
        ("context", f"**수집된 정보:**\n{state.contextual_data}"),
        ("human", f"**사용자 요청:**\n{state.initial_request}"),
    ])
    
    # LLM 호출
    itinerary_response = llm.invoke(generator_prompt).content
    
    # 상태 업데이트: 최종 답변을 chat_history에 추가하고 flow를 종료
    new_history = state.chat_history + [HumanMessage(content=state.initial_request), SystemMessage(content=itinerary_response)]
    return state.copy(update={
        "chat_history": new_history,
        "next_action": "final_answer", # 최종 답변 완료
    })

# 3-4. LangGraph Flow 구성
def build_graph():
    """LangGraph 상태 그래프를 구축합니다."""
    # Graph 초기화
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("planner", planner_agent)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("itinerary_generator", itinerary_generator)

    # 엣지(Edge) 정의: 노드 간 이동 경로
    # 1. 시작 -> 플래너
    workflow.add_edge(START, "planner")

    # 2. 플래너 노드의 결과에 따라 이동
    def route_planner(state: AgentState):
        if state.next_action == "tool_call":
            return "execute_tool"
        elif state.next_action == "generate_itinerary":
            return "itinerary_generator"
        else:
            return END # 예외 상황 대비
    
    workflow.add_conditional_edges(
        "planner", 
        route_planner
    )
    
    # 3. 도구 실행 노드 -> 일정 생성 노드 (리서치 후에는 항상 일정 생성)
    workflow.add_edge("execute_tool", "itinerary_generator")
    
    # 4. 일정 생성 노드 -> 종료 (최종 답변 완료)
    workflow.add_edge("itinerary_generator", END)
    
    # 메모리 저장을 위한 Checkpoint 설정 (멀티턴 대화/상태 저장을 위한 선택 사항)
    memory = SqliteSaver.from_conn_string(":memory:")
    
    # 그래프 컴파일
    app = workflow.compile(checkpointer=memory)
    return app

# Graph 컴파일 (Streamlit 리소스 캐시 활용)
@st.cache_resource
def get_graph():
    return build_graph()

graph_app = get_graph()

# ==============================================================================
# 4. Streamlit UI (서비스 개발)
# ==============================================================================

st.set_page_config(
    page_title="AI 여행 플래너 ✈️ (LangChain/RAG/Streamlit)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자에게 LLM 호출 및 RAG 구성을 안내합니다.
with st.sidebar:
    st.header("과제 요구사항 구현")
    st.markdown("""
    **✅ 구현 요소:**
    - **Streamlit**: 대화형 UI 구현
    - **LangChain/LangGraph**: Multi-Agent Flow 설계
    - **RAG**: 내장 지식(FAISS) 기반 정보 검색
    - **Prompt Engineering**: 역할 부여, CoT, Few-shot 활용
    - **ReAct (Tool)**: RAG 검색 기능을 도구로 활용
    - **API Key 관리**: 환경 변수 (Mock)
    """)
    st.info("💡 **사용법:** 여행지, 기간, 예산, 관심사를 포함하여 상세한 요청을 입력하세요.")
    
    # Chat History 초기화 버튼
    if st.button("새로운 여행 계획 시작", use_container_width=True):
        st.session_state.messages = [SystemMessage(content="안녕하세요! 저는 AI 여행 플래너입니다. 어떤 여행 계획을 도와드릴까요? (예: 3박 4일 파리 가족 여행, 예산 500만원)")]
        st.session_state.run_id = None
        st.session_state.initial_request = None
        st.success("새로운 대화가 시작되었습니다.")


st.title("AI 여행 플래너 ✈️")
st.caption("요구사항에 맞춰 LangGraph Multi-Agent, RAG, Streamlit으로 구현되었습니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="안녕하세요! 저는 AI 여행 플래너입니다. 어떤 여행 계획을 도와드릴까요? (예: 3박 4일 파리 가족 여행, 예산 500만원)")]
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "initial_request" not in st.session_state:
    st.session_state.initial_request = None


# 이전 대화 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, SystemMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    # LangGraph 내부 메시지는 표시하지 않음 (선택적으로 표시 가능)


# 사용자 입력 처리
if prompt := st.chat_input("여행 계획을 입력하세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 응답을 위한 Chat Assistant 영역
    with st.chat_message("assistant"):
        st.session_state.messages.append(SystemMessage(content="여행 계획 생성을 시작합니다..."))
        # 로딩 스피너
        with st.spinner("최고의 여행 일정을 만들기 위해 AI Agent들이 협력 중입니다..."):
            
            # LangGraph 실행을 위한 초기 상태
            initial_state = AgentState(
                initial_request=prompt,
                chat_history=[],
                contextual_data="",
                next_action="planner",
                tool_invocation=None
            )
            
            # LangGraph 실행 (stream 사용)
            # 여기서는 state.chat_history에 최종 답변만 담기 때문에 스트리밍은 단순 로딩 효과용입니다.
            # 복잡한 멀티턴 대화나 상태 저장/복구 기능은 `checkpointer`가 담당합니다.
            
            final_state = None
            try:
                # 그래프 실행
                for s in graph_app.stream(initial_state, config={"configurable": {"thread_id": "my_travel_plan"}}):
                    # 상태 변화를 감지하고 최종 상태를 저장합니다.
                    final_state = s
                    
                # 최종 상태에서 생성된 답변 추출
                if final_state and "itinerary_generator" in final_state:
                    # itinerary_generator 노드가 최종 상태에 도달했을 때의 chat_history를 사용
                    response_message = final_state["itinerary_generator"].chat_history[-1].content
                    
                    # 최종 답변 표시 및 세션 업데이트
                    st.markdown(response_message)
                    st.session_state.messages = st.session_state.messages[:-1] + [SystemMessage(content=response_message)]
                else:
                    error_message = "죄송합니다. 여행 일정 생성 중 문제가 발생했습니다. 요청을 다시 시도해 주세요."
                    st.markdown(error_message)
                    st.session_state.messages = st.session_state.messages[:-1] + [SystemMessage(content=error_message)]

            except Exception as e:
                error_message = f"에이전트 실행 중 심각한 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages = st.session_state.messages[:-1] + [SystemMessage(content=error_message)]
