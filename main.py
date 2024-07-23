import streamlit as st
import torch
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnableSequence
from config import SYSTEM_PROMPT, QUERY_WRAPPER_PROMPT
from load_llm import LLMLoader


st.title("나의 챗GPT ")


#모델은 한번만 로드하고 재사용하도록 코드 수정
@st.cache_resource
def load_model():
    llm_loader = LLMLoader(SYSTEM_PROMPT, QUERY_WRAPPER_PROMPT)
    return llm_loader.load_llm(device="cuda" if torch.cuda.is_available() else "cpu")

llm = load_model()

#처음 한번만 실행하기 위한 코드
if "messages" not in st.session_state:
    #대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content = message))


###############################################################################
def create_chain():
    #3가지를 묶어서 체인으로 반환예정
    #prompt | llm | output_parser
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            ("user", "#Question:\n{question}")
        ]
    )

    # llm 로더 초기화 및 LLM 로드
    #llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    # llm_loader = LLMLoader(SYSTEM_PROMPT, QUERY_WRAPPER_PROMPT)
    # llm = llm_loader.load_llm()

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    #chain = prompt | llm | output_parser
    #chain = LLMChain(llm=llm, prompt=prompt, output_parser= output_parser)
    chain = RunnableSequence(prompt | llm | output_parser)

    return chain
###############################################################################


print_messages()


#사용자 입력
user_input = st.chat_input("입력하세요")

if user_input:
    # 사용자 입력
    st.chat_message("user").write(user_input)

    # chain을 생성
    chain = create_chain()
    ai_answer = chain.invoke({"question": user_input})

    print("Type of ai_answer:", type(ai_answer))
    print("Content of ai_answer:", ai_answer)

    # AI 답변
    ai_response = str(ai_answer)
    st.chat_message("assistant").write(ai_response)
    

    # 대화기록 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)