import streamlit as st 
import chat

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "Agentic Workflow (Tool Use)": [
        "Agent를 이용해 다양한 툴을 사용할 수 있습니다. 여기에서는 날씨, 시간, 도서추천, 인터넷 검색을 제공합니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Flow": [
        "Bedrock Flow를 이용하여 Workflow를 구현합니다."
    ],
    "Agent": [
        "Bedrock Agent를 이용하여 RAG를 포함한 Workflow를 구현합니다."
    ],
    "번역하기": [
        "한국어와 영어에 대한 번역을 제공합니다. 한국어로 입력하면 영어로, 영어로 입력하면 한국어로 번역합니다."        
    ],
    "문법 검토하기": [
        "영어와 한국어 문법의 문제점을 설명하고, 수정된 결과를 함께 제공합니다."
    ],
    "이미지 분석": [
        "이미지를 업로드하면 이미지의 내용을 요약할 수 있습니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Nova Pro를 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 일상적인 대화와 각종 툴을 이용해 Agent를 구현할 수 있습니다." 
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/bedrock-agent)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "Agentic Workflow (Tool Use)", "RAG", "Flow", "Agent", "번역하기", "문법 검토하기", '이미지 분석'], index=0
    )   
    st.info(mode_descriptions[mode][0])
    # limit = st.slider(
    #     label="Number of cards",
    #     min_value=1,
    #     max_value=mode_descriptions[mode][2],
    #     value=6,
    # )

    # selectionbox
    # option = st.selectbox(
    #     '🖊️ 대화 형태를 선택하세요. ',
    #     ('일상적인 대화', 'Agentic Workflow (Tool Use)', 'Flow', 'Agent', '번역하기', '문법 검토하기', '이미지 분석')
    # )

    print('mode: ', mode)

    st.subheader("🌇 이미지 업로드")
    uploaded_file = st.file_uploader("이미지 선택", type=["png", "jpg", "jpeg"])

    st.success("Connected to Nova Pro", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    print('clear_button: ', clear_button)

st.title('🔮 '+ mode)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    uploaded_file = None
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()
file_name = ""
# Preview the uploaded image in the sidebar
if uploaded_file and clear_button==False and mode == '이미지 분석':
    st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)

    file_name = uploaded_file.name
    image_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
    print('image_url: ', image_url)
        
# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")

    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)            
            response = st.write_stream(stream)
            print('response: ', response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

            chat.save_chat_history(prompt, response)

        elif mode == 'Agentic Workflow (Tool Use)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_agent_executor2(prompt, st, debugMode)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

                chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_rag_with_knowledge_base(prompt)        
                st.write(response)
                print('response: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

                chat.save_chat_history(prompt, response)

        elif mode == 'Flow':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_flow(prompt)        
                st.write(response)
                print('response: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

                chat.save_chat_history(prompt, response)
        
        elif mode == 'Agent':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_bedrock_agent(prompt)        
                st.write(response)
                print('response: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

                chat.save_chat_history(prompt, response)
        
        elif mode == '번역하기':
            response = chat.translate_text(prompt)
            st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)

        elif mode == '문법 검토하기':
            response = chat.check_grammer(prompt)
            st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)

        elif mode == '이미지 분석':
            if uploaded_file is None or uploaded_file == "":
                st.error("파일을 먼저 업로드하세요.")
                st.stop()

            else:                
                with st.status("thinking...", expanded=True, state="running") as status:
                    summary, img_base64 = chat.summary_image(file_name, prompt)
                    st.write(summary)
                    print('summary: ', summary)
                    st.session_state.messages.append({"role": "assistant", "content": summary})

                    chat.save_chat_history(prompt, summary)

                    text = chat.extract_text(img_base64)
                    st.write(text)
                    st.session_state.messages.append({"role": "assistant", "content": text})

                    st.rerun()

                    chat.save_chat_history(prompt, text)

        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            print('response: ', response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        


