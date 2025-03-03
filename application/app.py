import streamlit as st 
import chat
import utils
import knowledge_base as kb

import cost_analysis as cost

# logging
logger = utils.CreateLogger("streamlit")

# title
st.set_page_config(page_title='Bedrock Agent', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Flow": [
        "Bedrock Flow를 이용하여 Workflow를 구현합니다."
    ],
    "Agent": [
        "Bedrock Agent를 이용하여 Workflow를 구현합니다."
    ],
    "Agent with Knowlege Base": [
        "Bedrock Agent와 Knowledge Base를 이용하여 Workflow를 구현합니다."
    ],
    "Multi Agent Collaboration": [
        "Multi Bedrock Collabotion을 통해 suprervisor가 여러개의 collaborator Agent들을 효과적으로 활용할 수 있습니다."
    ],
    "번역하기": [
        "한국어와 영어에 대한 번역을 제공합니다. 한국어로 입력하면 영어로, 영어로 입력하면 한국어로 번역합니다."        
    ],
    "문법 검토하기": [
        "영어와 한국어 문법의 문제점을 설명하고, 수정된 결과를 함께 제공합니다."
    ],
    "이미지 분석": [
        "이미지를 업로드하면 이미지의 내용을 요약할 수 있습니다."
    ],
    "비용 분석": [
        "Cloud 사용에 대한 분석을 수행합니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 일상적인 대화와 각종 툴을 이용해 Agent를 구현할 수 있습니다." 
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/bedrock-agent)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        # label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Flow", "Agent", "번역하기", "문법 검토하기"], index=0
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent", "Agent with Knowlege Base", "Multi Agent Collaboration", "번역하기", "문법 검토하기", "이미지 분석", "비용 분석"], index=0
    )   
    st.info(mode_descriptions[mode][0])

    # logger.info(f"mode: {mode}")

    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Nova Pro', 'Nova Lite', 'Nova Micro', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=3
    )

    uploaded_file = None
    if mode=='이미지 분석':
        st.subheader("🌇 이미지 업로드")
        uploaded_file = st.file_uploader("이미지 요약을 위한 파일을 선택합니다.", type=["png", "jpg", "jpeg"])
    elif mode=='RAG' or mode=="Agent" or mode=="Agent with Knowlege Base":
        st.subheader("📋 문서 업로드")
        # print('fileId: ', chat.fileId)
        uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)

    # code interpreter checkbox
    select_code_interpreter = st.checkbox('Code Interpreter', value=False)
    code_interpreter = 'Enable' if select_code_interpreter else 'Disable'
    #print('code_interpreter: ', code_interpreter)

    if code_interpreter=='Enable' and mode=="Agent":
        st.subheader("📋 분석할 문서 업로드")  
        uploaded_file = st.file_uploader("분석할 파일을 선택하세요.", type=["csv", "xls", "xlsx", "YAML", "json", "doc", "docx", "html", "md", "txt", "pdf"])

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    chat.update(modelName, debugMode, st)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # logger.info(f"clear_button: {clear_button}")

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()
    cost.cost_data = {}
    cost.visualizations = {}

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
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

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

# Preview the uploaded image in the sidebar
file_name = ""
state_of_code_interpreter = False
if uploaded_file is not None and clear_button==False:
    logger.info(f"uploaded_file.name: {uploaded_file.name}")
    logger.info(f"code_interpreter: {code_interpreter}")
    if uploaded_file.name:
        logger.info(f"csv type? {uploaded_file.name.lower().endswith(('.csv'))}")

    if uploaded_file.name and code_interpreter=="Disable" and not mode == '이미지 분석':
        chat.initiate()

        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            logger.info(f"status: {status}")
            st.info(status)

        file_name = uploaded_file.name
        logger.info(f"uploading... file_name: {file_name}")
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"file_url: {file_url}")

        kb.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        # my_bar = st.sidebar.progress(0, text=status)
        
        # for percent_complete in range(100):
        #     time.sleep(0.2)
        #     my_bar.progress(percent_complete + 1, text=status)
        if debugMode=='Enable':
            logger.info(f"status: {status}")
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        logger.info(f"msg: {msg}")

    if uploaded_file and clear_button==False and mode == '이미지 분석':
        st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)

        file_name = uploaded_file.name
        url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"url: {url}")   

    elif uploaded_file.name and code_interpreter == "Enable":
        # guide = "Code Interpreter가 준비되었습니다. 원하는 동작을 입력하세요."
        # st.write(guide)
        # st.session_state.messages.append({"role": "assistant", "content": guide})
        state_of_code_interpreter = True

# print("state_of_code_interpreter: ", state_of_code_interpreter)

if clear_button==False and mode == '비용 분석':
    st.subheader("📈 Cost Analysis")

    if not cost.visualizations:
        cost.get_visualiation()

    if 'service_pie' in cost.visualizations:
        st.plotly_chart(cost.visualizations['service_pie'])
    if 'daily_trend' in cost.visualizations:
        st.plotly_chart(cost.visualizations['daily_trend'])
    if 'region_bar' in cost.visualizations:
        st.plotly_chart(cost.visualizations['region_bar'])

    with st.status("thinking...", expanded=True, state="running") as status:
        if not cost.cost_data:
            st.info("비용 데이터를 가져옵니다.")
            cost_data = cost.get_cost_analysis()
            logger.info(f"cost_data: {cost_data}")
            cost.cost_data = cost_data
        else:
            if not cost.insights:        
                st.info("잠시만 기다리세요. 지난 한달간의 사용량을 분석하고 있습니다...")
                insights = cost.generate_cost_insights()
                logger.info(f"insights: {insights}")
                cost.insights = insights
            
            st.markdown(cost.insights)
            st.session_state.messages.append({"role": "assistant", "content": cost.insights})

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
            logger.info(f"response: {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("running...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_rag_with_knowledge_base(prompt, st)                           
                st.write(response)
                logger.info(f"response: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Flow':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_flow(prompt)        
                st.write(response)
                logger.info(f"response: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
        
        elif mode == 'Agent':
            if not state_of_code_interpreter: 
                sessionState = ""
                with st.status("thinking...", expanded=True, state="running") as status:                
                    response, image_url, reference_docs = chat.run_bedrock_agent(prompt, chat.agent_name, sessionState, st)

                    # if chat.isKorean(response)==False:
                    #     logger.info(f"translate to korean")
                    #     response = chat.translate_text(response)

                    st.write(response)
                    logger.info(f"response: {response}, image_url: {image_url}")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content":  response,
                        "images": image_url if image_url else []
                    })
                    chat.save_chat_history(prompt, response)
                
                show_references(reference_docs) 

            else: # code interpreter
                file_name = uploaded_file.name
                logger.info(f"file_name: {file_name}")
                content_type = utils.get_contents_type(file_name)
                logger.info(f"content_type: {content_type}")

                sessionState = {
                    'files': [
                        {
                            'name': file_name,
                            'source': {
                                'byteContent': {
                                    'data': uploaded_file.getvalue(),
                                    'mediaType': content_type
                                },
                                'sourceType': 'BYTE_CONTENT'
                            },
                            'useCase': 'CODE_INTERPRETER'
                        },
                    ]
                }
                with st.status("thinking...", expanded=True, state="running") as status:
                    # prompt = "첨부 파일의 내용을 분석해주세요."
                    response, image_url, reference_docs = chat.run_bedrock_agent(prompt, chat.agent_name, sessionState, st)
                    st.write(response)
                    logger.info(f"response: {response}")                
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "images": image_url if image_url else []
                    })
                 
        elif mode == 'Agent with Knowlege Base':
            sessionState = ""
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = chat.run_bedrock_agent(prompt, chat.agent_kb_name, "", st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })
                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Multi Agent Collaboration':
            sessionState = ""
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url = chat.run_bedrock_multi_agent_collaboration(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })
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
                if modelName == "Claude 3.5 Haiku":
                    st.error("Claude 3.5 Haiku은 이미지를 지원하지 않습니다. 다른 모델을 선택해주세요.")
                else:
                    with st.status("thinking...", expanded=True, state="running") as status:
                        summary = chat.get_image_summarization(file_name, prompt, st)
                        st.write(summary)

                        st.session_state.messages.append({"role": "assistant", "content": summary})

        elif mode == '비용 분석':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = cost.ask_cost_insights(prompt)
                st.write(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            logger.info(f"response: {response}")

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        


