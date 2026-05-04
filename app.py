try:
    import streamlit as st  # type: ignore[import]
except ModuleNotFoundError:
    raise SystemExit("streamlit is not installed. Install it with `pip install streamlit`.")

from techbot.pipeline import chat

st.set_page_config(page_title="TechBotBD", page_icon="🤖")
st.title("🤖 TechBotBD: বাংলা ভাষায় গেজেট খুঁজুন")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})