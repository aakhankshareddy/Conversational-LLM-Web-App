import streamlit as st
import test_inference
import io
from datetime import datetime
import time

# --- UI CONFIG ---
# Theme switcher
if "theme" not in st.session_state:
    st.session_state.theme = "light"

st.set_page_config(page_title="Mistral Chatbot", layout="centered")

# --- THEME SWITCHER ---
theme = st.sidebar.radio("Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = theme

# Inject theme CSS
if theme == "dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #181818 !important; color: #f1f1f1 !important; }
        .stTextInput>div>div>input { background: #222 !important; color: #f1f1f1 !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #fff !important; color: #222 !important; }
        .stTextInput>div>div>input { background: #fff !important; color: #222 !important; }
        </style>
    """, unsafe_allow_html=True)

# --- MODEL OPTIONS ---
MODEL_OPTIONS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/DialoGPT-medium",
    "gpt2"
]

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of {"role": ..., "content": ..., "timestamp": ...}
if "model" not in st.session_state:
    st.session_state.model = MODEL_OPTIONS[0]
if "waiting_for_bot" not in st.session_state:
    st.session_state.waiting_for_bot = False
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "bot_responded" not in st.session_state:
    st.session_state.bot_responded = False

# --- SIDEBAR ---
st.sidebar.title("Settings")
st.sidebar.markdown("**Model Selection**")
model_choice = st.sidebar.selectbox("Choose a model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.model))
st.session_state.model = model_choice

# Debug info
st.sidebar.markdown("---")
st.sidebar.markdown("**Debug Info**")
st.sidebar.markdown(f"Messages: {len(st.session_state.messages)}")
st.sidebar.markdown(f"Waiting: {st.session_state.waiting_for_bot}")
st.sidebar.markdown(f"Responded: {st.session_state.bot_responded}")
st.sidebar.markdown(f"Last input: {st.session_state.last_user_input[:20]}...")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.last_user_input = ""
    st.session_state.bot_responded = False
    st.session_state.waiting_for_bot = False
    st.rerun()

# --- EXPORT BUTTON ---
def export_chat(messages):
    lines = []
    for msg in messages:
        prefix = "🧑" if msg["role"] == "user" else "🤖"
        ts = msg.get("timestamp", "")
        lines.append(f"{prefix} [{ts}] {msg['content']}")
    return "\n".join(lines)

exported = export_chat(st.session_state.messages)
st.sidebar.download_button(
    label="Export Chat (.txt)",
    data=exported,
    file_name="chat_history.txt",
    mime="text/plain"
)

# --- MAIN CHAT UI ---
st.title("🤖 Mistral Chatbot")
st.markdown("""
A conversational AI powered by Mistral-7B-Instruct (Hugging Face Inference API).\
**Note:** For best results, add your Hugging Face token in `test_inference.py` if you have one.
""")

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    ts = msg.get("timestamp", "")
    if msg["role"] == "user":
        st.markdown(f"🧑 **You** [{ts}]: {msg['content']}")
    else:
        st.markdown(f"🤖 **Assistant** [{ts}]: {msg['content']}")

# --- USER INPUT ---
def input_box():
    # Use a unique key that changes with each message to prevent input reuse
    key = f"user_input_{len(st.session_state.messages)}"
    return st.text_input("Your message:", key=key, placeholder="Type your message and press Enter...", label_visibility="collapsed")

user_input = input_box()

# --- HANDLE USER MESSAGE ---
def handle_user_message():
    if user_input.strip() == "":
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": now})
    st.session_state.last_user_input = user_input
    st.session_state.bot_responded = False
    st.rerun()

# --- BOT RESPONSE GENERATION ---
# Only generate response if we have a new user message and haven't responded yet
if (st.session_state.last_user_input and 
    not st.session_state.bot_responded and 
    not st.session_state.waiting_for_bot):
    
    st.session_state.waiting_for_bot = True
    
    # Show animated typing indicator
    with st.spinner("🤖 Assistant is typing..."):
        # Simulate animated dots
        typing_placeholder = st.empty()
        for i in range(10):
            typing_placeholder.markdown(f"<span style='font-size:1.2em;'>🤖 Assistant is typing{'.' * (i % 4)}</span>", unsafe_allow_html=True)
            time.sleep(0.25)
        # Call the model (token is set in test_inference.py)
        reply = test_inference.get_mistral_response(
            st.session_state.messages,
            model=st.session_state.model,
            api_token=test_inference.HF_API_TOKEN  # <-- Add your token in test_inference.py
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "assistant", "content": reply, "timestamp": now})
        st.session_state.waiting_for_bot = False
        st.session_state.bot_responded = True
        st.rerun()

# --- SUBMIT USER MESSAGE ---
# Only handle new user input if we're not waiting for bot and the last message wasn't from user
if user_input and not st.session_state.waiting_for_bot:
    if user_input.strip():
        # Check if this is a new message (not already in messages)
        if not st.session_state.messages or st.session_state.messages[-1]["role"] == "assistant":
            handle_user_message()
        elif (st.session_state.messages and 
              st.session_state.messages[-1]["role"] == "user" and 
              st.session_state.messages[-1]["content"] != user_input):
            # This is a new user message, handle it
            handle_user_message()

# --- BOTTOM SPACER ---
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# --- COMMENT FOR TOKEN ---
# To use a Hugging Face token, open test_inference.py and set HF_API_TOKEN = "YOUR_TOKEN_HERE".
