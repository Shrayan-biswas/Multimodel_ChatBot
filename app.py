import streamlit as st

from llm_providers import (
    DEFAULT_SYSTEM_PROMPT,
    OLLAMA_URL,
    PROVIDERS,
    get_response,
    guess_mime,
    list_ollama_models,
    supported_media_text,
)

CREATOR = "Shrayan Biswas"

st.set_page_config(
    page_title="AI Studio | Multimodal Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- White + cornsilk visual system ----------
st.markdown(
    """
    <style>
    :root {
        --ink: #29251f;
        --muted: #746d61;
        --paper: #fffefa;
        --cornsilk: #fff8dc;
        --cornsilk-deep: #f4e8bd;
        --line: #eadfbe;
        --accent: #9a7430;
        --accent-dark: #74551f;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(255,254,250,.92); }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, span, div { color: var(--ink); }
    .hero {
        padding: 1.7rem 2rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: linear-gradient(120deg, #fff8dc 0%, #fffdf3 68%, #ffffff 100%);
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 28px rgba(100, 77, 25, .06);
    }
    .hero h1 {
        margin: 0;
        color: #352b1b;
        font-size: clamp(2rem, 4vw, 2.8rem);
        letter-spacing: -1.4px;
    }
    .hero p {
        margin: .45rem 0 0;
        color: #76684d;
        font-size: 1rem;
    }
    .eyebrow {
        color: #92733b;
        font-size: .75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .14em;
        margin-bottom: .45rem;
    }
    .section-card {
        background: #fffdf5;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: .5rem 0 1rem;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: .35rem .75rem;
        margin: .7rem .35rem 0 0;
        background: #f5edcf;
        color: #725923;
        border: 1px solid #e8d9a8;
        font-size: .78rem;
    }
    [data-testid="stChatMessage"] {
        background: #fffdf6;
        border: 1px solid #eee5cd;
        border-radius: 16px;
        padding: .8rem 1rem;
        margin-bottom: .7rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #fff8df;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #ddc98f;
        border-radius: 16px;
        background: #fffefa;
    }
    [data-testid="stChatInput"] textarea {
        color: #29251f !important;
    }
    .stButton > button {
        border-radius: 11px;
        border: 1px solid #d9c58e;
        background: #f8edc9;
        color: #59451d;
        font-weight: 600;
    }
    .stButton > button:hover {
        border-color: #a98743;
        background: #f2e2ae;
        color: #3e3016;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextArea textarea {
        background: #fffefa !important;
        border-color: #e3d7b8 !important;
        border-radius: 10px !important;
    }
    /* File uploader: Streamlit uses a separate BaseWeb dropzone surface. */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > section,
    [data-testid="stFileUploaderDropzone"] {
        background: #fffdf5 !important;
        color: #29251f !important;
        border: 1px dashed #d8c58f !important;
        border-radius: 14px !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #51452e !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #f8edc9 !important;
        color: #59451d !important;
        border: 1px solid #d9c58e !important;
    }

    /* Chat composer: force the fixed bottom input and its form to the light theme. */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] textarea {
        background: #fffefa !important;
        color: #29251f !important;
        border: 1px solid #ddc98f !important;
        border-radius: 14px !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #877b65 !important;
        -webkit-text-fill-color: #877b65 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button {
        background: #f2e2ae !important;
        color: #59451d !important;
        border: 1px solid #d9c58e !important;
    }

    /* Keep Streamlit's bottom composer area from inheriting a dark surface. */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
        background: #fffefa !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #fffefa !important;
    }
    input, textarea, [contenteditable="true"] {
        background-color: #fffefa !important;
        color: #29251f !important;
        caret-color: #74551f !important;
    }
    button, [role="button"] {
        color: #59451d;
    }
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [role="option"] {
        background: #fffefa !important;
        color: #29251f !important;
    }
    .footer {
        text-align: center;
        color: #978b72;
        font-size: .8rem;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = "Google Gemini"

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">MULTIMODAL AI WORKSPACE</div>
        <h1>✦ AI Studio</h1>
        <p>One workspace for text, images, documents, audio and video — powered by the model you choose.</p>
        <span class="status-pill">Multiple providers</span>
        <span class="status-pill">Multimodal chat</span>
        <span class="status-pill">Local Ollama support</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Main-screen model and provider controls ----------
st.markdown("### Choose your model")
provider_col, model_col, action_col = st.columns([1.05, 1.45, 0.8], vertical_alignment="bottom")

with provider_col:
    provider = st.selectbox(
        "AI provider",
        list(PROVIDERS.keys()),
        format_func=lambda p: f"{PROVIDERS[p]['icon']}  {p}",
        key="selected_provider",
    )

cfg = PROVIDERS[provider]
api_key = None
ollama_url = OLLAMA_URL

with model_col:
    if provider == "Ollama (local)":
        ollama_url = st.text_input("Ollama server URL", value=OLLAMA_URL)
        installed_models = list_ollama_models(ollama_url)
        if installed_models:
            model_choice = st.selectbox("Model", installed_models)
            model_id = model_choice
        else:
            st.info("No local models detected. Start Ollama and pull a model, or enter its name.")
            model_id = st.text_input("Model name", placeholder="e.g. llama3.2")
    else:
        model_labels = list(cfg["models"].keys()) + ["Custom model ID…"]
        model_choice = st.selectbox("Model", model_labels)
        if model_choice == "Custom model ID…":
            model_id = st.text_input("Enter model ID", placeholder="Provider model identifier")
        else:
            model_id = cfg["models"][model_choice]

with action_col:
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

st.caption(f"**Supported by {provider}:** {supported_media_text(provider)}")

# All settings remain in the main page; no sidebar is used.
with st.expander("🔐 API credentials", expanded=False):
    if provider == "Ollama (local)":
        st.info("Ollama runs locally and does not need a cloud API key.")
    else:
        api_key = st.text_input(
            f"{provider} API key",
            type="password",
            key=f"key_{provider}",
            placeholder="Paste API key (session only)",
            help=f"If left blank, {cfg['env']} is read from your local .env file.",
        )
        st.caption("Keys entered here stay in the current Streamlit session. Never commit API keys.")

with st.expander("⚙️ Generation settings", expanded=False):
    setting_col1, setting_col2 = st.columns(2)
    with setting_col1:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    with setting_col2:
        max_tokens = st.slider("Maximum output tokens", 256, 8192, 2048, 256)
    system_prompt = st.text_area(
        "System instruction",
        DEFAULT_SYSTEM_PROMPT,
        height=110,
    )

st.divider()

# ---------- Attachment handling ----------
st.markdown("### Your conversation")
attachments = []
key = st.session_state.uploader_key

with st.expander("📎 Attach files or record audio", expanded=not st.session_state.messages):
    tab_files, tab_voice = st.tabs(["Upload files", "Record voice"])
    with tab_files:
        uploaded_files = st.file_uploader(
            "Add images, audio, video or documents",
            type=[
                "png", "jpg", "jpeg", "webp", "gif",
                "mp3", "wav", "m4a", "aac", "ogg", "flac",
                "mp4", "mov", "avi", "webm", "mkv",
                "pdf", "txt", "md", "csv", "json",
            ],
            accept_multiple_files=True,
            key=f"files_{key}",
        )
        for uploaded in uploaded_files or []:
            attachments.append(
                {
                    "name": uploaded.name,
                    "mime": guess_mime(uploaded.name, uploaded.type),
                    "data": uploaded.getvalue(),
                }
            )
    with tab_voice:
        if hasattr(st, "audio_input"):
            recording = st.audio_input("Record a voice message", key=f"voice_{key}")
            if recording is not None:
                attachments.append(
                    {
                        "name": "voice_message.wav",
                        "mime": "audio/wav",
                        "data": recording.getvalue(),
                    }
                )
        else:
            st.info("Voice recording requires a recent Streamlit version: pip install --upgrade streamlit")

if attachments:
    st.success(f"{len(attachments)} attachment(s) ready. Compatibility depends on the selected provider/model.")

def render_attachments(items):
    for item in items:
        mime = item["mime"]
        if mime.startswith("image/"):
            st.image(item["data"], caption=item["name"], width=320)
        elif mime.startswith("audio/"):
            st.audio(item["data"], format=mime)
        elif mime.startswith("video/"):
            st.video(item["data"], format=mime)
        else:
            st.caption(f"📎 {item['name']}")

# ---------- Existing conversation ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_attachments(message.get("attachments", []))
        st.markdown(message["content"])
        if message.get("by"):
            st.caption(message["by"])
        for warning in message.get("warnings", []):
            st.warning(warning)

# ---------- Chat input ----------
prompt = st.chat_input("Message your selected AI model…")

if prompt:
    if not model_id:
        st.error("Choose or enter a model before sending a message.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "attachments": attachments}
        )
        with st.chat_message("user"):
            render_attachments(attachments)
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"{provider} is thinking…"):
                try:
                    history = [
                        {"role": item["role"], "content": item["content"]}
                        for item in st.session_state.messages[:-1]
                    ]
                    answer, warnings = get_response(
                        provider,
                        model_id,
                        prompt,
                        attachments=attachments,
                        chat_history=history,
                        api_key=api_key,
                        ollama_url=ollama_url,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system=system_prompt,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "attachments": [],
                            "by": f"{cfg['icon']} {provider} · {model_id}",
                            "warnings": warnings,
                        }
                    )
                    st.session_state.uploader_key += 1
                    st.rerun()
                except Exception as error:
                    st.session_state.messages.pop()
                    st.error(f"Something went wrong: {error}")

st.markdown(
    f'<div class="footer">AI Studio · Designed in white & cornsilk · Built by {CREATOR}</div>',
    unsafe_allow_html=True,
)
