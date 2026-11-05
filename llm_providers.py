import streamlit as st
from ai_backend import (
    PROVIDERS,
    get_response,
    list_provider_models,
)

provider = st.selectbox(
    "Select AI Provider",
    list(PROVIDERS.keys())
)

api_key = st.text_input(
    "API Key",
    type="password"
)

if provider == "NVIDIA Nemotron":
    st.markdown(
        PROVIDERS[provider]["svg"],
        unsafe_allow_html=True
    )

models = list_provider_models(provider, api_key)

model = st.selectbox(
    "Select Model",
    list(models.keys())
)

prompt = st.text_area("Enter your prompt")

if st.button("Generate"):
    answer, warnings = get_response(
        provider=provider,
        model=models[model],
        prompt=prompt,
        api_key=api_key,
    )

    st.write(answer)

    for warning in warnings:
        st.warning(warning)
