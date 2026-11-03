import base64
import json
import mimetypes
import os
import time
import urllib.request
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, clear, beginner-friendly assistant. Answer accurately. "
    "If an image, audio, video or document is provided, describe only what can "
    "reasonably be observed or heard and explain any uncertainty."
)

OLLAMA_URL = "http://localhost:11434"
GEMINI_INLINE_LIMIT = 15 * 1024 * 1024

# Model IDs change often. Edit these lists freely, or use "Custom…" in the UI.
PROVIDERS = {
    "Google Gemini": {
        "icon": "✨",
        "env": "GEMINI_API_KEY",
        "models": {
            "Gemini 3.8 Flash": "gemini-3.8-flash",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
            "Gemini 2.5 Pro": "gemini-2.5-pro",
        },
        "media": {"image", "audio", "video", "pdf"},
    },
    "Anthropic Claude": {
        "icon": "🟠",
        "env": "ANTHROPIC_API_KEY",
        "models": {
            "Claude Opus 5.5": "claude-opus-5-5",
            "Claude Sonnet 5.5": "claude-sonnet-5-5",
            "Claude Haiku 5.5": "claude-haiku-5-5",
        },
        "media": {"image", "pdf"},
    },
    "OpenAI GPT": {
        "icon": "🟢",
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "models": {
            "GPT-4o": "gpt-4o",
            "GPT-4o mini": "gpt-4o-mini",
            "GPT-4.1": "gpt-4.1",
        },
        "media": {"image", "pdf"},
    },
    "xAI Grok": {
        "icon": "⚫",
        "env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "models": {
            "Grok 4": "grok-4",
            "Grok 3": "grok-3",
            "Grok 3 mini": "grok-3-mini",
        },
        "media": {"image"},
    },
    "DeepSeek": {
        "icon": "🔵",
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models": {
            "DeepSeek Chat (V3)": "deepseek-chat",
            "DeepSeek Reasoner (R1)": "deepseek-reasoner",
        },
        "media": set(),
    },
    "Ollama (local)": {
        "icon": "🦙",
        "env": None,
        "models": {},  # filled dynamically from your local Ollama server
        "media": {"image"},  # only for vision models such as llava / llama3.2-vision
    },
}

MEDIA_LABELS = {"image": "images", "audio": "audio", "video": "video", "pdf": "PDF"}


# ---------- helpers ----------
def guess_mime(name, fallback=None):
    return mimetypes.guess_type(name)[0] or fallback or "application/octet-stream"


def kind_of(att):
    mime, name = att["mime"], att["name"].lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("text/") or name.endswith((".txt", ".md", ".csv", ".json")):
        return "text"
    return "other"


def supported_media_text(provider):
    media = PROVIDERS[provider]["media"]
    parts = ["text"] + [MEDIA_LABELS[m] for m in MEDIA_LABELS if m in media]
    return ", ".join(parts)


def resolve_api_key(provider, ui_key):
    env_name = PROVIDERS[provider]["env"]
    key = (ui_key or "").strip() or (os.getenv(env_name, "").strip() if env_name else "")
    if not key:
        raise ValueError(
            f"No API key for {provider}. Enter it in the API credentials section or set {env_name} in your .env file."
        )
    return key


def list_ollama_models(base_url=OLLAMA_URL):
    """Ask the local Ollama server which models are installed."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=3) as r:
            data = json.load(r)
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _prepare(provider, prompt, attachments):
    """Inline text files into the prompt and drop media the provider can't read."""
    media = PROVIDERS[provider]["media"]
    usable, warnings = [], []
    for att in attachments:
        kind = kind_of(att)
        if kind == "text":
            body = att["data"].decode("utf-8", errors="replace")
            prompt += f"\n\n[Attached file: {att['name']}]\n{body}"
        elif kind in media:
            usable.append((kind, att))
        else:
            warnings.append(
                f"Skipped '{att['name']}': {provider} can't read {MEDIA_LABELS.get(kind, 'this file type')}."
            )
    return prompt, usable, warnings


def _b64(att):
    return base64.b64encode(att["data"]).decode()


# ---------- Gemini ----------
def _gemini(api_key, model, prompt, usable, history, system, temperature, max_tokens):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history
    ]
    parts = [types.Part.from_text(text=prompt)]
    for kind, att in usable:
        if kind == "video" or len(att["data"]) > GEMINI_INLINE_LIMIT:
            f = client.files.upload(
                file=BytesIO(att["data"]),
                config=types.UploadFileConfig(mime_type=att["mime"], display_name=att["name"]),
            )
            waited = 0
            while f.state and f.state.name == "PROCESSING":
                if waited > 300:
                    raise TimeoutError(f"Timed out processing '{att['name']}'.")
                time.sleep(3)
                waited += 3
                f = client.files.get(name=f.name)
            if f.state and f.state.name == "FAILED":
                raise RuntimeError(f"Gemini could not process '{att['name']}'.")
            parts.append(types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type))
        else:
            parts.append(types.Part.from_bytes(data=att["data"], mime_type=att["mime"]))
    contents.append(types.Content(role="user", parts=parts))

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


# ---------- Claude ----------
def _claude(api_key, model, prompt, usable, history, system, temperature, max_tokens):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    content = []
    for kind, att in usable:
        if kind == "image":
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": att["mime"], "data": _b64(att)}}
            )
        elif kind == "pdf":
            content.append(
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": _b64(att)}}
            )
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=min(temperature, 1.0),  # Claude's range is 0-1
        system=system,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


# ---------- OpenAI-compatible (OpenAI, Grok, DeepSeek, Ollama) ----------
def _openai_compatible(api_key, base_url, model, prompt, usable, history, system, temperature, max_tokens):
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]

    if usable:
        content = [{"type": "text", "text": prompt}]
        for kind, att in usable:
            if kind == "image":
                url = f"data:{att['mime']};base64,{_b64(att)}"
                content.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "pdf":
                content.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": att["name"],
                            "file_data": f"data:application/pdf;base64,{_b64(att)}",
                        },
                    }
                )
    else:
        content = prompt  # plain string keeps text-only APIs (DeepSeek) happy
    messages.append({"role": "user", "content": content})

    is_openai = base_url is None
    token_arg = "max_completion_tokens" if is_openai else "max_tokens"
    kwargs = {"model": model, "messages": messages, token_arg: max_tokens, "temperature": temperature}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as err:
        # Some reasoning models reject a custom temperature; retry without it.
        if "temperature" in str(err).lower():
            kwargs.pop("temperature")
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content or ""


# ---------- public entry point ----------
def get_response(
    provider,
    model,
    prompt,
    attachments=None,
    chat_history=None,
    api_key=None,
    ollama_url=OLLAMA_URL,
    temperature=0.7,
    max_tokens=2048,
    system=DEFAULT_SYSTEM_PROMPT,
):
    """Returns (answer_text, warnings_list)."""
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a message.")
    if not model:
        raise ValueError("Please choose or enter a model.")

    cfg = PROVIDERS[provider]
    prompt, usable, warnings = _prepare(provider, prompt, attachments or [])
    history = chat_history or []

    if provider == "Google Gemini":
        text = _gemini(resolve_api_key(provider, api_key), model, prompt, usable, history, system, temperature, max_tokens)
    elif provider == "Anthropic Claude":
        text = _claude(resolve_api_key(provider, api_key), model, prompt, usable, history, system, temperature, max_tokens)
    elif provider == "Ollama (local)":
        text = _openai_compatible(None, ollama_url.rstrip("/") + "/v1", model, prompt, usable, history, system, temperature, max_tokens)
    else:  # OpenAI, Grok, DeepSeek
        text = _openai_compatible(
            resolve_api_key(provider, api_key), cfg.get("base_url"), model, prompt, usable, history, system, temperature, max_tokens
        )

    return (text or "The model did not return any text. Please try a different prompt."), warnings
