
import base64
import json
import mimetypes
import os
import time
import urllib.request
import urllib.error
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, clear, beginner-friendly assistant. "
    "Answer accurately. If an image, audio, video or document "
    "is provided, describe only what can reasonably be observed "
    "or heard and explain any uncertainty."
)

OLLAMA_URL = "http://localhost:11434"
GEMINI_INLINE_LIMIT = 15 * 1024 * 1024


# ============================================================
# NVIDIA SVG LOGO
# ============================================================

NVIDIA_SVG = """
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 120 120">
    <rect width="120" height="120" rx="18" fill="#76B900"/>
    <path d="M15 59 Q46 20 85 44 Q108 61 83 83
             Q58 104 30 75 Q50 87 69 70
             Q84 57 67 44 Q45 29 15 59Z"
          fill="white"/>
    <circle cx="64" cy="59" r="13" fill="#76B900"/>
    <circle cx="64" cy="59" r="6" fill="white"/>
</svg>
"""


# ============================================================
# AI PROVIDERS
# ============================================================

PROVIDERS = {

    "Google Gemini": {
        "icon": "✨",
        "env": "GEMINI_API_KEY",
        "models": {
            "Gemini 2.5 Pro": "gemini-2.5-pro",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
        },
        "media": {"image", "audio", "video", "pdf"},
        "discovery": "gemini",
    },

    "Anthropic Claude": {
        "icon": "🟠",
        "env": "ANTHROPIC_API_KEY",
        "models": {
            "Claude Opus": "claude-opus-4-1",
            "Claude Sonnet": "claude-sonnet-4-5",
            "Claude Haiku": "claude-haiku-4-5",
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
        "models_url": "https://api.x.ai/v1/language-models",
        "models": {},
        "media": {"image"},
        "discovery": "openai",
    },

    "DeepSeek": {
        "icon": "🔵",
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models_url": "https://api.deepseek.com/models",
        "models": {},
        "media": set(),
        "discovery": "openai",
    },

    "NVIDIA Nemotron": {
        "icon": "🟩",
        "svg": NVIDIA_SVG,
        "env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "models": {
            "Nemotron 3.5 Super VL 120B A12B":
                "nvidia/nemotron-3.5-super-vl-120b-a12b",
        },
        "media": {"image"},
    },

    "Ollama (local)": {
        "icon": "🦙",
        "env": None,
        "models": {},
        "media": {"image"},
    },
}


MEDIA_LABELS = {
    "image": "images",
    "audio": "audio",
    "video": "video",
    "pdf": "PDF",
}


# ============================================================
# API KEY MANAGEMENT
# ============================================================

def resolve_api_key(provider, ui_key=None):

    env_name = PROVIDERS[provider]["env"]

    key = (ui_key or "").strip()

    if not key and env_name:
        key = os.getenv(env_name, "").strip()

    if not key and env_name:
        raise ValueError(
            f"No API key configured for {provider}. "
            f"Enter it in the UI or set {env_name} in .env."
        )

    return key


# ============================================================
# MODEL DISCOVERY
# ============================================================

def list_ollama_models(base_url=OLLAMA_URL):
    """Discover installed local Ollama models."""

    try:
        url = base_url.rstrip("/") + "/api/tags"

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.load(response)

        return {
            item["name"]: item["name"]
            for item in data.get("models", [])
        }

    except Exception:
        return {}


def list_gemini_models(api_key):
    """Discover Gemini models available to the API key."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        models = {}

        for item in client.models.list():

            name = getattr(item, "name", None)

            if not name:
                continue

            model_id = name.replace("models/", "")

            actions = getattr(item, "supported_actions", None)

            if actions and not any(
                "generatecontent" in str(action).lower()
                for action in actions
            ):
                continue

            models[model_id] = model_id

        return models

    except Exception:
        return {}


def list_remote_models(provider, api_key):
    """
    Discover models from providers exposing OpenAI-style
    model listing endpoints.
    """

    cfg = PROVIDERS[provider]
    models_url = cfg.get("models_url")

    if not models_url or not api_key:
        return {}

    try:

        request = urllib.request.Request(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)

        items = data.get("data", data.get("models", []))

        models = {}

        for item in items:

            model_id = item.get("id")

            if not model_id:
                continue

            # Keep only conversational language models where
            # the provider supplies modality metadata.
            modalities = item.get("input_modalities")

            if modalities and "text" not in [
                str(value).lower() for value in modalities
            ]:
                continue

            models[model_id] = model_id

        return models

    except Exception:
        return {}


def list_provider_models(provider, api_key=None,
                         ollama_url=OLLAMA_URL):
    """
    Return model IDs available for the selected provider.
    Dynamically discovers Gemini, Grok, DeepSeek and Ollama.
    """

    cfg = PROVIDERS[provider]

    if provider == "Ollama (local)":
        return list_ollama_models(ollama_url)

    if provider == "Google Gemini":

        key = resolve_api_key(provider, api_key)
        models = list_gemini_models(key)

        return models or cfg["models"]

    if cfg.get("discovery") == "openai":

        key = resolve_api_key(provider, api_key)
        models = list_remote_models(provider, key)

        return models or cfg["models"]

    return cfg["models"].copy()


# ============================================================
# FILE HELPERS
# ============================================================

def guess_mime(name, fallback=None):

    return (
        mimetypes.guess_type(name)[0]
        or fallback
        or "application/octet-stream"
    )


def kind_of(att):

    mime = att["mime"]
    name = att["name"].lower()

    if mime.startswith("image/"):
        return "image"

    if mime.startswith("audio/"):
        return "audio"

    if mime.startswith("video/"):
        return "video"

    if mime == "application/pdf":
        return "pdf"

    if mime.startswith("text/") or name.endswith(
        (".txt", ".md", ".csv", ".json", ".py", ".log")
    ):
        return "text"

    return "other"


def supported_media_text(provider):

    media = PROVIDERS[provider]["media"]

    parts = ["text"]

    parts.extend(
        MEDIA_LABELS[item]
        for item in MEDIA_LABELS
        if item in media
    )

    return ", ".join(parts)


def _prepare(provider, prompt, attachments):

    media = PROVIDERS[provider]["media"]

    usable = []
    warnings = []

    for att in attachments:

        kind = kind_of(att)

        if kind == "text":

            body = att["data"].decode(
                "utf-8",
                errors="replace"
            )

            prompt += (
                f"\n\n[Attached file: {att['name']}]\n"
                f"{body}"
            )

        elif kind in media:

            usable.append((kind, att))

        else:

            warnings.append(
                f"Skipped '{att['name']}': "
                f"{provider} cannot read "
                f"{MEDIA_LABELS.get(kind, 'this file type')}."
            )

    return prompt, usable, warnings


def _b64(att):

    return base64.b64encode(
        att["data"]
    ).decode("utf-8")


# ============================================================
# GEMINI API
# ============================================================

def _gemini(
    api_key,
    model,
    prompt,
    usable,
    history,
    system,
    temperature,
    max_tokens
):

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents = []

    for message in history:

        contents.append(
            types.Content(
                role=(
                    "user"
                    if message["role"] == "user"
                    else "model"
                ),
                parts=[
                    types.Part.from_text(
                        text=message["content"]
                    )
                ],
            )
        )

    parts = [
        types.Part.from_text(text=prompt)
    ]

    for kind, att in usable:

        if (
            kind == "video"
            or len(att["data"]) > GEMINI_INLINE_LIMIT
        ):

            file_obj = client.files.upload(
                file=BytesIO(att["data"]),
                config=types.UploadFileConfig(
                    mime_type=att["mime"],
                    display_name=att["name"],
                ),
            )

            waited = 0

            while (
                file_obj.state
                and file_obj.state.name == "PROCESSING"
            ):

                if waited >= 300:
                    raise TimeoutError(
                        f"Timed out processing {att['name']}."
                    )

                time.sleep(3)
                waited += 3

                file_obj = client.files.get(
                    name=file_obj.name
                )

            if (
                file_obj.state
                and file_obj.state.name == "FAILED"
            ):
                raise RuntimeError(
                    f"Gemini could not process {att['name']}."
                )

            parts.append(
                types.Part.from_uri(
                    file_uri=file_obj.uri,
                    mime_type=file_obj.mime_type,
                )
            )

        else:

            parts.append(
                types.Part.from_bytes(
                    data=att["data"],
                    mime_type=att["mime"],
                )
            )

    contents.append(
        types.Content(
            role="user",
            parts=parts,
        )
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    return response.text or ""


# ============================================================
# CLAUDE API
# ============================================================

def _claude(
    api_key,
    model,
    prompt,
    usable,
    history,
    system,
    temperature,
    max_tokens
):

    import anthropic

    client = anthropic.Anthropic(
        api_key=api_key
    )

    messages = [
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history
    ]

    content = []

    for kind, att in usable:

        if kind == "image":

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": att["mime"],
                    "data": _b64(att),
                },
            })

        elif kind == "pdf":

            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": _b64(att),
                },
            })

    content.append({
        "type": "text",
        "text": prompt,
    })

    messages.append({
        "role": "user",
        "content": content,
    })

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=min(temperature, 1.0),
        system=system,
        messages=messages,
    )

    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", "") == "text"
    )


# ============================================================
# OPENAI-COMPATIBLE API
# OpenAI / Grok / DeepSeek / NVIDIA / Ollama
# ============================================================

def _openai_compatible(
    api_key,
    base_url,
    model,
    prompt,
    usable,
    history,
    system,
    temperature,
    max_tokens
):

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or "not-needed",
        base_url=base_url,
    )

    messages = [
        {
            "role": "system",
            "content": system,
        }
    ]

    messages.extend(
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history
    )

    if usable:

        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for kind, att in usable:

            if kind == "image":

                image_url = (
                    f"data:{att['mime']};base64,"
                    f"{_b64(att)}"
                )

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                })

            elif kind == "pdf":

                content.append({
                    "type": "file",
                    "file": {
                        "filename": att["name"],
                        "file_data": (
                            "data:application/pdf;base64,"
                            + _b64(att)
                        ),
                    },
                })

    else:

        content = prompt

    messages.append({
        "role": "user",
        "content": content,
    })

    is_openai = base_url is None

    token_arg = (
        "max_completion_tokens"
        if is_openai
        else "max_tokens"
    )

    kwargs = {
        "model": model,
        "messages": messages,
        token_arg: max_tokens,
        "temperature": temperature,
    }

    try:

        response = client.chat.completions.create(
            **kwargs
        )

    except Exception as error:

        # Some reasoning models reject custom temperature.

        if "temperature" in str(error).lower():

            kwargs.pop("temperature")

            response = client.chat.completions.create(
                **kwargs
            )

        else:
            raise

    return (
        response.choices[0].message.content
        or ""
    )


# ============================================================
# MAIN RESPONSE FUNCTION
# ============================================================

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
    """
    Main entry point.

    Returns:
        (answer_text, warnings_list)
    """

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}"
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Please enter a message."
        )

    if not model:
        raise ValueError(
            "Please choose or enter a model."
        )

    cfg = PROVIDERS[provider]

    prompt, usable, warnings = _prepare(
        provider,
        prompt,
        attachments or [],
    )

    history = chat_history or []

    if provider == "Google Gemini":

        text = _gemini(
            resolve_api_key(provider, api_key),
            model,
            prompt,
            usable,
            history,
            system,
            temperature,
            max_tokens,
        )

    elif provider == "Anthropic Claude":

        text = _claude(
            resolve_api_key(provider, api_key),
            model,
            prompt,
            usable,
            history,
            system,
            temperature,
            max_tokens,
        )

    elif provider == "Ollama (local)":

        text = _openai_compatible(
            None,
            ollama_url.rstrip("/") + "/v1",
            model,
            prompt,
            usable,
            history,
            system,
            temperature,
            max_tokens,
        )

    else:

        text = _openai_compatible(
            resolve_api_key(provider, api_key),
            cfg.get("base_url"),
            model,
            prompt,
            usable,
            history,
            system,
            temperature,
            max_tokens,
        )

    return (
        text or (
            "The model did not return any text. "
            "Please try a different prompt."
        ),
        warnings,
    )
