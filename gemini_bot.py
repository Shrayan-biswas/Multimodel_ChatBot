import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ENV_FILE = Path(r"C:\Users\Shrayan.Abhijit\Desktop\.env")
load_dotenv(dotenv_path=ENV_FILE)


def get_api_key():
    """Read and validate the Gemini API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            f"GEMINI_API_KEY was not found in {ENV_FILE}. Add an entry like "
            "GEMINI_API_KEY=your_api_key."
        )
    return api_key


def create_gemini_client():
    """Create and return the Google Gen AI SDK client."""
    return genai.Client(api_key=get_api_key())


def build_conversation(prompt, chat_history):
    """Convert simple chat history into Gemini Content objects."""
    conversation = []
    for message in chat_history:
        role = "user" if message["role"] == "user" else "model"
        conversation.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message["content"])],
            )
        )

    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    )
    return conversation


def get_gemini_response(prompt, image=None, chat_history=None):
    """Send text, or text plus an image, to Gemini."""
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a message.")

    if chat_history is None:
        chat_history = []

    client = create_gemini_client()
    contents = build_conversation(prompt, chat_history)

    if image is not None:
        image_buffer = BytesIO()
        image.convert("RGB").save(image_buffer, format="JPEG", quality=90)
        image_part = types.Part.from_bytes(
            data=image_buffer.getvalue(),
            mime_type="image/jpeg",
        )
        contents[-1].parts.append(image_part)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful, clear, beginner-friendly assistant. "
                "Answer accurately. If an image is provided, describe only "
                "what can reasonably be observed and explain uncertainty."
            ),
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    if not response.text:
        return "Gemini did not return text. Please try a different prompt."
    return response.text


def test_text_chat():
    """Small command-line test for the text chatbot."""
    question = input("Ask Gemini: ")
    print("\nGemini:", get_gemini_response(question))


if __name__ == "__main__":
    test_text_chat()
