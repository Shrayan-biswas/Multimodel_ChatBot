# AI Studio — Multimodal AI Chatbot

A Streamlit-based multimodal chat application that brings several hosted AI providers and locally hosted Ollama models into one clean interface. Choose a provider and model, attach supported files, adjust generation settings and chat from a single screen.

## Features

- Multiple providers: Google Gemini, Anthropic Claude, OpenAI, xAI Grok, DeepSeek and local Ollama.
- Main-screen provider and model selection — no sidebar.
- White and cornsilk UI theme.
- Custom model ID entry for providers that support it.
- Image, audio, video and PDF attachments where supported by the selected provider/model.
- TXT, Markdown, CSV and JSON text-file context.
- Optional voice recording on compatible Streamlit versions.
- Conversation history for the active session.
- Temperature, output-token and system-instruction controls.
- Local model support through Ollama.

## Project structure

```text
gemini_multimodal_chatbot/
├── app.py
├── llm_providers.py
├── gemini_bot.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

## Requirements

- Python 3.10 or newer (Python 3.11–3.13 recommended).
- Internet access for hosted providers.
- An API key for the hosted provider you want to use.
- Ollama installed and running if you want local inference.

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

If activation is blocked, call the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## API key configuration

Enter the API key in the app's **API credentials** section, or place it in `.env`:

```dotenv
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
XAI_API_KEY=
DEEPSEEK_API_KEY=
```

Get keys from the official provider dashboards:

- Google AI Studio: https://aistudio.google.com/
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/
- xAI: https://console.x.ai/
- DeepSeek: https://platform.deepseek.com/

Only configure the provider you plan to use. Never commit `.env` or expose API keys in screenshots, logs or source code.

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Use local models with Ollama

1. Install Ollama: https://ollama.com/
2. Pull a model:

   ```bash
   ollama pull llama3.2
   ```

3. Make sure Ollama is running. If needed:

   ```bash
   ollama serve
   ```

4. Select **Ollama (local)** in the provider selector on the main page.
5. Select an installed model from the model field.

For image prompts, use a vision-capable model compatible with your Ollama setup. The default Ollama endpoint is `http://localhost:11434`.

## Attachments and model compatibility

The app accepts common image, audio, video and document formats. Actual support depends on the provider API and selected model. Text files (TXT, MD, CSV and JSON) are decoded and added to the prompt. Unsupported media may be skipped with a warning.

Provider capabilities, API formats, model IDs, file-size limits and pricing change. Verify current provider documentation before relying on a particular model or media type. Hosted API calls may incur charges.

## Troubleshooting

- **Authentication error:** Check the API key and confirm that the selected model is enabled for your account.
- **Model not found:** Confirm the exact model ID in the provider's current documentation.
- **Ollama connection error:** Start Ollama and check that `http://localhost:11434` is reachable.
- **No local models listed:** Run `ollama pull <model-name>` and refresh the app.
- **Unsupported attachment:** Switch to a provider/model that supports that media type.
- **Dependency errors:** Recreate `.venv` with a supported Python version and reinstall `requirements.txt`.

## Security and privacy

- `.env` and local Streamlit secrets should never be committed.
- API keys entered in the app are held in the active Streamlit session.
- Review each provider's data retention and privacy policy before sending sensitive material.
- Do not deploy this app publicly without authentication, rate limiting, upload-size controls and secure secret management.

## Author

**Shrayan Biswas**

## License

See [LICENSE](LICENSE). If no license file is included in your repository, add one before presenting the project as open source.
