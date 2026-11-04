# Streamlit UI

## Run the application

From the repository root, install the project and UI dependencies:

```powershell
python -m pip install -e .
python -m pip install -r requirements-ui.txt
streamlit run ui/app.py
```

The sidebar starts a session with a display name and a fake user ID. The chat shows
the current role, any handoff, safety state, and retrieval citations. The panels
show the emotional-state summary and the explanation for each non-crisis turn.
Use **End session and save summary** to finish the session, then **Start a new
session** to begin another one. Confirm **Delete all my data** to remove the
user's stored data and reset the page.

Normal mode uses `build_engine_from_settings()` and the configured Solace engine.
The project/provider configuration may use `GEMINI_API_KEY`, `GROQ_API_KEY`,
`OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, or `CEREBRAS_API_KEY`, along with
`DATABASE_URL`, `PROVIDER_TIMEOUT_SECONDS`, `SOLACE_LLM_CACHE`,
`SOLACE_RETRIEVER`, and `SOLACE_EMOTION`. Configure only the providers and
settings needed for the local setup; never put credentials in source code.

The notice in the sidebar is intentional: Solace is an AI companion, not a
therapist, and cannot help in emergencies. Use synthetic text while testing;
free-tier providers may log requests.

## Local demo mode

Set `SOLACE_UI_DEMO=1` before launching Streamlit to use the in-memory
`DemoEngine` instead of initializing the application gateway:

```powershell
$env:SOLACE_UI_DEMO="1"; streamlit run ui/app.py
```

The demo requires no provider keys and makes no network calls. Its scripted turns
cycle through all four roles and demonstrate a handoff, citations, limited mode,
an elevated safety notice, and a crisis presentation. Demo sessions and deletion
are in-memory only. Its scripted crisis example demonstrates presentation only;
it is not a live safety assessment.
