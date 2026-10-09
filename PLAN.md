# Implementation Plan: Magic School Interview Agent

Based on [SPECT.md](SPECT.md).

The core is one engine with no saved state between calls. It takes a conversation record and the student's message, and returns the reply plus the updated record. Locally it runs from a text command line, and behind a small HTTP API that a React frontend uses for voice. On AWS the same API and frontend are deployed. Nothing in the engine knows where it's running.

## 1. Project layout

```
orbio_agent/
├── pyproject.toml          # Python 3.12, plain venv + pip (uv-compatible)
├── Makefile                # install, test, lint, typecheck, run-text, serve, frontend, docker
├── Dockerfile / docker-compose.yml
├── .env.example            # ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, STORAGE=file|s3
├── data/catalog.json       # 15 clubs/classes
├── src/agent/
│   ├── config.py           # pydantic-settings, reads env and .env
│   ├── bootstrap.py        # builds engine, STT and TTS from settings (shared by CLI and API)
│   ├── domain/
│   │   ├── models.py       # StudentProfile, ExtractedFields, TurnOutput, ConversationRecord
│   │   └── catalog.py      # load, hard-filter, deterministic fallback ranking
│   ├── engine/
│   │   ├── engine.py       # Engine.start() / Engine.handle_turn(conversation_id, text)
│   │   ├── phases.py       # required vs skippable fields, readiness, frustration
│   │   └── prompts.py      # interview prompt, one-off prompts, fixed fallback messages
│   ├── llm/      base.py (LLMClient ABC + errors) · claude.py · fake.py
│   ├── speech/   base.py (STT, TTS ABCs) · deepgram.py · elevenlabs.py (TTS + STT) · fake.py · pipeline.py
│   ├── storage/  base.py (ConversationRepository ABC) · file_repo.py · s3_repo.py
│   └── interfaces/
│       ├── cli.py          # text only (also what the tests drive)
│       └── api.py          # FastAPI: used by the frontend, deployed to AWS
├── frontend/               # Vite + React + TypeScript: push-to-talk voice UI
└── tests/
```

## 2. Core design

### Data models (`domain/models.py`)

- **`StudentProfile`**: all fields optional, with enums and ranges enforced by Pydantic (`school_year` 1–7, `weekly_free_hours` 0–168). Choice fields accept any capitalization. `apply(updates)` validates each new value on its own and returns the merged profile plus an error per rejected field, so one bad value never discards the good ones.
- **`ExtractedFields`**: the values Claude extracts, deliberately loosely typed (plain strings, not enums). If a student says "necromancy", Claude reports it as said and `apply()` rejects it with a message; with enums in the schema, Claude would be forced to pick a wrong valid option.
- **`TurnOutput`**: what Claude returns every turn:
  - `extracted`, `reply`,
  - `intent` (`answer | correction | confirm | reject | off_topic | refuse | stop`); `stop` lets the student end early,
  - `refused_fields`: which questions the student declined, so the code doesn't have to guess from question order,
  - `sentiment` (`neutral | positive | frustrated`), `language`.
- **`ConversationRecord`**: the single JSON document per session: id, version, phase, transcript with timestamps, profile, per-field errors, skipped fields, sentiment history, language, recommendations, summary, model and prompt version. It doubles as the session state.

### Catalog (`domain/catalog.py`)

- Each entry has `excluded_specializations` as a list (none, one or several) and a nested `ideal_profile` (`magic_orientation`, `learning_modality`), matching the spec's grouping.
- `filter_catalog` checks `min_school_year`, `min_free_hours_required` and `excluded_specializations`, and keeps every reason an entry was rejected, so the bot can explain itself when nothing qualifies.
- `rank_by_ideal_profile` ranks entries by orientation and learning-style matches without the LLM; it's the fallback when the recommendation call fails.

### Phases (`engine/phases.py`)

`CONSENT → COLLECTING → CONFIRMING → DONE`. Recommending happens inside the turn where the student confirms, so it isn't a separate phase.

- **Required fields** (`school_year`, `specialization`, `weekly_free_hours`) feed the hard filters and can't be skipped; if refused, the bot explains why and offers to end.
- **Skippable fields** (`interests`, `magic_orientation`, `learning_modality`) only affect ranking.
- The interview is ready for confirmation when no required or unskipped field is missing and there are no pending errors.

### Engine loop (`engine.handle_turn`)

1. Load the record and work on a copy: if the turn fails, the stored record stays exactly as it was and the student can repeat their message.
2. Empty or over-2000-character input gets a fixed reply without calling the LLM. A conversation that ended without an interview (consent declined) gets a fixed "this conversation has ended".
3. One Claude call with structured output returns a `TurnOutput`. The system prompt contains the static instructions plus the current state: phase, known values, pending fields, skipped fields, last turn's rejected values, recommendations given, and a frustration note after two frustrated turns in a row. The full transcript is sent as the message history, which gives the bot multi-turn memory.
4. `apply()` merges the extracted values; rejected ones are stored so the next prompt asks again.
5. Python decides phase changes, not the LLM:
   - consent: agree → COLLECTING; decline or stop → DONE.
   - collecting: refused skippable fields are marked skipped; once ready and something changed → CONFIRMING.
   - confirming: a new value → confirm again; `confirm` → recommend → DONE; `reject` / `correction` without a value → COLLECTING.
   - done: follow-up questions are answered from the recommendations; a profile correction → CONFIRMING again.
   - `stop` at any point → DONE with a summary.
6. Save through the repository, which rejects stale versions.

**LLM calls per turn.** Collecting turns make one call. Transitions add dedicated calls so the student sees exactly what's stored: the confirmation message (one call), and on confirm the recommendation and the summary (two calls). Expect 2–3 calls on the confirming turn.

### Recommender

- Python filters the catalog; a structured call ranks what passes against `magic_orientation`, `learning_modality` and `interests`, returning `[{name, reason}]` and a spoken reply.
- Names not in the filtered list are dropped. If any are dropped, the LLM's reply is discarded (it may mention them) and a fixed English message lists the valid ones.
- If the call fails, the deterministic ranking is used instead.
- If nothing passes the filters, an LLM call explains the reasons in the student's language; a fixed English message is used if that call fails.

### Summary

When the conversation ends, one LLM call writes 3–6 key points in English for school staff. The profile and recommendations are copied in from the record, so the summary can't contradict them. If the call fails, the summary stays empty and the student's turn still succeeds.

### Claude adapter (`llm/claude.py`)

- Model and effort come from `ANTHROPIC_MODEL` and `LLM_EFFORT` (defaults in `config.py`: `claude-opus-5-5`, `low`, since this is chat and latency matters). The adapter has no default of its own, so `config.py` is the single source. Each record stores the model that produced it.
- `messages.create` with `output_config.format` (JSON schema from the SDK's `transform_schema`), validated with Pydantic in the adapter. If validation fails (e.g. truncated output), one retry with the error attached.
- 20 s timeout and the SDK's built-in retries (`max_retries=2`) for timeouts, connection errors, 429 and 5xx.
- SDK errors map to two kinds: configuration errors (bad or missing key, bad request, unknown model) stop the app with a clear message; recoverable errors (unavailable, refusal, invalid output) only fail that turn with a fixed reply.
- `stop_reason` is checked before reading the output. Server-side `fallbacks: "default"` reroutes a refused request automatically.
- Each call logs latency, token counts and the request id.
- The API key is read from `.env` by the settings and passed to the client, because the SDK only reads the real environment.

### Speech (`speech/`)

- **Deepgram Nova-3** (`deepgram.py`): REST call with `detect_language=true`, for the multiple-languages bonus. The detected tag (`es-ES`) is reduced to the ISO 639-1 code (`es`) the engine uses.
- **ElevenLabs Flash v2.5** (`elevenlabs.py`): returns MP3; it detects the language from the text, so no language code is sent. The voice is set with `ELEVENLABS_VOICE_ID`.
- **ElevenLabs Scribe v2** (`elevenlabs.py`, optional STT): selected with `STT_PROVIDER=elevenlabs`, so one ElevenLabs key covers both directions. Scribe has no overall confidence, so the adapter averages the per-word probabilities (`exp(logprob)`, spacing ignored) to keep the same "unclear audio" threshold; 3-letter language codes are dropped. Deepgram stays the default because its free credit is separate, while Scribe shares the free plan's TTS credits.
- Each STT client reports its `name`, shown by `/health` and in the frontend, so the active provider is visible in the demo.
- Both use `httpx` with a 15 s timeout and turn every provider failure into a `SpeechError`, so callers handle one error type.

### Voice pipeline (`speech/pipeline.py`)

One voice turn: audio → STT → engine → TTS.

- An unknown conversation is rejected before calling Deepgram, so no transcription is paid for.
- Unclear audio (empty transcript or confidence below `STT_MIN_CONFIDENCE`, default 0.4) or an STT failure gets a "please repeat" reply without running the engine; the stored conversation doesn't change.
- If TTS fails, the reply is still returned as text without audio (`speak_or_none`, also used by the text endpoints).
- Logs the time spent in STT, engine and TTS for each turn.

### Wiring (`bootstrap.py`)

Builds the real engine, STT and TTS clients from the settings, so the CLI and the API share one setup and tests can pass fakes instead. `STT_PROVIDER` (`deepgram` default, or `elevenlabs`) picks the speech-to-text adapter. Speech is optional: without Deepgram or ElevenLabs keys, the STT/TTS clients are `None`, text still works and replies have no audio. An empty `KEY=` line in `.env` counts as not set.

### HTTP API (`interfaces/api.py`)

`create_app()` is an app factory (`make serve` runs `uvicorn --factory` on :8000); tests pass fake components to it.

| Endpoint | What it does |
|---|---|
| `GET /health` | Status, whether STT and TTS are configured, and the STT provider name |
| `POST /sessions` | Starts a conversation and returns the consent message; `{"speak": true}` adds audio |
| `GET /sessions/{id}` | The full saved record |
| `POST /sessions/{id}/messages` | Text turn: `{"text": "...", "speak": false}` |
| `POST /sessions/{id}/audio` | Voice turn: `multipart/form-data` upload in a field named `audio` |

- Every turn returns the same response: transcript, reply, base64 audio and its media type, phase, language, profile, field errors, recommendations and summary, so the frontend's debug panel needs no extra request.
- Ids are checked against the generated format in the path itself.
- Errors: 404 unknown conversation, 409 two turns at once on the same conversation, 413 clip too large (`MAX_AUDIO_BYTES`), 415 not audio, 400 empty clip, 503 LLM or speech not configured. The 503 never includes provider error details.
- CORS allows only `CORS_ORIGINS` (default `http://localhost:5173`).
- Endpoints are synchronous; FastAPI runs them in a thread pool.
- No authentication: anyone with a conversation id can read it. Ids are random 32-character strings, acceptable for a demo; listed as a README improvement.

## 3. Local environment

| Piece | Choice |
|---|---|
| Python and dependencies | Python 3.12 in `.venv` (`make install`), `anthropic`, `pydantic`, `pydantic-settings`, `fastapi`, `uvicorn`, `python-multipart`, `httpx`, `boto3` |
| Voice | The React frontend in the browser (see section 4). No audio libraries on the Python side |
| Frontend | Node 20+, Vite + React + TypeScript; `npm run dev` on :5173 |
| Storage | `STORAGE=file` saves to `./conversations/{id}.json`. Writes go to a temp file then are renamed (no half-written records), and only generated ids are accepted (no path traversal) |
| Commands | `make run-text` (`ARGS=-v` shows phase and profile, `ARGS="--resume ID"` continues a conversation), `make serve` (FastAPI on :8000), `make frontend` (Vite on :5173), `make test` |
| Docker | `docker compose up` runs the API (`STORAGE=file`, mounted volume) and the frontend. Add LocalStack only if you want to test S3 storage locally |
| Code quality | `ruff` + `mypy --strict` (`make lint typecheck`) |

### Tests

No network: the fake LLM returns queued outputs or errors and records each call's prompts; the Claude adapter is tested against a fake SDK client.

- each Pydantic field's validation;
- merging partial values and corrections;
- phase changes, including reject → back to collecting;
- catalog filtering, and the case where nothing passes;
- dropping recommendations that aren't in the catalog;
- LLM timeout, malformed output and refusal paths;
- Deepgram, ElevenLabs TTS and Scribe STT request/response handling against a mocked HTTP transport, including Scribe's confidence averaging and language codes;
- STT provider selection in `bootstrap.py`;
- voice pipeline: empty or low-confidence transcript, STT failure, TTS failure, unknown conversation;
- 2–3 complete scripted conversations;
- FastAPI tests with `TestClient`: every endpoint, every error status, CORS, with fake STT/TTS;
- the S3 repository tested with `moto`, which mocks AWS.

## 4. Voice frontend (React)

```
Browser (React)                                  FastAPI (local or AWS)
 hold button → getUserMedia + MediaRecorder
 release     → POST /sessions/{id}/audio  ─────▶ STT → engine → TTS
 show text + play audio          ◀───────────── { transcript, reply_text, audio_base64, phase, profile }
```

- **Recording:** `navigator.mediaDevices.getUserMedia({ audio: true })` and `MediaRecorder` record WebM/Opus, which Deepgram accepts directly, so the backend doesn't convert audio.
- **Playback:** the ElevenLabs MP3 comes back base64-encoded in the JSON and plays with `new Audio("data:audio/mpeg;base64,...")`. JSON also avoids configuring binary responses in API Gateway and Lambda.
- **UI (~150 lines):** a hold-to-talk button, the transcript, a text box that calls `/messages` for testing without speaking, and a debug panel showing `phase` and `profile` filling in live.
- **API:** the endpoints in section 2 (HTTP API). On AWS, `CORS_ORIGINS` is set to the frontend's domain.
- **Secure page required:** browsers only grant microphone access over HTTPS or on `localhost`. Local testing works; `http://192.168.x.x` from another device doesn't.
- **Latency:** about 1–3 s per turn, because upload, Deepgram, Claude and ElevenLabs run one after another. Fine for a demo; streaming each step is listed as a README improvement.

## 5. AWS deployment

The same API (section 2) is deployed unchanged. Because the engine stores nothing in memory, any instance can handle any request.

**Recommended setup: AWS Lambda running the container image, behind API Gateway.** It costs almost nothing while idle, there are no servers to manage, and the same Docker image works locally and in Lambda (through the `mangum` adapter).

```
Browser ── CloudFront + S3 (built React app)
   └────── API Gateway (HTTP API) ── Lambda (container, FastAPI+Mangum)
                                          ├── S3: conversations/{id}.json   (STORAGE=s3)
                                          ├── Secrets Manager: API keys
                                          ├── CloudWatch: structured logs (turn latency, tokens, phase)
                                          └── Claude API / Deepgram / ElevenLabs
```

**Infrastructure as code:** one small stack using AWS CDK in Python (same language as the app) or Terraform. Resources:

- an ECR repository for the image,
- the Lambda function (1 GB memory, 30 s timeout),
- the HTTP API,
- an S3 bucket with encryption and a lifecycle rule for conversations,
- an S3 bucket + CloudFront distribution for the frontend (HTTPS, needed for microphone access),
- the secrets,
- an IAM role with only the permissions it needs.

**Deployment from GitHub Actions:** run the tests, build the image, push it to ECR, build the frontend, then run `cdk deploy` (which uploads the frontend to S3).

**Ways to extend it** (for the README's scalability section; not built):

- **ECS Fargate + ALB:** if WebSockets are needed for streaming voice, or longer requests.
- **DynamoDB:** replaces S3 for session state if many writes per session cause conflicts.
- **SQS:** for generating summaries in the background.
- **Rate limits:** per-student limits at API Gateway.
- **Claude Platform on AWS:** the Python SDK's `AnthropicAWS` client adds AWS IAM and billing. Since it's just another LLM adapter, it also shows off the abstract interface.

**Notes:**

- Concurrent turns in the same session overwrite each other. Prevent this by checking the record's version number on write, using S3 conditional writes.
- Audio uploads must stay under the API Gateway payload limit. Push-to-talk clips are small, so this is fine.

## 6. Schedule

| Step | Hours | Done when |
|---|---|---|
| Models, catalog, repository interface + file storage | 0.75 | Validation tests pass |
| LLM interface, fake, Claude adapter, prompts, engine and phases | 1.75 | A complete conversation works in `make run-text` |
| Recommender, summary, sentiment, language | 0.75 | End-to-end JSON record is correct |
| Tests for error cases, scripted conversations | 0.75 | `make test` passes, coverage of `engine/` and `domain/` above 85% |
| STT/TTS adapters, FastAPI with CORS and audio endpoint | 0.75 | Audio round trip works with `curl` |
| React frontend | 1–1.5 | Voice conversation works in the browser on `localhost` |
| README, demo video | 1 | Deliverables complete |
| **Core total** | **~6.5–7** | |
| S3 repository, Dockerfile, docker-compose | 0.5 | `docker compose up` works |
| CDK stack (incl. frontend hosting), deploy, CI | 1–1.5 | Public HTTPS URL with working voice |

Do the AWS part only after the core is recorded. Even if it's skipped, the FastAPI service and the repository interface give a concrete answer to "how would this scale?"
