# Implementation Plan: Magic School Interview Agent

Based on [SPECT.md](SPECT.md).

The core is one engine with no saved state between calls. It takes a conversation record and the student's message, and returns the reply plus the updated record. Locally it runs from a text command line, and behind a small HTTP API that a React frontend uses for voice. On AWS the same API and frontend are deployed. Nothing in the engine knows where it's running.

## 1. Project layout

```
orbio_agent/
├── pyproject.toml          # uv; Python 3.12
├── Makefile                # install, test, lint, run-text, serve, frontend, docker
├── Dockerfile / docker-compose.yml
├── .env.example            # ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, STORAGE=file|s3
├── data/catalog.json       # ~15 clubs/classes
├── src/agent/
│   ├── config.py           # pydantic-settings, reads env
│   ├── domain/
│   │   ├── models.py       # StudentProfile, TurnOutput, CatalogEntry, ConversationRecord
│   │   └── catalog.py      # load + hard-filter (pure Python)
│   ├── engine/
│   │   ├── engine.py       # Engine.handle_turn(record, text) -> (reply, record)
│   │   ├── phases.py       # CONSENT → COLLECTING → CONFIRMING → RECOMMENDING → DONE
│   │   └── prompts.py      # system prompt, per-turn context, recommender, summary
│   ├── llm/      base.py (LLMClient ABC) · claude.py · fake.py
│   ├── speech/   base.py (STT, TTS ABCs) · deepgram.py · elevenlabs.py · fake.py
│   ├── storage/  base.py (ConversationRepository ABC) · file_repo.py · s3_repo.py
│   └── interfaces/
│       ├── cli.py          # text only (also what the tests drive)
│       └── api.py          # FastAPI: used by the frontend, deployed to AWS
├── frontend/               # Vite + React + TypeScript: push-to-talk voice UI
└── tests/
```

## 2. Core design

### Data models (`domain/models.py`)

- **`StudentProfile`**: all fields optional, with enums and ranges enforced by Pydantic (`school_year: int = Field(ge=1, le=7)`, etc.). A helper method lists which fields are still missing.
- **`TurnOutput`**: what Claude returns every turn:
  - `extracted` (partial profile values),
  - `reply`,
  - `intent` (`answer | correction | confirm | reject | off_topic | refuse`),
  - `sentiment` (`neutral | positive | frustrated`),
  - `language`.
- **`ConversationRecord`**: the single JSON document per session:
  - id and phase,
  - transcript with timestamps,
  - the profile, and per-field validation errors,
  - sentiment history and language,
  - recommendations, summary, model version and prompt version.

  It doubles as the session state, so storing the conversation and resuming it use the same object.

### Engine loop (`engine.handle_turn`)

1. Add the student's message to the transcript.
2. Make one Claude call with structured output (`client.messages.parse(..., output_format=TurnOutput)`). The prompt includes:
   - the current phase,
   - the fields still missing,
   - last turn's validation errors,
   - the full history, which is what gives the bot multi-turn memory.
3. Validate each extracted field on its own. Valid values go into the profile; invalid ones are kept as errors so the next prompt asks again ("year 9 isn't possible, years run 1–7"). One bad field doesn't throw away the good ones.
4. Python decides phase changes, not the LLM:
   - no fields missing → CONFIRMING;
   - `confirm` → RECOMMENDING;
   - `correction` / `reject` → back to COLLECTING.
5. Sentiment: two frustrated turns in a row switch the prompt to shorter mode, which offers to skip a question or finish early.
6. Save the record through the repository.

### Recommender

- Python filters the catalog on `min_school_year`, `min_free_hours_required` and `excluded_specialization`.
- A second structured call ranks what passes against `magic_orientation`, `learning_modality` and `interests`, returning `[{name, reason}]`.
- Guard against made-up entries: drop any name that isn't in the filtered list.
- If nothing passes the filters, skip the LLM and give a fixed explanation built from the filter reasons.

### Summary

When the conversation ends, one LLM call produces the key points. The profile and recommendations are copied in from the record, not regenerated, so the summary can't contradict them.

### Claude adapter (`llm/claude.py`)

- Model: `claude-opus-5-5`, set by an environment variable. Use `effort: "low"` for interview turns, since this is chat and latency matters.
- `timeout=20s`, and the SDK's built-in `max_retries=2`.
- Handle errors with specific SDK exception types rather than one catch-all: rate limits and server errors are retried, a 400 is a bug and isn't. If a call still fails, the bot sends a fixed "sorry, could you repeat that?" without changing the record.
- Check `stop_reason` before reading the output. On `refusal`, send a fixed reply.
- Server-side `fallbacks: "default"` reroutes a refused request automatically. It's on by default and is one line to remove.
- If the output fails validation, retry once with the validation error attached.

### Speech

- **Deepgram Nova-3:** language detection or multilingual mode, for the multiple-languages bonus. An empty or low-confidence transcript gets "I didn't catch that."
- **ElevenLabs Flash v2.5:** if speech fails, show the text and continue.

## 3. Local environment

| Piece | Choice |
|---|---|
| Python and dependencies | `uv`, Python 3.12, `anthropic`, `pydantic`, `pydantic-settings`, `fastapi`, `uvicorn`, `python-multipart`, `httpx`, `boto3` |
| Voice | The React frontend in the browser (see section 4). No audio libraries on the Python side |
| Frontend | Node 20+, Vite + React + TypeScript; `npm run dev` on :5173 |
| Storage | `STORAGE=file` saves to `./conversations/{id}.json` |
| Commands | `make run-text`, `make serve` (FastAPI on :8000), `make frontend` (Vite on :5173), `make test` |
| Docker | `docker compose up` runs the API (`STORAGE=file`, mounted volume) and the frontend. Add LocalStack only if you want to test S3 storage locally |
| Code quality | `ruff` + `mypy`; a pre-commit hook is optional |

### Tests

No network; the fake LLM returns scripted `TurnOutput`s.

- each Pydantic field's validation;
- merging partial values and corrections;
- phase changes, including reject → back to collecting;
- catalog filtering, and the case where nothing passes;
- dropping recommendations that aren't in the catalog;
- LLM timeout, malformed output and refusal paths;
- empty STT transcript and TTS failure;
- 2–3 complete scripted conversations;
- FastAPI tests with `TestClient`, including the audio endpoint with fake STT/TTS;
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
- **API changes:** `CORSMiddleware` allowing `http://localhost:5173` locally and the frontend domain on AWS; audio received with `UploadFile`; a Pydantic response model for the reply.
- **Secure page required:** browsers only grant microphone access over HTTPS or on `localhost`. Local testing works; `http://192.168.x.x` from another device doesn't.
- **Latency:** about 1–3 s per turn, because upload, Deepgram, Claude and ElevenLabs run one after another. Fine for a demo; streaming each step is listed as a README improvement.

## 5. AWS deployment

The API needs three endpoints:

- `POST /sessions`, which returns the consent message;
- `POST /sessions/{id}/messages` for text;
- `POST /sessions/{id}/audio` for voice, which runs STT → engine → TTS and returns the reply as text plus audio.

Because the engine stores nothing in memory, any instance can handle any request.

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
