from pathlib import Path

from agent.bootstrap import build_engine, build_stt, build_tts
from agent.config import Settings
from agent.speech.deepgram import DeepgramSTT
from agent.speech.elevenlabs import ElevenLabsSTT, ElevenLabsTTS


def settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, conversations_dir=tmp_path, **overrides)  # type: ignore[call-arg, arg-type]


def test_engine_uses_the_configured_model(tmp_path: Path) -> None:
    engine = build_engine(settings(tmp_path, anthropic_model="claude-sonnet-5-5"))
    assert engine._llm.model_name == "claude-sonnet-5-5"


def test_speech_is_disabled_without_keys(tmp_path: Path) -> None:
    empty = settings(tmp_path, deepgram_api_key="", elevenlabs_api_key="")
    assert build_stt(empty) is None
    assert build_tts(empty) is None


def test_speech_clients_are_built_with_keys(tmp_path: Path) -> None:
    configured = settings(tmp_path, deepgram_api_key="dg", elevenlabs_api_key="el")
    assert isinstance(build_stt(configured), DeepgramSTT)
    assert isinstance(build_tts(configured), ElevenLabsTTS)


def test_deepgram_is_the_default_stt_provider(tmp_path: Path) -> None:
    configured = settings(tmp_path, deepgram_api_key="dg", elevenlabs_api_key="el")
    assert configured.stt_provider == "deepgram"
    assert isinstance(build_stt(configured), DeepgramSTT)


def test_elevenlabs_stt_reuses_the_elevenlabs_key(tmp_path: Path) -> None:
    configured = settings(
        tmp_path, stt_provider="elevenlabs", deepgram_api_key="", elevenlabs_api_key="el"
    )
    assert isinstance(build_stt(configured), ElevenLabsSTT)


def test_elevenlabs_stt_without_key_is_disabled(tmp_path: Path) -> None:
    configured = settings(
        tmp_path, stt_provider="elevenlabs", deepgram_api_key="dg", elevenlabs_api_key=""
    )
    assert build_stt(configured) is None
