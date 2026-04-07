# offhook

SIP voice assistant: answers calls, transcribes speech with faster-whisper,
generates responses with Ollama, and speaks replies back with Piper TTS.
Audio is streamed over standard RTP/G.711. Piper synthesis starts on the
first complete sentence from Ollama (not after the full LLM response) for low
latency. Multiple concurrent calls are each handled by an independent pipeline
thread. A greeting is generated once at startup and played instantly to every
caller.

## Architecture

```
Caller ──SIP/RTP──► PJSUA2 ──PCM──► VAD ──utterance──► Wyoming faster-whisper
                                                              │ transcript
                                                              ▼
                                                       Ollama (streaming)
                                                              │ sentence tokens
                                                              ▼
                                                       Wyoming Piper TTS
                                                              │ PCM chunks
Caller ◄─SIP/RTP── PJSUA2 ◄─PCM─── AudioQueue ◄────────────┘
```

Audio flows as 8 kHz 16-bit PCM (160 samples per 20 ms RTP frame). Each call
owns its own Session with an independent pipeline thread and audio queue. Up
to `--max_calls` (default 8) concurrent calls are supported.

Whisper and Piper communicate over the
[Wyoming protocol](https://github.com/rhasspy/wyoming) — a lightweight
newline-delimited JSON + binary framing over plain TCP.

## Dependencies (external services)

| Service        | Default address       | Protocol         | Docker image                     |
| -------------- | --------------------- | ---------------- | -------------------------------- |
| faster-whisper | `whisper:10300`       | Wyoming TCP      | `rhasspy/wyoming-faster-whisper` |
| Ollama         | `http://ollama:11434` | HTTP (streaming) | `ollama/ollama`                  |
| Piper          | `piper:10200`         | Wyoming TCP      | `rhasspy/wyoming-piper`          |

## Build

```sh
bazel build -c opt //:offhook
```

## Usage

```sh
# Minimal — all services on localhost with defaults
bazel run //:offhook

# Custom endpoints and config file
bazel run //:offhook -- \
  --whisper     whisper:10300 \
  --ollama      http://ollama:11434 \
  --piper       piper:10200 \
  --config_file bot_config.yaml \
  --port        5060
```

Dial in with any SIP softphone (Linphone, Zoiper, Baresip, etc.) using
`sip:bot@<server-ip>`.

## Flags

| Flag            | Default               | Description                                      |
| --------------- | --------------------- | ------------------------------------------------ |
| `--whisper`     | `whisper:10300`       | Wyoming faster-whisper endpoint (`host:port`)    |
| `--ollama`      | `http://ollama:11434` | Ollama HTTP endpoint                             |
| `--piper`       | `piper:10200`         | Wyoming Piper TTS endpoint (`host:port`)         |
| `--config_file` | _(empty)_             | Path to YAML config file (see below)             |
| `--pbx`         | _(empty)_             | SIP host for call transfers (e.g. `192.168.1.1`) |
| `--port`        | `5060`                | SIP UDP listen port                              |
| `--public_addr` | _(empty)_             | Public IP for NAT/SDP                            |

## Config file

All behaviour is controlled through a YAML file passed with `--config_file`.
All fields are optional; omitted ones use built-in defaults.

```yaml
# Ollama model name (default: gemma3:1b)
# model: gemma3:1b

# System prompt — multi-line supported
system_prompt: |
  You are a helpful voice assistant answering a phone call.
  Keep all responses short and conversational.

# Prompt used to generate the pre-recorded startup greeting
greeting_prompt: Greet the caller warmly in one short sentence.

# Maximum simultaneous SIP calls (default: 8)
# max_calls: 8

# VAD tuning
# vad:
#   threshold: 500     # RMS level for voiced frame; lower = more sensitive
#   silence_ms: 600    # ms of silence to end an utterance
#   min_speech_ms: 200 # minimum utterance length; shorter bursts discarded

# Model options — forwarded verbatim to Ollama's /api/chat "options" object.
# Any Ollama option can be added here without recompiling.
options:
  temperature: 0.2
  top_k: 64
  top_p: 0.95
  min_p: 0.01
  repeat_penalty: 1.0
  # num_ctx: 4096
```

See the [Ollama modelfile docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
for the full list of supported options.

## Thinking models

Models that emit `<think>...</think>` blocks (Qwen3, Gemma thinking variants,
etc.) are handled automatically — thinking tokens are stripped from the TTS
stream so callers never hear them. The full response is still stored in
conversation history.

## VAD tuning

Voice activity detection uses a simple RMS energy threshold, configurable via
the `vad` section of the config file:

- `threshold` (default 500): RMS level above which a frame is considered
  voiced. Lower = more sensitive.
- `silence_ms` (default 600): milliseconds of silence required to end an
  utterance and trigger recognition.
- `min_speech_ms` (default 200): minimum utterance length; shorter bursts are
  discarded.
