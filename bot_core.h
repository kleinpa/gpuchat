#pragma once
// bot_core.h — SIP chatbot core: client interfaces, per-call session, SIP
// classes.
//
// Abstract client interfaces (injectable for tests):
//   AsrClient — PCM → transcript
//   TtsClient — text → PCM
//   LlmClient — streaming LLM chat
//
// Real implementations (defined in bot_core.cc):
//   WyomingAsrClient — STT via Wyoming faster-whisper (TCP)
//   WyomingTtsClient — TTS via Wyoming piper (TCP)
//   OllamaClient     — LLM chat via Ollama HTTP API (streaming)
//
// Session      — per-call pipeline; owns asr/tts/llm by unique_ptr
// BotAudioPort — PJSUA2 audio port wired to Session
// BotCall / BotAccount — PJSIP call/account handlers
//
// Globals are declared here (defined in bot_core.cc) so that main.cc and
// tests can set them directly.

#include <pjsua2.hpp>

#include "chatbot_lib.h"

#include "absl/log/log.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ── Runtime globals (defined in bot_core.cc) ─────────────────────────────────
// Connection addresses — set by main() from flags or config.
extern std::string g_whisper_addr;
extern std::string g_ollama_url;
extern std::string g_piper_addr;
extern std::string g_pbx_host;
// LLM config.
extern std::string g_model;
extern std::string g_system_prompt;
extern int g_max_calls;
extern nlohmann::json g_model_options;
// VAD tuning (overridable via YAML).
extern double g_vad_threshold;
extern int g_silence_ms;
extern int g_min_speech_ms;
// Pre-synthesized greeting (generated once at startup).
extern std::vector<int16_t> g_greeting_pcm;
extern std::string g_greeting_text;

// ── Client interfaces ─────────────────────────────────────────────────────────

// Speech-to-text: PCM → transcript (empty on failure/silence).
class AsrClient {
 public:
  virtual ~AsrClient() = default;
  virtual std::string Transcribe(const std::vector<int16_t>& pcm) = 0;
};

// Text-to-speech: text → PCM at kAudioRate (empty on failure).
class TtsClient {
 public:
  virtual ~TtsClient() = default;
  virtual std::vector<int16_t> Synthesize(const std::string& text) = 0;
};

// LLM chat: streams sentences via on_sentence; returns full response text.
class LlmClient {
 public:
  virtual ~LlmClient() = default;
  virtual std::string Chat(
      const std::string& user_text,
      const nlohmann::json& history,
      std::function<void(const std::string& sentence)> on_sentence,
      const std::string& caller) = 0;
};

// Real implementations wired to the global address variables.
std::unique_ptr<AsrClient> MakeWyomingAsrClient();
std::unique_ptr<TtsClient> MakeWyomingTtsClient();
std::unique_ptr<LlmClient> MakeOllamaClient();

// ── VAD parameters ────────────────────────────────────────────────────────────
struct VadParams {
  double threshold = kVadThreshold;
  int silence_ms = kSilenceMs;
  int min_speech_ms = kMinSpeechMs;
};

// ── Per-call session ──────────────────────────────────────────────────────────
// Owns the three AI clients and drives the STT→LLM→TTS pipeline for one call.
// Construct with all dependencies; call Start() to launch the pipeline thread.
class Session {
 public:
  Session(std::string caller,
          VadParams vad_params,
          std::function<void(const std::string& ext)> transfer_fn,
          std::unique_ptr<AsrClient> asr,
          std::unique_ptr<TtsClient> tts,
          std::unique_ptr<LlmClient> llm);
  ~Session();

  // Not copyable or movable — owns a thread and mutexes.
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  void Start();

  // Called from the PJSUA2 audio thread — feeds incoming PCM into the VAD.
  void ProcessRxFrame(const int16_t* samples, int n);

  // Pipeline thread reads from this; BotAudioPort drains it.
  AudioQueue audio_out;
  std::atomic_bool active{true};

 private:
  void SubmitUtterance(std::vector<int16_t> pcm);
  void AppendHistory(const std::string& role, const std::string& content);
  void SpeakAndLog(const std::string& text);
  void RunPipeline();

  std::string caller_;
  std::function<void(const std::string& ext)> transfer_fn_;

  std::unique_ptr<AsrClient> asr_;
  std::unique_ptr<TtsClient> tts_;
  std::unique_ptr<LlmClient> llm_;

  Vad vad_;

  nlohmann::json history_ = nlohmann::json::array();
  std::mutex history_mu_;

  std::thread pipeline_thread_;
  std::queue<std::vector<int16_t>> pending_audio_;
  std::mutex pending_mu_;
  std::condition_variable pending_cv_;
};

// ── Custom audio media port ───────────────────────────────────────────────────
class BotAudioPort : public pj::AudioMediaPort {
 public:
  explicit BotAudioPort(Session& session);

  void onFrameRequested(pj::MediaFrame& frame) override;
  void onFrameReceived(pj::MediaFrame& frame) override;

 private:
  Session& session_;
};

// ── SIP call ──────────────────────────────────────────────────────────────────
class BotCall : public pj::Call {
 public:
  explicit BotCall(pj::Account& acc, int call_id = PJSUA_INVALID_ID);
  ~BotCall() override;

  void onCallState(pj::OnCallStateParam& prm) override;
  void onCallMediaState(pj::OnCallMediaStateParam& prm) override;

 private:
  std::unique_ptr<Session> session_;
  std::unique_ptr<BotAudioPort> port_;
};

// ── SIP account ───────────────────────────────────────────────────────────────
class BotAccount : public pj::Account {
 public:
  void onIncomingCall(pj::OnIncomingCallParam& prm) override;
};
