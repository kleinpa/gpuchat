// main.cc — SIP voice chatbot entrypoint.
//
// Parses CLI flags, loads YAML config, generates the startup greeting,
// initialises PJSIP, wires client objects into BotCall, then runs the
// event loop.  All core logic lives in bot_core.cc.
//
// Stack
//   • PJSUA2 (PJSIP C++ API)  — SIP / RTP / G.722 wideband (16 kHz) codec
//   • Wyoming faster-whisper   — speech-to-text (VAD → PCM → transcript)
//   • Ollama HTTP API          — LLM inference (streaming token output)
//   • Wyoming piper            — text-to-speech (sentence → PCM audio)
//
// Audio flow
//   RX: RTP/G.722 → PJSUA2 decode → 16 kHz PCM → VAD → Whisper → text
//   LLM: text → Ollama (streaming) → sentence tokens → Piper → PCM
//   TX: PCM chunks → AudioQueue → onFrameRequested → RTP/G.722

#include "bot_core.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <csignal>
#include <string>
#include <thread>

// ── CLI flags ─────────────────────────────────────────────────────────────────
ABSL_FLAG(std::string, whisper, "whisper:10300",
          "Wyoming faster-whisper endpoint (host:port)");
ABSL_FLAG(std::string, ollama, "http://ollama:11434", "Ollama HTTP endpoint");
ABSL_FLAG(std::string, piper, "piper:10200",
          "Wyoming piper TTS endpoint (host:port)");
ABSL_FLAG(std::string, config_file, "",
          "Path to YAML config file (model, system_prompt, greeting_prompt, "
          "vad, options)");
ABSL_FLAG(std::string, pbx, "",
          "SIP host for call transfers (e.g. 192.168.1.1 or pbx.local)");
ABSL_FLAG(int, port, 5060, "SIP listen port (UDP)");
ABSL_FLAG(std::string, public_addr, "",
          "Public IP to advertise in SDP/Contact (for NAT)");

// ── Signal handling ───────────────────────────────────────────────────────────
static std::atomic_bool g_running{true};
static void SigHandler(int) { g_running = false; }

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
  std::signal(SIGINT, SigHandler);
  std::signal(SIGTERM, SigHandler);
  std::signal(SIGPIPE, SIG_IGN);

  absl::SetProgramUsageMessage(
      "SIP voice chatbot — Whisper STT + Ollama LLM + Piper TTS.\n"
      "Dial sip:bot@<host> from any SIP softphone.");
  absl::ParseCommandLine(argc, argv);

  g_whisper_addr = absl::GetFlag(FLAGS_whisper);
  g_ollama_url = absl::GetFlag(FLAGS_ollama);
  g_piper_addr = absl::GetFlag(FLAGS_piper);
  g_pbx_host = absl::GetFlag(FLAGS_pbx);
  int sip_port = absl::GetFlag(FLAGS_port);
  std::string public_addr = absl::GetFlag(FLAGS_public_addr);

  static constexpr char kDefaultSystemPrompt[] =
      "You are a helpful voice assistant answering a phone call. "
      "Keep all responses short and conversational — one to three sentences "
      "at most. "
      "Never use emoji, bullet points, markdown, or lists. "
      "Speak in plain prose as if talking to someone on the phone. "
      "If the caller asks to be transferred or connected to someone, end your "
      "reply "
      "with the token [TRANSFER:NNNN] where NNNN is the 4-digit extension. "
      "For example: \"Connecting you now. [TRANSFER:1042]\" — "
      "the token is never spoken aloud, only the surrounding text is.";

  static constexpr char kDefaultGreetingPrompt[] =
      "Greet the caller warmly in one short sentence, consistent with your "
      "persona.";

  std::string greeting_prompt = kDefaultGreetingPrompt;

  std::string config_file = absl::GetFlag(FLAGS_config_file);
  if (!config_file.empty()) {
    try {
      YAML::Node cfg = YAML::LoadFile(config_file);

      if (cfg["system_prompt"])
        g_system_prompt = cfg["system_prompt"].as<std::string>();
      if (cfg["greeting_prompt"])
        greeting_prompt = cfg["greeting_prompt"].as<std::string>();
      if (cfg["model"] && !cfg["model"].as<std::string>().empty())
        g_model = cfg["model"].as<std::string>();
      if (cfg["max_calls"] && cfg["max_calls"].IsScalar())
        g_max_calls = cfg["max_calls"].as<int>();

      if (cfg["vad"] && cfg["vad"].IsMap()) {
        const YAML::Node& vad = cfg["vad"];
        if (vad["threshold"]) g_vad_threshold = vad["threshold"].as<double>();
        if (vad["silence_ms"]) g_silence_ms = vad["silence_ms"].as<int>();
        if (vad["min_speech_ms"])
          g_min_speech_ms = vad["min_speech_ms"].as<int>();
      }

      if (cfg["options"] && cfg["options"].IsMap()) {
        for (const auto& kv : cfg["options"]) {
          std::string key = kv.first.as<std::string>();
          const YAML::Node& val = kv.second;
          if (val.IsScalar()) {
            try {
              g_model_options[key] = val.as<double>();
              continue;
            } catch (...) {}
            try {
              g_model_options[key] = val.as<int>();
              continue;
            } catch (...) {}
            g_model_options[key] = val.as<std::string>();
          }
        }
      }

      LOG(INFO) << "[config] loaded from " << config_file;
    } catch (const YAML::Exception& e) {
      LOG(WARNING) << "[config] failed to load " << config_file << ": "
                   << e.what();
    }
  }

  if (g_system_prompt.empty()) {
    g_system_prompt = kDefaultSystemPrompt;
    LOG(INFO) << "[config] using default system prompt";
  } else {
    LOG(INFO) << "[config] system prompt loaded (" << g_system_prompt.size()
              << " chars)";
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  // Synthesise the greeting in the background so PJSIP starts up (and the
  // SIP port opens) immediately.  Calls that arrive before the greeting is
  // ready will play silence instead.
  std::thread greeting_thread([greeting_prompt] {
    LOG(INFO) << "[startup] generating greeting (background)";
    auto llm = MakeOllamaClient();
    auto tts = MakeWyomingTtsClient();
    llm->Chat(
        greeting_prompt, nlohmann::json::array(),
        [&](const std::string& sentence) {
          g_greeting_text += sentence;
          auto pcm = tts->Synthesize(sentence);
          if (!pcm.empty()) {
            g_greeting_pcm.insert(g_greeting_pcm.end(), pcm.begin(), pcm.end());
          } else {
            LOG(WARNING) << "[startup] Piper returned no PCM for: " << sentence;
          }
        },
        "");
    if (g_greeting_text.empty())
      LOG(WARNING) << "[startup] greeting is empty — Ollama may be unreachable";
    else
      LOG(INFO) << "[startup] greeting: " << g_greeting_text;
  });

  pj::Endpoint ep;
  try {
    ep.libCreate();

    pj::EpConfig cfg;
    cfg.uaConfig.maxCalls = g_max_calls;
    cfg.logConfig.level = 3;
    cfg.logConfig.consoleLevel = 3;
    cfg.medConfig.clockRate = kAudioRate;   // conference bridge clock (16000)
    cfg.medConfig.sndClockRate = kAudioRate;  // sound device clock (16000)
    cfg.medConfig.noVad = true;
    cfg.medConfig.ecTailLen = 0;
    ep.libInit(cfg);

    pj::TransportConfig transport_cfg;
    transport_cfg.port = sip_port;
    if (!public_addr.empty()) transport_cfg.publicAddress = public_addr;
    ep.transportCreate(PJSIP_TRANSPORT_UDP, transport_cfg);
    try {
      ep.transportCreate(PJSIP_TRANSPORT_UDP6, transport_cfg);
    } catch (...) {}

    ep.libStart();
    ep.audDevManager().setNullDev();

    // G.722 wideband (16 kHz, PT=9) — sole codec; disable all others.
    // PJSIP registers G.722 with clock_rate=16000 (the actual audio rate);
    // the RTP payload type 9 clock of 8000 Hz is only in the SDP, not the
    // codec manager ID.  The ID used here must match g722_enum_codecs().
    ep.codecSetPriority("G722/16000", 255);  // internal 16 kHz audio clock
    ep.codecSetPriority("PCMU/8000", 0);     // disable G.711 µ-law
    ep.codecSetPriority("PCMA/8000", 0);     // disable G.711 A-law

    {
      pj::AccountConfig account_cfg;
      account_cfg.idUri = "sip:bot@0.0.0.0";
      BotAccount account;
      account.create(account_cfg);

      LOG(INFO) << "SIP chatbot listening on UDP port " << sip_port;
      LOG(INFO) << "Whisper: " << g_whisper_addr << "  (Wyoming)";
      LOG(INFO) << "Ollama:  " << g_ollama_url << "  model=" << g_model;
      LOG(INFO) << "Piper:   " << g_piper_addr << "  (Wyoming)";

      while (g_running) ep.libHandleEvents(100);
    }

    ep.libDestroy();
  } catch (pj::Error& e) {
    LOG(FATAL) << "Fatal: " << e.info();
    greeting_thread.detach();
    return 1;
  }
  if (greeting_thread.joinable()) greeting_thread.join();
  return 0;
}
