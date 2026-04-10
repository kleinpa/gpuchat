// main.cpp — SIP voice chatbot server
//
// Stack
//   • PJSUA2 (PJSIP C++ API)  — SIP / RTP / G.711 µ-law (PCMU) codec
//   • Wyoming faster-whisper   — speech-to-text (VAD → PCM → transcript)
//   • Ollama HTTP API          — LLM inference (streaming token output)
//   • Wyoming piper            — text-to-speech (sentence → PCM audio)
//
// Audio flow
//   RX: RTP/PCMU → PJSUA2 decode → PCM → VAD → Whisper → text
//   LLM: text → Ollama (streaming) → sentence tokens → Piper → PCM
//   TX: PCM chunks → AudioQueue → onFrameRequested → RTP/PCMU
//
// Latency strategy
//   • Piper synthesis starts on the first complete sentence from Ollama,
//     not after the full LLM response.  Sentences are split on '.', '!', '?'.
//   • PCM from Piper is queued in kFrameSamples-sized chunks.
//   • VAD uses a simple RMS energy threshold; silence for vad.silence_ms ms
//     triggers recognition.
//
// Concurrency
//   • Each SIP call owns its own Session (independent threads + queues).
//   • Up to --max_calls simultaneous calls are served concurrently.

#include <pjsua2.hpp>

#include "chatbot_lib.hpp"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#define CPPHTTPLIB_SEND_FLAGS MSG_NOSIGNAL
#define CPPHTTPLIB_NO_EXCEPTIONS
#include "httplib.h"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ── CLI flags ─────────────────────────────────────────────────────────────────
ABSL_FLAG(std::string, whisper,     "whisper:10300",
          "Wyoming faster-whisper endpoint (host:port)");
ABSL_FLAG(std::string, ollama,      "http://ollama:11434",
          "Ollama HTTP endpoint");
ABSL_FLAG(std::string, piper,       "piper:10200",
          "Wyoming piper TTS endpoint (host:port)");
ABSL_FLAG(std::string, config_file, "",
          "Path to YAML config file (model, system_prompt, greeting_prompt, vad, options)");
ABSL_FLAG(std::string, pbx,        "",
          "SIP host for call transfers (e.g. 192.168.1.1 or pbx.local)");
ABSL_FLAG(int,         port,        5060, "SIP listen port (UDP)");
ABSL_FLAG(std::string, public_addr, "",
          "Public IP to advertise in SDP/Contact (for NAT)");

// ── Runtime globals ───────────────────────────────────────────────────────────
static std::string g_whisper_addr;  // host:port for Wyoming faster-whisper
static std::string g_ollama_url;
static std::string g_piper_addr;    // host:port for Wyoming piper
static std::string g_pbx_host;      // SIP host used when building transfer URIs
static std::string g_model         = "gemma3:1b";
static std::string g_system_prompt;
static int         g_max_calls     = 8;

// ── Model options (populated from YAML config; passed verbatim to Ollama) ─────
static nlohmann::json g_model_options = {
    {"temperature",    0.2},
    {"top_k",          64},
    {"top_p",          0.95},
    {"min_p",          0.01},
    {"repeat_penalty", 1.0},
};

// ── VAD tuning params (overridable via YAML config) ───────────────────────────
static double g_vad_threshold = kVadThreshold;
static int    g_silence_ms    = kSilenceMs;
static int    g_min_speech_ms = kMinSpeechMs;

// Pre-synthesized greeting — generated once at startup so calls start instantly.
static std::vector<int16_t> g_greeting_pcm;
static std::string          g_greeting_text;

// ── TCP helpers (used by Wyoming protocol) ───────────────────────────────────

static int ConnectTcp(const std::string &host, int port) {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0) return -1;
    int fd = -1;
    for (auto *p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static bool SendAll(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        buf += n; len -= n;
    }
    return true;
}

// ── HTTP client ───────────────────────────────────────────────────────────────

// Streaming POST via cpp-httplib; calls cb(data, len) for each body chunk.
// Returns false if the connection or request fails.
static bool HttpPostStream(const std::string &base_url,
                           const std::string &path,
                           const std::string &content_type,
                           const std::string &body,
                           std::function<bool(const char *, size_t)> cb) {
    auto url = ParseUrl(base_url);
    httplib::Client cli(url.host, url.port);
    cli.set_read_timeout(120);

    httplib::Request req;
    req.method  = "POST";
    req.path    = path;
    req.headers = {{"Content-Type", content_type}};
    req.body    = body;
    req.content_receiver =
        [&](const char *data, size_t len, uint64_t /*off*/, uint64_t /*tot*/) {
            return cb(data, len);
        };

    httplib::Response res;
    httplib::Error    err = httplib::Error::Success;
    if (!cli.send(req, res, err)) {
        LOG(ERROR) << "[http] POST " << base_url << path
                   << " failed: " << httplib::to_string(err);
        return false;
    }
    if (res.status / 100 != 2) {
        LOG(ERROR) << "[http] POST " << base_url << path
                   << " status=" << res.status << " body=" << res.body;
        return false;
    }
    return true;
}

// ── Wyoming protocol helpers ──────────────────────────────────────────────────

// Parse "host:port" into components.
static std::pair<std::string, int> ParseHostPort(const std::string &addr,
                                                  int default_port) {
    auto colon = addr.rfind(':');
    if (colon == std::string::npos) return {addr, default_port};
    return {addr.substr(0, colon), std::stoi(addr.substr(colon + 1))};
}

// Send one Wyoming frame: header JSON + optional data JSON + optional payload.
static bool WyomingSend(int fd,
                        const std::string &type,
                        const std::string &data_json,   // "" if none
                        const void *payload = nullptr,
                        size_t payload_len = 0) {
    std::string header = "{\"type\":\"" + type + "\",\"version\":\"1.8.0\"";
    if (!data_json.empty())
        header += ",\"data_length\":" + std::to_string(data_json.size());
    if (payload && payload_len > 0)
        header += ",\"payload_length\":" + std::to_string(payload_len);
    header += "}\n";
    if (!SendAll(fd, header.c_str(), header.size())) return false;
    if (!data_json.empty() && !SendAll(fd, data_json.c_str(), data_json.size()))
        return false;
    if (payload && payload_len > 0 && !SendAll(fd, static_cast<const char*>(payload), payload_len))
        return false;
    return true;
}

// Read one Wyoming frame. Returns the event type; fills data_json and payload.
static std::string WyomingRecv(int fd,
                                std::string &data_json,
                                std::vector<uint8_t> &payload) {
    data_json.clear();
    payload.clear();

    // Read header line (terminated by '\n').
    std::string header;
    char ch;
    while (recv(fd, &ch, 1, 0) == 1) {
        if (ch == '\n') break;
        header += ch;
    }
    if (header.empty()) return {};

    std::string type = JsonGetString(header, "type");

    // data_length
    auto dl_pos = header.find("\"data_length\":");
    if (dl_pos != std::string::npos) {
        size_t dl = std::stoul(header.substr(dl_pos + 14));
        if (dl > 0) {
            data_json.resize(dl);
            size_t got = 0;
            while (got < dl) {
                ssize_t n = recv(fd, &data_json[got], dl - got, 0);
                if (n <= 0) return {};
                got += n;
            }
        }
    }

    // payload_length
    auto pl_pos = header.find("\"payload_length\":");
    if (pl_pos != std::string::npos) {
        size_t pl = std::stoul(header.substr(pl_pos + 17));
        if (pl > 0) {
            payload.resize(pl);
            size_t got = 0;
            while (got < pl) {
                ssize_t n = recv(fd, payload.data() + got, pl - got, 0);
                if (n <= 0) return {};
                got += n;
            }
        }
    }

    return type;
}

// ── Wyoming services (STT + TTS) ─────────────────────────────────────────────
static std::string Transcribe(const std::vector<int16_t> &pcm) {
    if (pcm.empty()) return {};
    auto [host, port] = ParseHostPort(g_whisper_addr, 10300);
    int fd = ConnectTcp(host, port);
    if (fd < 0) { LOG(ERROR) << "[stt] connect failed: " << g_whisper_addr; return {}; }

    // AudioChunk — send all PCM as one chunk at kAudioRate
    std::string audio_data = "{\"rate\":" + std::to_string(kAudioRate)
                           + ",\"width\":2,\"channels\":1,\"timestamp\":null}";
    const void *raw = pcm.data();
    size_t raw_len  = pcm.size() * sizeof(int16_t);
    if (!WyomingSend(fd, "audio-chunk", audio_data, raw, raw_len)) {
        close(fd); return {};
    }
    // AudioStop
    if (!WyomingSend(fd, "audio-stop", "{\"timestamp\":null}")) {
        close(fd); return {};
    }

    // Read until Transcript
    std::string result;
    for (int i = 0; i < 32; i++) {
        std::string data_json;
        std::vector<uint8_t> payload;
        std::string type = WyomingRecv(fd, data_json, payload);
        if (type.empty()) break;
        if (type == "transcript") {
            result = JsonGetString(data_json, "text");
            break;
        }
        if (type == "error") {
            LOG(ERROR) << "[stt] wyoming error: " << data_json;
            break;
        }
    }
    close(fd);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int16_t> Synthesize(const std::string &text) {
    if (text.empty()) return {};
    auto [host, port] = ParseHostPort(g_piper_addr, 10200);
    int fd = ConnectTcp(host, port);
    if (fd < 0) { LOG(ERROR) << "[tts] connect failed: " << g_piper_addr; return {}; }

    // Synthesize — include channels to avoid "# channels not specified" from piper
    std::string synth_data = "{\"text\":\"" + JsonEscape(text) + "\",\"channels\":1}";
    if (!WyomingSend(fd, "synthesize", synth_data)) {
        close(fd); return {};
    }

    // Collect AudioChunk payloads until AudioStop
    std::vector<int16_t> pcm;
    int piper_rate = kAudioRate;
    for (int i = 0; i < 4096; i++) {
        std::string data_json;
        std::vector<uint8_t> payload;
        std::string type = WyomingRecv(fd, data_json, payload);
        if (type.empty()) break;
        if (type == "audio-start") {
            auto rate_pos = data_json.find("\"rate\":");
            if (rate_pos != std::string::npos)
                piper_rate = std::stoi(data_json.substr(rate_pos + 7));
        } else if (type == "audio-chunk") {
            size_t n = payload.size() / 2;
            size_t base = pcm.size();
            pcm.resize(base + n);
            memcpy(pcm.data() + base, payload.data(), payload.size());
        } else if (type == "audio-stop") {
            break;
        } else if (type == "error") {
            LOG(ERROR) << "[tts] wyoming error: " << data_json;
            break;
        }
    }
    close(fd);

    if (pcm.empty()) { LOG(WARNING) << "[tts] empty response for: " << text; return {}; }

    // Downsample if piper uses a different rate (e.g. 22050 → 8000)
    if (piper_rate != kAudioRate) {
        double ratio = static_cast<double>(piper_rate) / kAudioRate;
        std::vector<int16_t> resampled;
        resampled.reserve(static_cast<size_t>(pcm.size() / ratio) + 1);
        for (size_t i = 0; i < pcm.size(); i += static_cast<size_t>(ratio))
            resampled.push_back(pcm[i]);
        return resampled;
    }
    return pcm;
}

// ── In-band audio cues ────────────────────────────────────────────────────────
// All synthesized locally; no external dependencies.  Returns 8 kHz mono s16le.

// Two-note ascending chime (E5 → A5) played on call answer.
static std::vector<int16_t> SynthesizeChime() {
    struct Note { float freq; float duration_ms; };
    constexpr Note notes[] = {{659.3f, 120.f}, {880.0f, 280.f}};
    constexpr float kDecay  = 6.0f;
    constexpr float kGain   = 0.35f;

    std::vector<int16_t> out;
    for (auto [freq, dur_ms] : notes) {
        int n = static_cast<int>(kAudioRate * dur_ms / 1000.f);
        out.reserve(out.size() + n);
        for (int i = 0; i < n; i++) {
            float t        = static_cast<float>(i) / kAudioRate;
            float envelope = std::exp(-kDecay * t);
            float sample   = kGain * envelope * std::sin(2.f * M_PI * freq * t);
            out.push_back(static_cast<int16_t>(sample * 32767.f));
        }
    }
    return out;
}

// Quick bright single note played the moment VAD enters listening state.
static std::vector<int16_t> SynthesizeSttListeningChime() {
    constexpr float kFreq     = 1047.0f;  // C6 — bright, high-pitched
    constexpr float kDuration = 80.f;     // ms
    constexpr float kDecay    = 10.0f;
    constexpr float kGain     = 0.30f;

    int n = static_cast<int>(kAudioRate * kDuration / 1000.f);
    std::vector<int16_t> out;
    out.reserve(n);
    for (int i = 0; i < n; i++) {
        float t        = static_cast<float>(i) / kAudioRate;
        float envelope = std::exp(-kDecay * t);
        float sample   = kGain * envelope * std::sin(2.f * M_PI * kFreq * t);
        out.push_back(static_cast<int16_t>(sample * 32767.f));
    }
    return out;
}

// Short single low-pitched click played when VAD captures an utterance.
static std::vector<int16_t> SynthesizeSegmentChime() {
    constexpr float kFreq     = 330.0f;   // E4 — lower than the answer chime
    constexpr float kDuration = 60.f;     // ms
    constexpr float kDecay    = 20.0f;
    constexpr float kGain     = 0.20f;    // quieter

    int n = static_cast<int>(kAudioRate * kDuration / 1000.f);
    std::vector<int16_t> out;
    out.reserve(n);
    for (int i = 0; i < n; i++) {
        float t        = static_cast<float>(i) / kAudioRate;
        float envelope = std::exp(-kDecay * t);
        float sample   = kGain * envelope * std::sin(2.f * M_PI * kFreq * t);
        out.push_back(static_cast<int16_t>(sample * 32767.f));
    }
    return out;
}

// Repeating quiet low beep played while waiting for LLM/TTS.
// Returns `duration_ms` worth of PCM with beeps every `interval_ms`.
static std::vector<int16_t> SynthesizeThinkingBeep(int duration_ms,
                                                    int interval_ms = 600) {
    constexpr float kFreq  = 220.0f;   // A3 — low and unobtrusive
    constexpr float kBeepMs = 80.f;
    constexpr float kDecay = 30.0f;
    constexpr float kGain  = 0.08f;    // very quiet

    int total   = kAudioRate * duration_ms / 1000;
    int spacing = kAudioRate * interval_ms / 1000;
    int beep_n  = static_cast<int>(kAudioRate * kBeepMs / 1000.f);

    std::vector<int16_t> out(total, 0);
    for (int start = 0; start < total; start += spacing) {
        for (int i = 0; i < beep_n && start + i < total; i++) {
            float t      = static_cast<float>(i) / kAudioRate;
            float env    = std::exp(-kDecay * t);
            float sample = kGain * env * std::sin(2.f * M_PI * kFreq * t);
            out[start + i] = static_cast<int16_t>(sample * 32767.f);
        }
    }
    return out;
}

// ── Ollama streaming chat ─────────────────────────────────────────────────────
static std::string OllamaChat(
    const std::string &user_text,
    const std::string &conversation_json,
    std::function<void(const std::string &)> on_sentence,
    const std::string &caller = "")
{
    nlohmann::json messages = nlohmann::json::array();
    if (!g_system_prompt.empty())
        messages.push_back({{"role", "system"}, {"content", g_system_prompt}});
    if (!conversation_json.empty() && conversation_json != "[]") {
        auto history = nlohmann::json::parse(conversation_json, nullptr, false);
        if (history.is_array())
            for (auto &m : history) messages.push_back(m);
    }
    messages.push_back({{"role", "user"}, {"content", user_text}});

    nlohmann::json req_json = {
        {"model",    g_model},
        {"messages", messages},
        {"options",  g_model_options},
        {"stream",   true},
    };
    std::string req_body = req_json.dump();

    LOG(INFO) << (caller.empty() ? "" : caller + " ")
              << "[llm] " << g_model << " ← " << user_text;

    std::string full_response;
    std::string sentence_buf;
    // Track whether we are inside a <think>...</think> block emitted by
    // reasoning models (e.g. Qwen3, Gemma thinking variants).  Tokens inside
    // the block are accumulated for the return value but must not be spoken.
    bool in_think = false;
    std::string think_tail;  // leftover from previous chunk while searching
    HttpPostStream(g_ollama_url, "/api/chat", "application/json", req_body,
        [&](const char *data, size_t len) -> bool {
            std::string chunk(data, len);
            std::istringstream ss(chunk);
            std::string line;
            while (std::getline(ss, line)) {
                if (line.empty()) continue;
                auto j = nlohmann::json::parse(line, nullptr, false);
                if (j.is_discarded()) continue;
                std::string token;
                if (j.contains("message") && j["message"].contains("content"))
                    token = j["message"]["content"].get<std::string>();
                if (token.empty()) continue;
                full_response += token;

                // Strip <think>...</think> spans before feeding to TTS.
                think_tail += token;
                std::string visible;
                while (!think_tail.empty()) {
                    if (in_think) {
                        auto end = think_tail.find("</think>");
                        if (end == std::string::npos) {
                            think_tail.clear();  // consumed, still thinking
                            break;
                        }
                        in_think = false;
                        think_tail = think_tail.substr(end + 8);
                    } else {
                        auto start = think_tail.find("<think>");
                        if (start == std::string::npos) {
                            visible += think_tail;
                            think_tail.clear();
                            break;
                        }
                        visible += think_tail.substr(0, start);
                        in_think = true;
                        think_tail = think_tail.substr(start + 7);
                    }
                }

                if (visible.empty()) continue;
                sentence_buf += visible;
                for (const auto &s : SplitSentences(sentence_buf))
                    on_sentence(s);
            }
            return true;
        });

    // Flush any trailing fragment (no terminal punctuation).
    {
        static constexpr char kTrimLeading[]  = " \t\r\n\"'`";
        static constexpr char kTrimTrailing[] = "\"'`";
        size_t a = sentence_buf.find_first_not_of(kTrimLeading);
        if (a != std::string::npos) {
            size_t b = sentence_buf.find_last_not_of(kTrimTrailing);
            std::string tail = sentence_buf.substr(a, b - a + 1);
            int words = 0;
            for (char c : tail)
                if (std::isalnum(static_cast<unsigned char>(c))) ++words;
            if (words >= 2) on_sentence(tail);
        }
    }

    return full_response;
}

// ── Per-call session ──────────────────────────────────────────────────────────
struct Session {
    AudioQueue audio_out;

    std::string conversation_history = "[]";
    std::mutex  history_mu;

    Vad vad;

    std::atomic_bool active{true};

    std::thread                        pipeline_thread;
    std::queue<std::vector<int16_t>>   pending_audio;
    std::mutex                         pending_mu;
    std::condition_variable            pending_cv;

    // Set by BotCall before Start().
    std::string caller;
    std::function<void(const std::string &sip_uri)> transfer_fn;

    Session() = default;
    ~Session() {
        active = false;
        pending_cv.notify_all();
        if (pipeline_thread.joinable()) pipeline_thread.join();
    }

    void Start() {
        // Play chime then the pre-synthesized greeting immediately.
        auto chime = SynthesizeChime();
        audio_out.Push(chime.data(), chime.size());
        if (!g_greeting_pcm.empty())
            audio_out.Push(g_greeting_pcm.data(), g_greeting_pcm.size());
        if (!g_greeting_text.empty())
            AppendHistory("assistant", g_greeting_text);

        pipeline_thread = std::thread([this] { RunPipeline(); });
    }

    void SubmitUtterance(std::vector<int16_t> pcm) {
        {
            std::lock_guard<std::mutex> lk(pending_mu);
            pending_audio.push(std::move(pcm));
        }
        pending_cv.notify_one();
    }

    void AppendHistory(const std::string &role, const std::string &content) {
        std::lock_guard<std::mutex> lk(history_mu);
        auto arr = nlohmann::json::parse(conversation_history, nullptr, false);
        if (!arr.is_array()) arr = nlohmann::json::array();
        arr.push_back({{"role", role}, {"content", content}});
        conversation_history = arr.dump();
    }

    // Synthesize `text` via Piper, push PCM into audio_out, and log.
    void SpeakAndLog(const std::string &text) {
        LOG(INFO) << caller << " [tts] " << text;
        auto pcm = Synthesize(text);
        if (!pcm.empty())
            audio_out.Push(pcm.data(), pcm.size());
    }

    void RunPipeline() {
        while (active) {
            std::vector<int16_t> pcm;
            {
                std::unique_lock<std::mutex> lk(pending_mu);
                pending_cv.wait(lk, [this] {
                    return !pending_audio.empty() || !active;
                });
                if (!active && pending_audio.empty()) break;
                pcm = std::move(pending_audio.front());
                pending_audio.pop();
            }

            // ── Transcribe caller utterance ───────────────────────────────
            {
                auto seg = SynthesizeSegmentChime();
                audio_out.Push(seg.data(), seg.size());
            }
            LOG(INFO) << caller << " [stt] transcribing "
                      << (pcm.size() * 1000 / kAudioRate) << " ms";
            {
                int stt_ms = static_cast<int>(pcm.size() * 1000 / kAudioRate) + 2000;
                auto beeps = SynthesizeThinkingBeep(stt_ms);
                audio_out.Push(beeps.data(), beeps.size());
            }
            std::string text = Transcribe(pcm);
            audio_out.Clear();
            if (text.empty()) {
                LOG(INFO) << caller << " [stt] empty transcript, skipping";
                continue;
            }
            LOG(INFO) << caller << " [transcript] " << text;
            AppendHistory("user", text);

            // ── Generate and speak reply ──────────────────────────────────
            std::string history;
            { std::lock_guard<std::mutex> lk(history_mu); history = conversation_history; }

            {
                auto beeps = SynthesizeThinkingBeep(30000);
                audio_out.Push(beeps.data(), beeps.size());
            }
            bool first_sentence = true;
            std::string full_reply;
            OllamaChat(text, history,
                [&](const std::string &sentence) {
                    if (first_sentence) {
                        audio_out.Clear();
                        first_sentence = false;
                    }
                    full_reply += sentence;
                    // Strip [TRANSFER:...] before speaking — it's a command, not text.
                    std::string spoken = sentence;
                    {
                        const std::string tag = "[TRANSFER:";
                        auto p = spoken.find(tag);
                        if (p != std::string::npos) {
                            auto q = spoken.find(']', p);
                            if (q != std::string::npos) spoken.erase(p, q - p + 1);
                            for (size_t i; (i = spoken.find("  ")) != std::string::npos; )
                                spoken.erase(i, 1);
                        }
                    }
                    if (!spoken.empty()) SpeakAndLog(spoken);
                }, caller);

            // Check for a transfer command anywhere in the reply.
            // Format: [TRANSFER:NNNN] where NNNN is a 4-digit extension.
            std::string transfer_target;
            {
                const std::string tag = "[TRANSFER:";
                auto p = full_reply.find(tag);
                if (p != std::string::npos) {
                    auto q = full_reply.find(']', p);
                    if (q != std::string::npos)
                        transfer_target = full_reply.substr(p + tag.size(),
                                                            q - p - tag.size());
                }
            }

            // Store spoken version (with tag removed) in history.
            std::string spoken = full_reply;
            if (!transfer_target.empty()) {
                const std::string tag = "[TRANSFER:";
                auto p = spoken.find(tag);
                auto q = spoken.find(']', p);
                if (q != std::string::npos) spoken.erase(p, q - p + 1);
                for (size_t i; (i = spoken.find("  ")) != std::string::npos; )
                    spoken.erase(i, 1);
            }
            AppendHistory("assistant", spoken);

            if (!transfer_target.empty() && transfer_fn) {
                // Validate before committing: must be exactly 4 digits.
                bool valid = transfer_target.size() == 4 &&
                             transfer_target.find_first_not_of("0123456789")
                                 == std::string::npos;
                if (!valid) {
                    LOG(WARNING) << caller << " [call] ignoring invalid transfer target: "
                                 << transfer_target;
                } else {
                    // Wait for audio_out to drain before transferring.
                    while (active && audio_out.Size() > 0)
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    LOG(INFO) << caller << " [call] transferring to ext " << transfer_target;
                    transfer_fn(transfer_target);
                    active = false;
                }
            }
        }
    }

    // Called from PJSUA2 audio thread.
    void ProcessRxFrame(const int16_t *samples, int n) {
        // Don't let incoming speech interrupt the assistant while it is talking.
        if (audio_out.Size() > 0) {
            vad.Reset();
            return;
        }

        bool was_listening = vad.in_speech;
        if (vad.ProcessFrame(samples, n)) {
            SubmitUtterance(std::move(vad.ready));
        } else if (!was_listening && vad.in_speech) {
            // VAD just detected speech onset — play a bright chime so the caller
            // knows the system has started listening.
            auto chime = SynthesizeSttListeningChime();
            audio_out.Push(chime.data(), chime.size());
        }
    }
};

// ── Custom audio media port ───────────────────────────────────────────────────
class BotAudioPort : public pj::AudioMediaPort {
public:
    explicit BotAudioPort(Session &session) : session_(session) {
        pj::MediaFormatAudio fmt;
        fmt.type          = PJMEDIA_TYPE_AUDIO;
        fmt.clockRate     = kAudioRate;
        fmt.channelCount  = 1;
        fmt.bitsPerSample = 16;
        fmt.frameTimeUsec = kFrameMs * 1000;
        createPort("bot", fmt);
    }

    void onFrameRequested(pj::MediaFrame &frame) override {
        frame.type = PJMEDIA_FRAME_TYPE_AUDIO;
        unsigned bytes = frame.size;
        if (bytes == 0) return;
        frame.buf.resize(bytes);
        auto *samples = reinterpret_cast<int16_t *>(frame.buf.data());
        int   n       = static_cast<int>(bytes / sizeof(int16_t));
        session_.audio_out.Pop(samples, n);
    }

    void onFrameReceived(pj::MediaFrame &frame) override {
        if (frame.type != PJMEDIA_FRAME_TYPE_AUDIO) return;
        const auto *samples = reinterpret_cast<const int16_t *>(frame.buf.data());
        int n = static_cast<int>(frame.size / sizeof(int16_t));
        session_.ProcessRxFrame(samples, n);
    }

private:
    Session &session_;
};

// ── SIP call ──────────────────────────────────────────────────────────────────
class BotCall : public pj::Call {
public:
    explicit BotCall(pj::Account &acc, int call_id = PJSUA_INVALID_ID)
        : pj::Call(acc, call_id) {}

    ~BotCall() override { port_.reset(); session_.reset(); }

    void onCallState(pj::OnCallStateParam & /*prm*/) override {
        pj::CallInfo ci = getInfo();
        LOG(INFO) << ci.remoteUri << " [call] " << ci.stateText;
        if (ci.state == PJSIP_INV_STATE_DISCONNECTED) {
            // Destroy the audio port on this PJSUA2 thread (safe).
            // Then hand off the Session to a detached thread: joining its
            // pipeline thread can block for the duration of an in-flight
            // Ollama/Piper request, and we must not stall the SIP event loop.
            port_.reset();
            if (session_) session_->active = false;
            auto session = std::move(session_);
            std::thread([s = std::move(session)] {}).detach();
            delete this;
        }
    }

    void onCallMediaState(pj::OnCallMediaStateParam & /*prm*/) override {
        pj::CallInfo ci = getInfo();
        for (unsigned i = 0; i < ci.media.size(); i++) {
            const auto &mi = ci.media[i];
            if (mi.type != PJMEDIA_TYPE_AUDIO) continue;
            if (mi.status == PJSUA_CALL_MEDIA_ACTIVE) {
                auto *aud = dynamic_cast<pj::AudioMedia *>(getMedia(i));
                if (!aud) continue;
                if (!session_) {
                    session_ = std::make_unique<Session>();
                    session_->caller = ci.remoteUri;
                    session_->vad.vad_threshold = g_vad_threshold;
                    session_->vad.silence_ms    = g_silence_ms;
                    session_->vad.min_speech_ms = g_min_speech_ms;
                    pjsua_call_id cid = getId();
                    session_->transfer_fn = [cid](const std::string &ext) {
                        std::string uri = "sip:" + ext
                                        + (g_pbx_host.empty() ? "" : "@" + g_pbx_host);
                        static thread_local pj_thread_desc desc;
                        static thread_local pj_thread_t   *thr = nullptr;
                        if (!thr) pj_thread_register("transfer", desc, &thr);
                        pj_str_t target = pj_str(const_cast<char *>(uri.c_str()));
                        pj_status_t st = pjsua_call_xfer(cid, &target, nullptr);
                        if (st != PJ_SUCCESS) {
                            char errbuf[80];
                            pj_strerror(st, errbuf, sizeof(errbuf));
                            LOG(ERROR) << "[call] transfer failed: " << errbuf;
                        }
                    };
                    session_->Start();
                    port_    = std::make_unique<BotAudioPort>(*session_);
                }
                try {
                    aud->startTransmit(*port_);
                    port_->startTransmit(*aud);
                    LOG(INFO) << ci.remoteUri << " [media] RTP connected";
                } catch (pj::Error &e) {
                    LOG(ERROR) << "Media link error: " << e.info();
                }
            } else if (mi.status == PJSUA_CALL_MEDIA_NONE && session_) {
                session_->active = false;
            }
        }
    }

private:
    std::unique_ptr<Session>      session_;
    std::unique_ptr<BotAudioPort> port_;
};

// ── SIP account ───────────────────────────────────────────────────────────────
class BotAccount : public pj::Account {
public:
    void onIncomingCall(pj::OnIncomingCallParam &prm) override {
        LOG(INFO) << "[call] incoming from " << prm.rdata.srcAddress;
        auto *call = new BotCall(*this, prm.callId);
        pj::CallOpParam op;
        op.statusCode = PJSIP_SC_OK;
        try {
            call->answer(op);
        } catch (pj::Error &e) {
            LOG(ERROR) << "Answer error: " << e.info();
        }
    }
};

// ── Signal handling ───────────────────────────────────────────────────────────
static std::atomic_bool g_running{true};
static void sig_handler(int) { g_running = false; }

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);
    std::signal(SIGPIPE, SIG_IGN);

    absl::SetProgramUsageMessage(
        "SIP voice chatbot — Whisper STT + Ollama LLM + Piper TTS.\n"
        "Dial sip:bot@<host> from any SIP softphone.");
    absl::ParseCommandLine(argc, argv);

    g_whisper_addr = absl::GetFlag(FLAGS_whisper);
    g_ollama_url   = absl::GetFlag(FLAGS_ollama);
    g_piper_addr   = absl::GetFlag(FLAGS_piper);
    g_pbx_host     = absl::GetFlag(FLAGS_pbx);
    int sip_port  = absl::GetFlag(FLAGS_port);
    std::string public_addr = absl::GetFlag(FLAGS_public_addr);

    static constexpr char kDefaultSystemPrompt[] =
        "You are a helpful voice assistant answering a phone call. "
        "Keep all responses short and conversational — one to three sentences at most. "
        "Never use emoji, bullet points, markdown, or lists. "
        "Speak in plain prose as if talking to someone on the phone. "
        "If the caller asks to be transferred or connected to someone, end your reply "
        "with the token [TRANSFER:NNNN] where NNNN is the 4-digit extension. "
        "For example: \"Connecting you now. [TRANSFER:1042]\" — "
        "the token is never spoken aloud, only the surrounding text is.";

    static constexpr char kDefaultGreetingPrompt[] =
        "Greet the caller warmly in one short sentence, consistent with your persona.";

    // Greeting prompt used at startup — may be overridden by config file.
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

            // VAD tuning — all three fields are optional.
            if (cfg["vad"] && cfg["vad"].IsMap()) {
                const YAML::Node &vad = cfg["vad"];
                if (vad["threshold"])  g_vad_threshold = vad["threshold"].as<double>();
                if (vad["silence_ms"]) g_silence_ms    = vad["silence_ms"].as<int>();
                if (vad["min_speech_ms"]) g_min_speech_ms = vad["min_speech_ms"].as<int>();
            }

            // Any key under the "options" mapping is forwarded to Ollama.
            // Values are passed as-is; yaml-cpp preserves numeric types.
            if (cfg["options"] && cfg["options"].IsMap()) {
                for (const auto &kv : cfg["options"]) {
                    std::string key = kv.first.as<std::string>();
                    const YAML::Node &val = kv.second;
                    if (val.IsScalar()) {
                        // Preserve numeric types for the JSON payload.
                        try { g_model_options[key] = val.as<double>(); continue; } catch (...) {}
                        try { g_model_options[key] = val.as<int>();    continue; } catch (...) {}
                        g_model_options[key] = val.as<std::string>();
                    }
                }
            }

            LOG(INFO) << "[config] loaded from " << config_file;
        } catch (const YAML::Exception &e) {
            LOG(WARNING) << "[config] failed to load " << config_file << ": " << e.what();
        }
    }

    if (g_system_prompt.empty()) {
        g_system_prompt = kDefaultSystemPrompt;
        LOG(INFO) << "[config] using default system prompt";
    } else {
        LOG(INFO) << "[config] system prompt loaded (" << g_system_prompt.size() << " chars)";
    }

    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    absl::InitializeLog();

    // Generate greeting once at startup so every call plays it instantly.
    {
        LOG(INFO) << "[startup] generating greeting";
        OllamaChat(greeting_prompt, "[]",
            [&](const std::string &sentence) {
                g_greeting_text += sentence;
                auto pcm = Synthesize(sentence);
                if (!pcm.empty()) {
                    g_greeting_pcm.insert(g_greeting_pcm.end(),
                                          pcm.begin(), pcm.end());
                } else {
                    LOG(WARNING) << "[startup] Piper returned no PCM for: " << sentence;
                }
            });
        if (g_greeting_text.empty())
            LOG(WARNING) << "[startup] greeting is empty — Ollama may be unreachable or model not loaded";
        else
            LOG(INFO) << "[startup] greeting: " << g_greeting_text;
    }

    pj::Endpoint ep;
    try {
        ep.libCreate();

        pj::EpConfig cfg;
        cfg.uaConfig.maxCalls      = g_max_calls;
        cfg.logConfig.level        = 3;
        cfg.logConfig.consoleLevel = 3;
        cfg.medConfig.clockRate    = kAudioRate;
        cfg.medConfig.noVad        = true;
        cfg.medConfig.ecTailLen    = 0;
        ep.libInit(cfg);

        pj::TransportConfig transport_cfg;
        transport_cfg.port = sip_port;
        if (!public_addr.empty())
            transport_cfg.publicAddress = public_addr;
        ep.transportCreate(PJSIP_TRANSPORT_UDP, transport_cfg);
        try { ep.transportCreate(PJSIP_TRANSPORT_UDP6, transport_cfg); } catch (...) {}

        ep.libStart();
        ep.audDevManager().setNullDev();
        ep.codecSetPriority("PCMU/8000", 255);
        ep.codecSetPriority("PCMA/8000", 254);

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
    } catch (pj::Error &e) {
        LOG(FATAL) << "Fatal: " << e.info();
        return 1;
    }
    return 0;
}
