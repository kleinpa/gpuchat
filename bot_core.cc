// bot_core.cc — SIP chatbot core implementation.
//
// WyomingAsrClient / WyomingTtsClient — Wyoming TCP protocol (STT + TTS)
// OllamaClient                        — Ollama HTTP streaming chat
// Session / BotAudioPort / BotCall / BotAccount — per-call pipeline + SIP

#include "bot_core.h"

#include "absl/log/log.h"

#define CPPHTTPLIB_SEND_FLAGS MSG_NOSIGNAL
#define CPPHTTPLIB_NO_EXCEPTIONS
#include "httplib.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
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

// ── Runtime globals ───────────────────────────────────────────────────────────
std::string g_whisper_addr;
std::string g_ollama_url;
std::string g_piper_addr;
std::string g_pbx_host;
std::string g_model = "gemma3:1b";
std::string g_system_prompt;
int g_max_calls = 8;

nlohmann::json g_model_options = {
    {"temperature", 0.2},
    {"top_k", 64},
    {"top_p", 0.95},
    {"min_p", 0.01},
    {"repeat_penalty", 1.0},
};

double g_vad_threshold = kVadThreshold;
int g_silence_ms = kSilenceMs;
int g_min_speech_ms = kMinSpeechMs;

std::vector<int16_t> g_greeting_pcm;
std::string g_greeting_text;

// ── TCP helpers ───────────────────────────────────────────────────────────────

static int ConnectTcp(const std::string& host, int port) {
  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  std::string port_str = std::to_string(port);
  if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
    return -1;
  int fd = -1;
  for (auto* p = res; p; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  return fd;
}

static bool SendAll(int fd, const char* buf, size_t len) {
  while (len > 0) {
    ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
    if (n <= 0) return false;
    buf += n;
    len -= n;
  }
  return true;
}

// ── Wyoming protocol ──────────────────────────────────────────────────────────

static std::pair<std::string, int> ParseHostPort(const std::string& addr,
                                                 int default_port) {
  auto colon = addr.rfind(':');
  if (colon == std::string::npos) return {addr, default_port};
  return {addr.substr(0, colon), std::stoi(addr.substr(colon + 1))};
}

// Sends one Wyoming framed message: header line + optional data + payload.
static bool WyomingSend(int fd, const std::string& type,
                        const std::string& data_json,
                        const void* payload = nullptr,
                        size_t payload_len = 0) {
  std::string header = "{\"type\":\"" + type + "\",\"version\":\"1.8.0\"";
  if (!data_json.empty())
    header += ",\"data_length\":" + std::to_string(data_json.size());
  if (payload && payload_len > 0)
    header += ",\"payload_length\":" + std::to_string(payload_len);
  header += "}\n";
  if (!SendAll(fd, header.c_str(), header.size())) return false;
  if (!data_json.empty() &&
      !SendAll(fd, data_json.c_str(), data_json.size()))
    return false;
  if (payload && payload_len > 0 &&
      !SendAll(fd, static_cast<const char*>(payload), payload_len))
    return false;
  return true;
}

// Receives one Wyoming framed message; returns the type string.
static std::string WyomingRecv(int fd, std::string& data_json,
                               std::vector<uint8_t>& payload) {
  data_json.clear();
  payload.clear();

  std::string header;
  char ch;
  while (recv(fd, &ch, 1, 0) == 1) {
    if (ch == '\n') break;
    header += ch;
  }
  if (header.empty()) return {};

  auto ReadBytes = [&](int socket, char* dst, size_t n) -> bool {
    size_t got = 0;
    while (got < n) {
      ssize_t r = recv(socket, dst + got, n - got, 0);
      if (r <= 0) return false;
      got += r;
    }
    return true;
  };

  auto hdr = nlohmann::json::parse(header, nullptr, false);
  if (hdr.is_discarded()) return {};

  if (hdr.contains("data_length")) {
    size_t dl = hdr["data_length"].get<size_t>();
    if (dl > 0) {
      data_json.resize(dl);
      if (!ReadBytes(fd, data_json.data(), dl)) return {};
    }
  }

  if (hdr.contains("payload_length")) {
    size_t pl = hdr["payload_length"].get<size_t>();
    if (pl > 0) {
      payload.resize(pl);
      if (!ReadBytes(fd, reinterpret_cast<char*>(payload.data()), pl))
        return {};
    }
  }

  return hdr.value("type", std::string{});
}

// ── WyomingAsrClient ─────────────────────────────────────────────────────────

class WyomingAsrClient : public AsrClient {
 public:
  explicit WyomingAsrClient(std::string addr) : addr_(std::move(addr)) {}

  std::string Transcribe(const std::vector<int16_t>& pcm) override {
    if (pcm.empty()) return {};
    auto [host, port] = ParseHostPort(addr_, 10300);
    int fd = ConnectTcp(host, port);
    if (fd < 0) {
      LOG(ERROR) << "[stt] connect failed: " << addr_;
      return {};
    }

    std::string audio_data =
        "{\"rate\":" + std::to_string(kAudioRate) +
        ",\"width\":2,\"channels\":1,\"timestamp\":null}";
    if (!WyomingSend(fd, "audio-chunk", audio_data,
                     pcm.data(), pcm.size() * sizeof(int16_t)) ||
        !WyomingSend(fd, "audio-stop", "{\"timestamp\":null}")) {
      close(fd);
      return {};
    }

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

 private:
  std::string addr_;
};

// ── WyomingTtsClient ─────────────────────────────────────────────────────────

class WyomingTtsClient : public TtsClient {
 public:
  explicit WyomingTtsClient(std::string addr) : addr_(std::move(addr)) {}

  std::vector<int16_t> Synthesize(const std::string& text) override {
    if (text.empty()) return {};
    auto [host, port] = ParseHostPort(addr_, 10200);
    int fd = ConnectTcp(host, port);
    if (fd < 0) {
      LOG(ERROR) << "[tts] connect failed: " << addr_;
      return {};
    }

    std::string synth_data =
        "{\"text\":\"" + JsonEscape(text) + "\",\"channels\":1}";
    if (!WyomingSend(fd, "synthesize", synth_data)) {
      close(fd);
      return {};
    }

    std::vector<int16_t> pcm;
    int piper_rate = kAudioRate;
    for (int i = 0; i < 4096; i++) {
      std::string data_json;
      std::vector<uint8_t> payload;
      std::string type = WyomingRecv(fd, data_json, payload);
      if (type.empty()) break;
      if (type == "audio-start") {
        auto j = nlohmann::json::parse(data_json, nullptr, false);
        if (!j.is_discarded() && j.contains("rate"))
          piper_rate = j["rate"].get<int>();
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

    if (pcm.empty()) {
      LOG(WARNING) << "[tts] empty response for: " << text;
      return {};
    }

    // Downsample if piper uses a different rate (e.g. 22050 → 16000).
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

 private:
  std::string addr_;
};

// ── OllamaClient ─────────────────────────────────────────────────────────────

class OllamaClient : public LlmClient {
 public:
  OllamaClient(std::string base_url, std::string model,
               std::string system_prompt, nlohmann::json options)
      : base_url_(std::move(base_url)),
        model_(std::move(model)),
        system_prompt_(std::move(system_prompt)),
        options_(std::move(options)) {}

  std::string Chat(const std::string& user_text,
                   const nlohmann::json& history,
                   std::function<void(const std::string&)> on_sentence,
                   const std::string& caller) override {
    nlohmann::json messages = nlohmann::json::array();
    if (!system_prompt_.empty())
      messages.push_back({{"role", "system"}, {"content", system_prompt_}});
    for (const auto& m : history) messages.push_back(m);
    messages.push_back({{"role", "user"}, {"content", user_text}});

    std::string req_body = nlohmann::json{
        {"model", model_},
        {"messages", messages},
        {"options", options_},
        {"stream", true},
    }.dump();

    LOG(INFO) << (caller.empty() ? "" : caller + " ")
              << "[llm] " << model_ << " ← " << user_text;

    std::string full_response;
    std::string sentence_buf;
    bool in_think = false;
    std::string think_tail;

    auto url = ParseUrl(base_url_);
    httplib::Client cli(url.host, url.port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(120);

    httplib::Request req;
    req.method = "POST";
    req.path = "/api/chat";
    req.headers = {{"Content-Type", "application/json"}};
    req.body = req_body;
    req.content_receiver = [&](const char* data, size_t len,
                               uint64_t, uint64_t) -> bool {
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
            if (end == std::string::npos) { think_tail.clear(); break; }
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
        for (const auto& s : SplitSentences(sentence_buf)) on_sentence(s);
      }
      return true;
    };

    httplib::Response res;
    httplib::Error err = httplib::Error::Success;
    if (!cli.send(req, res, err)) {
      LOG(ERROR) << "[llm] POST failed: " << httplib::to_string(err);
    } else if (res.status / 100 != 2) {
      LOG(ERROR) << "[llm] POST status=" << res.status;
    }

    // Flush trailing fragment without terminal punctuation.
    static constexpr char kTrimLeading[] = " \t\r\n\"'`";
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

    return full_response;
  }

 private:
  std::string base_url_;
  std::string model_;
  std::string system_prompt_;
  nlohmann::json options_;
};

// ── Factory functions ─────────────────────────────────────────────────────────

std::unique_ptr<AsrClient> MakeWyomingAsrClient() {
  return std::make_unique<WyomingAsrClient>(g_whisper_addr);
}

std::unique_ptr<TtsClient> MakeWyomingTtsClient() {
  return std::make_unique<WyomingTtsClient>(g_piper_addr);
}

std::unique_ptr<LlmClient> MakeOllamaClient() {
  return std::make_unique<OllamaClient>(
      g_ollama_url, g_model, g_system_prompt, g_model_options);
}

// ── In-band audio cues ────────────────────────────────────────────────────────

// Generates a single decaying sine tone.
static std::vector<int16_t> SineChime(float freq, float duration_ms,
                                      float decay, float gain) {
  int n = static_cast<int>(kAudioRate * duration_ms / 1000.f);
  std::vector<int16_t> out;
  out.reserve(n);
  for (int i = 0; i < n; i++) {
    float t = static_cast<float>(i) / kAudioRate;
    float sample = gain * std::exp(-decay * t) * std::sin(2.f * M_PI * freq * t);
    out.push_back(static_cast<int16_t>(sample * 32767.f));
  }
  return out;
}

// Dual-tone answer chime played when a call connects.
static std::vector<int16_t> AnswerChime() {
  struct Note { float freq; float duration_ms; };
  constexpr Note kNotes[] = {{659.3f, 120.f}, {880.0f, 280.f}};
  std::vector<int16_t> out;
  for (auto [freq, dur] : kNotes) {
    auto tone = SineChime(freq, dur, /*decay=*/6.f, /*gain=*/0.35f);
    out.insert(out.end(), tone.begin(), tone.end());
  }
  return out;
}

// Short bright ping played when STT starts listening.
static std::vector<int16_t> ListeningChime() {
  return SineChime(/*freq=*/1047.f, /*duration_ms=*/80.f,
                   /*decay=*/10.f, /*gain=*/0.30f);
}

// Soft low tick played at the start of each transcription segment.
static std::vector<int16_t> SegmentChime() {
  return SineChime(/*freq=*/330.f, /*duration_ms=*/60.f,
                   /*decay=*/20.f, /*gain=*/0.20f);
}

// Repeating low beep played while the bot is thinking (STT or LLM).
static std::vector<int16_t> ThinkingBeep(int duration_ms,
                                         int interval_ms = 600) {
  constexpr float kFreq = 220.0f;
  constexpr float kBeepMs = 80.f;
  constexpr float kDecay = 30.0f;
  constexpr float kGain = 0.08f;

  int total = kAudioRate * duration_ms / 1000;
  int spacing = kAudioRate * interval_ms / 1000;
  int beep_n = static_cast<int>(kAudioRate * kBeepMs / 1000.f);

  std::vector<int16_t> out(total, 0);
  for (int start = 0; start < total; start += spacing) {
    for (int i = 0; i < beep_n && start + i < total; i++) {
      float t = static_cast<float>(i) / kAudioRate;
      float sample = kGain * std::exp(-kDecay * t) *
                     std::sin(2.f * M_PI * kFreq * t);
      out[start + i] = static_cast<int16_t>(sample * 32767.f);
    }
  }
  return out;
}

// ── Transfer-tag helpers ──────────────────────────────────────────────────────

static constexpr char kTransferTag[] = "[TRANSFER:";

// Removes the first [TRANSFER:NNNN] tag from `text` in-place and returns the
// 4-digit extension, or empty string if no valid tag is present.
static std::string StripTransferTag(std::string& text) {
  auto p = text.find(kTransferTag);
  if (p == std::string::npos) return {};
  auto q = text.find(']', p);
  if (q == std::string::npos) return {};
  std::string ext = text.substr(p + sizeof(kTransferTag) - 1,
                                q - p - (sizeof(kTransferTag) - 1));
  text.erase(p, q - p + 1);
  // Collapse any double-space left behind.
  for (size_t i; (i = text.find("  ")) != std::string::npos;)
    text.erase(i, 1);
  return ext;
}

// ── Session ───────────────────────────────────────────────────────────────────

Session::Session(std::string caller,
                 VadParams vad_params,
                 std::function<void(const std::string&)> transfer_fn,
                 std::unique_ptr<AsrClient> asr,
                 std::unique_ptr<TtsClient> tts,
                 std::unique_ptr<LlmClient> llm)
    : caller_(std::move(caller)),
      transfer_fn_(std::move(transfer_fn)),
      asr_(std::move(asr)),
      tts_(std::move(tts)),
      llm_(std::move(llm)) {
  vad_.vad_threshold = vad_params.threshold;
  vad_.silence_ms = vad_params.silence_ms;
  vad_.min_speech_ms = vad_params.min_speech_ms;
}

Session::~Session() {
  active = false;
  pending_cv_.notify_all();
  if (pipeline_thread_.joinable()) pipeline_thread_.join();
}

void Session::Start() {
  auto chime = AnswerChime();
  audio_out.Push(chime.data(), chime.size());
  if (!g_greeting_pcm.empty())
    audio_out.Push(g_greeting_pcm.data(), g_greeting_pcm.size());
  {
    auto listen = ListeningChime();
    audio_out.Push(listen.data(), listen.size());
  }
  if (!g_greeting_text.empty()) AppendHistory("assistant", g_greeting_text);

  pipeline_thread_ = std::thread([this] { RunPipeline(); });
}

void Session::SubmitUtterance(std::vector<int16_t> pcm) {
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    pending_audio_.push(std::move(pcm));
  }
  pending_cv_.notify_one();
}

void Session::AppendHistory(const std::string& role,
                            const std::string& content) {
  std::lock_guard<std::mutex> lk(history_mu_);
  history_.push_back({{"role", role}, {"content", content}});
}

void Session::SpeakAndLog(const std::string& text) {
  LOG(INFO) << caller_ << " [tts] " << text;
  auto pcm = tts_->Synthesize(text);
  if (!pcm.empty()) audio_out.Push(pcm.data(), pcm.size());
}

void Session::RunPipeline() {
  while (active) {
    std::vector<int16_t> pcm;
    {
      std::unique_lock<std::mutex> lk(pending_mu_);
      pending_cv_.wait(lk, [this] {
        return !pending_audio_.empty() || !active;
      });
      if (!active && pending_audio_.empty()) break;
      pcm = std::move(pending_audio_.front());
      pending_audio_.pop();
    }

    // ── Transcribe caller utterance ───────────────────────────────────
    {
      auto seg = SegmentChime();
      audio_out.Push(seg.data(), seg.size());
    }
    LOG(INFO) << caller_ << " [stt] transcribing "
              << (pcm.size() * 1000 / kAudioRate) << " ms";
    {
      int stt_ms = static_cast<int>(pcm.size() * 1000 / kAudioRate) + 2000;
      auto beeps = ThinkingBeep(stt_ms);
      audio_out.Push(beeps.data(), beeps.size());
    }
    std::string text = asr_->Transcribe(pcm);
    audio_out.Clear();
    if (text.empty()) {
      LOG(INFO) << caller_ << " [stt] empty transcript, skipping";
      continue;
    }
    LOG(INFO) << caller_ << " [transcript] " << text;
    AppendHistory("user", text);

    // ── Generate and speak reply ──────────────────────────────────────
    nlohmann::json history_snapshot;
    {
      std::lock_guard<std::mutex> lk(history_mu_);
      history_snapshot = history_;
    }

    {
      auto beeps = ThinkingBeep(30000);
      audio_out.Push(beeps.data(), beeps.size());
    }
    bool first_sentence = true;
    std::string full_reply;
    llm_->Chat(text, history_snapshot,
        [&](const std::string& sentence) {
          if (first_sentence) {
            audio_out.Clear();
            first_sentence = false;
          }
          full_reply += sentence;
          std::string spoken = sentence;
          StripTransferTag(spoken);
          if (!spoken.empty()) SpeakAndLog(spoken);
        },
        caller_);

    // Extract and strip transfer tag from the full reply.
    std::string spoken_reply = full_reply;
    std::string transfer_ext = StripTransferTag(spoken_reply);
    AppendHistory("assistant", spoken_reply);

    if (!transfer_ext.empty() && transfer_fn_) {
      bool valid = transfer_ext.size() == 4 &&
                   transfer_ext.find_first_not_of("0123456789") ==
                       std::string::npos;
      if (!valid) {
        LOG(WARNING) << caller_ << " [call] ignoring invalid transfer target: "
                     << transfer_ext;
      } else {
        while (active && audio_out.Size() > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        LOG(INFO) << caller_ << " [call] transferring to ext " << transfer_ext;
        transfer_fn_(transfer_ext);
        active = false;
      }
    } else if (active) {
      auto listen = ListeningChime();
      audio_out.Push(listen.data(), listen.size());
    }
  }
}

void Session::ProcessRxFrame(const int16_t* samples, int n) {
  if (audio_out.Size() > 0) {
    vad_.Reset();
    return;
  }
  if (vad_.ProcessFrame(samples, n)) SubmitUtterance(std::move(vad_.ready));
}

// ── BotAudioPort ─────────────────────────────────────────────────────────────

BotAudioPort::BotAudioPort(Session& session) : session_(session) {
  pj::MediaFormatAudio fmt;
  fmt.type = PJMEDIA_TYPE_AUDIO;
  fmt.clockRate = kAudioRate;
  fmt.channelCount = 1;
  fmt.bitsPerSample = 16;
  fmt.frameTimeUsec = kFrameMs * 1000;
  createPort("bot", fmt);
}

void BotAudioPort::onFrameRequested(pj::MediaFrame& frame) {
  frame.type = PJMEDIA_FRAME_TYPE_AUDIO;
  unsigned bytes = frame.size;
  if (bytes == 0) return;
  frame.buf.resize(bytes);
  auto* samples = reinterpret_cast<int16_t*>(frame.buf.data());
  int n = static_cast<int>(bytes / sizeof(int16_t));
  session_.audio_out.Pop(samples, n);
}

void BotAudioPort::onFrameReceived(pj::MediaFrame& frame) {
  if (frame.type != PJMEDIA_FRAME_TYPE_AUDIO) return;
  const auto* samples = reinterpret_cast<const int16_t*>(frame.buf.data());
  int n = static_cast<int>(frame.size / sizeof(int16_t));
  session_.ProcessRxFrame(samples, n);
}

// ── BotCall ───────────────────────────────────────────────────────────────────

BotCall::BotCall(pj::Account& acc, int call_id) : pj::Call(acc, call_id) {}

BotCall::~BotCall() {
  port_.reset();
  session_.reset();
}

void BotCall::onCallState(pj::OnCallStateParam&) {
  pj::CallInfo ci = getInfo();
  LOG(INFO) << ci.remoteUri << " [call] " << ci.stateText;
  if (ci.state == PJSIP_INV_STATE_DISCONNECTED) {
    port_.reset();
    if (session_) session_->active = false;
    auto session = std::move(session_);
    std::thread([s = std::move(session)] {}).detach();
    delete this;
  }
}

void BotCall::onCallMediaState(pj::OnCallMediaStateParam&) {
  pj::CallInfo ci = getInfo();
  for (unsigned i = 0; i < ci.media.size(); i++) {
    const auto& mi = ci.media[i];
    if (mi.type != PJMEDIA_TYPE_AUDIO) continue;
    if (mi.status == PJSUA_CALL_MEDIA_ACTIVE) {
      auto* aud = dynamic_cast<pj::AudioMedia*>(getMedia(i));
      if (!aud) continue;
      if (!session_) {
        pjsua_call_id cid = getId();
        auto transfer_fn = [cid](const std::string& ext) {
          std::string uri =
              "sip:" + ext + (g_pbx_host.empty() ? "" : "@" + g_pbx_host);
          static thread_local pj_thread_desc desc;
          static thread_local pj_thread_t* thr = nullptr;
          if (!thr) pj_thread_register("transfer", desc, &thr);
          pj_str_t target = pj_str(const_cast<char*>(uri.c_str()));
          pj_status_t st = pjsua_call_xfer(cid, &target, nullptr);
          if (st != PJ_SUCCESS) {
            char errbuf[80];
            pj_strerror(st, errbuf, sizeof(errbuf));
            LOG(ERROR) << "[call] transfer failed: " << errbuf;
          }
        };
        session_ = std::make_unique<Session>(
            ci.remoteUri,
            VadParams{g_vad_threshold, g_silence_ms, g_min_speech_ms},
            std::move(transfer_fn),
            MakeWyomingAsrClient(),
            MakeWyomingTtsClient(),
            MakeOllamaClient());
        session_->Start();
        port_ = std::make_unique<BotAudioPort>(*session_);
      }
      try {
        aud->startTransmit(*port_);
        port_->startTransmit(*aud);
        LOG(INFO) << ci.remoteUri << " [media] RTP connected";
      } catch (pj::Error& e) {
        LOG(ERROR) << "Media link error: " << e.info();
      }
    } else if (mi.status == PJSUA_CALL_MEDIA_NONE && session_) {
      session_->active = false;
    }
  }
}

// ── BotAccount ───────────────────────────────────────────────────────────────

void BotAccount::onIncomingCall(pj::OnIncomingCallParam& prm) {
  LOG(INFO) << "[call] incoming from " << prm.rdata.srcAddress;
  auto* call = new BotCall(*this, prm.callId);
  pj::CallOpParam op;
  op.statusCode = PJSIP_SC_OK;
  try {
    call->answer(op);
  } catch (pj::Error& e) {
    LOG(ERROR) << "Answer error: " << e.info();
  }
}
