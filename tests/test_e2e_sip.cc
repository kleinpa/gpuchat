// test_e2e_sip.cc — end-to-end pipeline tests for sloperator
//
// PipelineTest suite:
//   Tests the full per-call pipeline (VAD → STT → LLM → TTS → audio out)
//   using dependency injection.  Fake clients are injected directly into
//   Session — no network, no subprocess, no PJSIP required.
//
// SipAnswerTest suite:
//   Verifies that sloperator answers an incoming SIP call and connects media.
//   Requires a running sloperator binary; gated by SLOPERATOR_BIN env var
//   (or $TEST_SRCDIR/_main/sloperator from Bazel runfiles).  Skipped if unset.

#include "bot_core.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// ── Audio helpers ─────────────────────────────────────────────────────────────

static std::vector<int16_t> MakeSine(int n, float freq = 440.f,
                                     float amp = 8000.f) {
  std::vector<int16_t> out(n);
  for (int i = 0; i < n; i++) {
    float t = static_cast<float>(i) / kAudioRate;
    out[i] = static_cast<int16_t>(amp * std::sin(2.f * M_PI * freq * t));
  }
  return out;
}

static std::vector<int16_t> SilenceFrame() {
  return std::vector<int16_t>(kFrameSamples, 0);
}

// ── Fake clients ─────────────────────────────────────────────────────────────

class FakeAsrClient : public AsrClient {
 public:
  std::atomic_bool called{false};
  std::string reply = "hello bot";

  std::string Transcribe(const std::vector<int16_t>&) override {
    called = true;
    return reply;
  }
};

class FakeTtsClient : public TtsClient {
 public:
  std::atomic_bool called{false};

  std::vector<int16_t> Synthesize(const std::string&) override {
    called = true;
    return MakeSine(kAudioRate / 5);  // 200 ms
  }
};

class FakeLlmClient : public LlmClient {
 public:
  std::atomic_bool called{false};
  std::string reply = "Hello there.";

  std::string Chat(const std::string&, const nlohmann::json&,
                   std::function<void(const std::string&)> on_sentence,
                   const std::string&) override {
    called = true;
    on_sentence(reply);
    return reply;
  }
};

// ── Test helpers ──────────────────────────────────────────────────────────────

static void FeedVoiceThenSilence(Session& sess, int voiced_frames,
                                 int silence_frames) {
  auto voiced = MakeSine(kFrameSamples);
  auto silence = SilenceFrame();
  for (int i = 0; i < voiced_frames; i++)
    sess.ProcessRxFrame(voiced.data(), kFrameSamples);
  for (int i = 0; i < silence_frames; i++)
    sess.ProcessRxFrame(silence.data(), kFrameSamples);
}

static bool WaitFor(std::function<bool()> cond, int timeout_ms = 5000) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (cond()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return cond();
}

// ═══════════════════════════════════════════════════════════════════════════════
// PipelineTest — exercise Session directly without SIP or network I/O
// ═══════════════════════════════════════════════════════════════════════════════

TEST(PipelineTest, FullRoundTrip) {
  g_greeting_pcm.clear();
  g_greeting_text.clear();
  g_vad_threshold = 500.0;
  g_silence_ms = 200;
  g_min_speech_ms = 100;

  auto* asr = new FakeAsrClient();
  auto* tts = new FakeTtsClient();
  auto* llm = new FakeLlmClient();
  auto* asr_ptr = asr;
  auto* tts_ptr = tts;
  auto* llm_ptr = llm;

  Session sess(
      "sip:test@127.0.0.1",
      VadParams{g_vad_threshold, g_silence_ms, g_min_speech_ms},
      [](const std::string&) {},
      std::unique_ptr<AsrClient>(asr),
      std::unique_ptr<TtsClient>(tts),
      std::unique_ptr<LlmClient>(llm));
  sess.Start();

  // Start() pushes the answer chime into audio_out.  ProcessRxFrame discards
  // voice while audio_out is non-empty, so drain it first.
  sess.audio_out.Clear();

  // 5 frames of voice (100 ms) then 10 frames of silence (200 ms).
  FeedVoiceThenSilence(sess, 5, 10);

  ASSERT_TRUE(WaitFor([&] { return llm_ptr->called.load(); }, 8000))
      << "chat was never called";

  EXPECT_TRUE(asr_ptr->called) << "transcribe was never called";
  EXPECT_TRUE(tts_ptr->called) << "synthesize was never called";
  EXPECT_GT(sess.audio_out.Size(), 0u) << "no audio in output queue";

  sess.active = false;
}

TEST(PipelineTest, EmptyTranscriptSkipped) {
  g_greeting_pcm.clear();
  g_greeting_text.clear();
  g_vad_threshold = 500.0;
  g_silence_ms = 200;
  g_min_speech_ms = 100;

  auto* asr = new FakeAsrClient();
  asr->reply = "";  // whisper returns nothing
  auto* llm = new FakeLlmClient();
  auto* asr_ptr = asr;
  auto* llm_ptr = llm;

  Session sess(
      "sip:test@127.0.0.1",
      VadParams{g_vad_threshold, g_silence_ms, g_min_speech_ms},
      [](const std::string&) {},
      std::unique_ptr<AsrClient>(asr),
      std::make_unique<FakeTtsClient>(),
      std::unique_ptr<LlmClient>(llm));
  sess.Start();

  // Drain the answer chime so ProcessRxFrame doesn't discard voice frames.
  sess.audio_out.Clear();

  FeedVoiceThenSilence(sess, 5, 10);

  ASSERT_TRUE(WaitFor([&] { return asr_ptr->called.load(); }, 8000))
      << "transcribe was never called";

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(llm_ptr->called) << "chat was called despite empty transcript";

  sess.active = false;
}

TEST(PipelineTest, SpeechDuringPlaybackIgnored) {
  g_greeting_pcm.clear();
  g_greeting_text.clear();
  g_vad_threshold = 500.0;
  g_silence_ms = 200;
  g_min_speech_ms = 100;

  auto* asr = new FakeAsrClient();
  auto* asr_ptr = asr;

  Session sess(
      "sip:test@127.0.0.1",
      VadParams{g_vad_threshold, g_silence_ms, g_min_speech_ms},
      [](const std::string&) {},
      std::unique_ptr<AsrClient>(asr),
      std::make_unique<FakeTtsClient>(),
      std::make_unique<FakeLlmClient>());
  sess.Start();

  // Pre-fill audio_out so the bot appears to be mid-speech.
  auto filler = MakeSine(kAudioRate * 2);
  sess.audio_out.Push(filler.data(), filler.size());

  // Push voiced frames — should be discarded while audio_out is non-empty.
  auto voiced = MakeSine(kFrameSamples);
  for (int i = 0; i < 20; i++) sess.ProcessRxFrame(voiced.data(), kFrameSamples);

  sess.audio_out.Clear();

  // Silence after — should not trigger pipeline since VAD was reset.
  auto silence = SilenceFrame();
  for (int i = 0; i < 10; i++)
    sess.ProcessRxFrame(silence.data(), kFrameSamples);

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_FALSE(asr_ptr->called)
      << "transcribe was called even though speech arrived during playback";

  sess.active = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SipAnswerTest — verify sloperator answers calls (requires binary)
// ═══════════════════════════════════════════════════════════════════════════════

static int ListenTcp(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return -1;
  }
  if (listen(fd, 4) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

static std::string ReadLine(int fd) {
  std::string line;
  char ch;
  while (recv(fd, &ch, 1, 0) == 1) {
    if (ch == '\n') break;
    line += ch;
  }
  return line;
}

static bool RecvAllBytes(int fd, void* buf, size_t n) {
  auto* p = static_cast<char*>(buf);
  while (n > 0) {
    ssize_t r = recv(fd, p, n, 0);
    if (r <= 0) return false;
    p += r;
    n -= r;
  }
  return true;
}

static bool SendAllBytes(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const char*>(buf);
  while (n > 0) {
    ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
    if (r <= 0) return false;
    p += r;
    n -= r;
  }
  return true;
}

static size_t ParseJsonInt(const std::string& s, const std::string& key) {
  auto pos = s.find("\"" + key + "\":");
  if (pos == std::string::npos) return 0;
  pos += key.size() + 3;
  return static_cast<size_t>(std::stoul(s.substr(pos)));
}

static void WyomingSendMsg(int fd, const std::string& type,
                           const std::string& data_json,
                           const void* payload = nullptr,
                           size_t payload_len = 0) {
  std::string hdr = "{\"type\":\"" + type + "\",\"version\":\"1.8.0\"";
  if (!data_json.empty())
    hdr += ",\"data_length\":" + std::to_string(data_json.size());
  if (payload && payload_len > 0)
    hdr += ",\"payload_length\":" + std::to_string(payload_len);
  hdr += "}\n";
  SendAllBytes(fd, hdr.c_str(), hdr.size());
  if (!data_json.empty())
    SendAllBytes(fd, data_json.c_str(), data_json.size());
  if (payload && payload_len > 0) SendAllBytes(fd, payload, payload_len);
}

static std::string WyomingRecvType(int fd) {
  std::string hdr = ReadLine(fd);
  if (hdr.empty()) return {};
  std::string type = JsonGetString(hdr, "type");
  size_t dl = ParseJsonInt(hdr, "data_length");
  size_t pl = ParseJsonInt(hdr, "payload_length");
  if (dl > 0) {
    std::vector<char> buf(dl);
    RecvAllBytes(fd, buf.data(), dl);
  }
  if (pl > 0) {
    std::vector<char> buf(pl);
    RecvAllBytes(fd, buf.data(), pl);
  }
  return type;
}

// Probe a UDP SIP port by sending a minimal OPTIONS request and waiting for
// any response.  SIP runs over UDP, so TCP connect() always fails.
static bool WaitForSipUdp(int port, int timeout_ms) {
  static const char kOptions[] =
      "OPTIONS sip:bot@127.0.0.1 SIP/2.0\r\n"
      "Via: SIP/2.0/UDP 127.0.0.1:15099;branch=z9hG4bKprobe\r\n"
      "From: <sip:probe@127.0.0.1>;tag=probe\r\n"
      "To: <sip:bot@127.0.0.1>\r\n"
      "Call-ID: probe@127.0.0.1\r\n"
      "CSeq: 1 OPTIONS\r\n"
      "Content-Length: 0\r\n\r\n";

  sockaddr_in dest{};
  dest.sin_family = AF_INET;
  dest.sin_port = htons(static_cast<uint16_t>(port));
  dest.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    // Bind to a fixed port so replies come back to us.
    sockaddr_in src{};
    src.sin_family = AF_INET;
    src.sin_port = htons(15099);
    src.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    bind(fd, reinterpret_cast<sockaddr*>(&src), sizeof(src));

    sendto(fd, kOptions, sizeof(kOptions) - 1, 0,
           reinterpret_cast<sockaddr*>(&dest), sizeof(dest));

    // Wait up to 200 ms for a reply.
    struct timeval tv{0, 200000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    char buf[256];
    ssize_t n = recv(fd, buf, sizeof(buf), 0);
    close(fd);
    if (n > 0) return true;
  }
  return false;
}

struct FakeOllamaServer {
  int fd{-1};
  std::thread thread;
  std::atomic_bool received{false};

  bool Start(int port) {
    fd = ListenTcp(port);
    if (fd < 0) return false;
    thread = std::thread([this] {
      for (int i = 0; i < 8; i++) {
        int cfd = accept(fd, nullptr, nullptr);
        if (cfd < 0) return;
        std::string req;
        while (true) {
          std::string line = ReadLine(cfd);
          if (line.empty() || line == "\r") break;
          req += line + "\n";
        }
        size_t cl = 0;
        {
          auto pos = req.find("Content-Length:");
          if (pos == std::string::npos) pos = req.find("content-length:");
          if (pos != std::string::npos) cl = std::stoul(req.substr(pos + 15));
        }
        if (cl > 0) {
          std::vector<char> body(cl);
          RecvAllBytes(cfd, body.data(), cl);
        }
        received = true;
        static const char* chunks[] = {
            "{\"model\":\"test\",\"message\":{\"role\":\"assistant\","
            "\"content\":\"Hi.\"},\"done\":false}\n",
            "{\"model\":\"test\",\"message\":{\"role\":\"assistant\","
            "\"content\":\"\"},\"done\":true}\n",
        };
        static const char kHdr[] =
            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
            "Transfer-Encoding: chunked\r\n\r\n";
        SendAllBytes(cfd, kHdr, sizeof(kHdr) - 1);
        for (const char* c : chunks) {
          size_t len = strlen(c);
          char hex[16];
          snprintf(hex, sizeof(hex), "%zx\r\n", len);
          SendAllBytes(cfd, hex, strlen(hex));
          SendAllBytes(cfd, c, len);
          SendAllBytes(cfd, "\r\n", 2);
        }
        const char* end = "0\r\n\r\n";
        SendAllBytes(cfd, end, strlen(end));
        close(cfd);
      }
    });
    return true;
  }
  void Stop() {
    shutdown(fd, SHUT_RDWR);
    close(fd);
    if (thread.joinable()) thread.join();
  }
};

struct FakePiperServer {
  int fd{-1};
  std::thread thread;
  std::atomic_bool received{false};

  bool Start(int port) {
    fd = ListenTcp(port);
    if (fd < 0) return false;
    thread = std::thread([this] {
      for (int i = 0; i < 8; i++) {
        int cfd = accept(fd, nullptr, nullptr);
        if (cfd < 0) return;
        if (WyomingRecvType(cfd) == "synthesize") {
          received = true;
          std::string start_data = "{\"rate\":" + std::to_string(kAudioRate) +
                                   ",\"width\":2,\"channels\":1}";
          WyomingSendMsg(cfd, "audio-start", start_data);
          std::vector<int16_t> pcm(kAudioRate / 5, 4000);
          std::string chunk_data =
              "{\"rate\":" + std::to_string(kAudioRate) +
              ",\"width\":2,\"channels\":1,\"timestamp\":null}";
          WyomingSendMsg(cfd, "audio-chunk", chunk_data, pcm.data(),
                         pcm.size() * sizeof(int16_t));
          WyomingSendMsg(cfd, "audio-stop", "{\"timestamp\":null}");
        }
        close(cfd);
      }
    });
    return true;
  }
  void Stop() {
    shutdown(fd, SHUT_RDWR);
    close(fd);
    if (thread.joinable()) thread.join();
  }
};

struct FakeWhisperServer {
  int fd{-1};
  std::thread thread;
  std::atomic_bool received{false};

  bool Start(int port) {
    fd = ListenTcp(port);
    if (fd < 0) return false;
    thread = std::thread([this] {
      for (int i = 0; i < 8; i++) {
        int cfd = accept(fd, nullptr, nullptr);
        if (cfd < 0) return;
        std::string t1 = WyomingRecvType(cfd);
        std::string t2 = WyomingRecvType(cfd);
        if (t1 == "audio-chunk" && t2 == "audio-stop") {
          received = true;
          WyomingSendMsg(cfd, "transcript", "{\"text\":\"hello\"}");
        }
        close(cfd);
      }
    });
    return true;
  }
  void Stop() {
    shutdown(fd, SHUT_RDWR);
    close(fd);
    if (thread.joinable()) thread.join();
  }
};

static std::atomic_bool g_call_answered{false};

class CallerCall : public pj::Call {
 public:
  explicit CallerCall(pj::Account& acc) : pj::Call(acc) {}
  void onCallState(pj::OnCallStateParam&) override {
    if (getInfo().state == PJSIP_INV_STATE_CONFIRMED) g_call_answered = true;
  }
  void onCallMediaState(pj::OnCallMediaStateParam&) override {}
};

class CallerAccount : public pj::Account {
 public:
  void onIncomingCall(pj::OnIncomingCallParam&) override {}
};

class SipAnswerTest : public ::testing::Test {
 protected:
  static constexpr int kSipPort = 15060;
  static constexpr int kOllamaPort = 19434;
  static constexpr int kPiperPort = 19200;
  static constexpr int kWhisperPort = 19300;

  FakeOllamaServer ollama_;
  FakePiperServer piper_;
  FakeWhisperServer whisper_;

  pid_t child_pid_{-1};
  pj::Endpoint ep_;
  CallerAccount* account_{nullptr};
  bool ep_created_{false};

  static std::string FindBinary() {
    if (const char* e = getenv("SLOPERATOR_BIN"); e && *e) return e;
    if (const char* e = getenv("TEST_SRCDIR"); e && *e)
      return std::string(e) + "/_main/sloperator";
    if (const char* e = getenv("RUNFILES_DIR"); e && *e)
      return std::string(e) + "/_main/sloperator";
    return "";
  }

  void SetUp() override {
    g_call_answered = false;

    std::string bin = FindBinary();
    if (bin.empty() || access(bin.c_str(), X_OK) != 0) {
      GTEST_SKIP() << "sloperator binary not found or not executable; "
                      "set SLOPERATOR_BIN to run SIP tests (tried: "
                   << (bin.empty() ? "<none>" : bin) << ")";
    }

    ASSERT_TRUE(ollama_.Start(kOllamaPort)) << "fake ollama bind failed";
    ASSERT_TRUE(piper_.Start(kPiperPort)) << "fake piper bind failed";
    ASSERT_TRUE(whisper_.Start(kWhisperPort)) << "fake whisper bind failed";

    char config_path[] = "/tmp/sloperator_sip_test_XXXXXX";
    int tmpfd = mkstemp(config_path);
    ASSERT_GE(tmpfd, 0) << "mkstemp failed";
    std::string yaml =
        "system_prompt: \"Test bot.\"\n"
        "greeting_prompt: \"Say hi.\"\n"
        "vad:\n  threshold: 100\n  silence_ms: 200\n  min_speech_ms: 100\n";
    write(tmpfd, yaml.c_str(), yaml.size());
    close(tmpfd);

    child_pid_ = fork();
    ASSERT_GE(child_pid_, 0) << "fork failed";
    if (child_pid_ == 0) {
      int devnull = ::open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, 1);
        dup2(devnull, 2);
        close(devnull);
      }
      std::string ollama_url =
          "http://127.0.0.1:" + std::to_string(kOllamaPort);
      std::string piper_addr = "127.0.0.1:" + std::to_string(kPiperPort);
      std::string wh_addr = "127.0.0.1:" + std::to_string(kWhisperPort);
      std::string port_str = std::to_string(kSipPort);
      execl(bin.c_str(), bin.c_str(),
            "--port", port_str.c_str(),
            "--ollama", ollama_url.c_str(),
            "--piper", piper_addr.c_str(),
            "--whisper", wh_addr.c_str(),
            "--config_file", config_path,
            nullptr);
      _exit(127);
    }

    ASSERT_TRUE(WaitForSipUdp(kSipPort, 15000))
        << "sloperator SIP port " << kSipPort
        << " never responded to SIP OPTIONS";

    ep_.libCreate();
    ep_created_ = true;

    pj::EpConfig cfg;
    cfg.logConfig.level = 0;
    cfg.logConfig.consoleLevel = 0;
    cfg.medConfig.clockRate = kAudioRate;
    cfg.medConfig.sndClockRate = kAudioRate;
    cfg.medConfig.noVad = true;
    cfg.medConfig.ecTailLen = 0;
    ep_.libInit(cfg);

    pj::TransportConfig tcfg;
    tcfg.port = kSipPort + 1;
    ep_.transportCreate(PJSIP_TRANSPORT_UDP, tcfg);
    ep_.libStart();
    ep_.audDevManager().setNullDev();
    ep_.codecSetPriority("G722/16000", 255);
    ep_.codecSetPriority("PCMU/8000", 0);
    ep_.codecSetPriority("PCMA/8000", 0);

    pj::AccountConfig acfg;
    acfg.idUri = "sip:caller@127.0.0.1";
    account_ = new CallerAccount();
    account_->create(acfg);
  }

  void TearDown() override {
    try {
      pjsua_call_id ids[PJSUA_MAX_CALLS];
      unsigned count = PJ_ARRAY_SIZE(ids);
      if (pjsua_enum_calls(ids, &count) == PJ_SUCCESS)
        for (unsigned i = 0; i < count; i++)
          pjsua_call_hangup(ids[i], PJSIP_SC_DECLINE, nullptr, nullptr);
    } catch (...) {}

    if (ep_created_) {
      try {
        ep_.libDestroy();
      } catch (...) {}
      ep_created_ = false;
    }

    if (child_pid_ > 0) {
      kill(child_pid_, SIGTERM);
      int st;
      waitpid(child_pid_, &st, 0);
      child_pid_ = -1;
    }
    whisper_.Stop();
    piper_.Stop();
    ollama_.Stop();
  }

  void PumpEvents(int ms,
                  std::function<bool()> done = [] { return false; }) {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < deadline) {
      ep_.libHandleEvents(50);
      if (done()) break;
    }
  }
};

TEST_F(SipAnswerTest, BotAnswersCall) {
  auto* call = new CallerCall(*account_);
  std::string dest =
      "sip:bot@127.0.0.1:" + std::to_string(kSipPort);
  pj::CallOpParam op(true);
  op.opt.audioCount = 1;
  op.opt.videoCount = 0;
  ASSERT_NO_THROW(call->makeCall(dest, op));

  PumpEvents(8000, [] { return g_call_answered.load(); });
  EXPECT_TRUE(g_call_answered) << "sloperator did not answer the SIP call";

  try {
    pj::CallOpParam hop;
    hop.statusCode = PJSIP_SC_OK;
    call->hangup(hop);
  } catch (...) {}
  PumpEvents(1000);
}
