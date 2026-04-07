#pragma once
// chatbot_lib.hpp — pure utility functions for sloperator
//
// All functions here are free of network I/O and PJSIP dependencies so they
// can be unit-tested without any external services or SIP infrastructure.
//
// Covers:
//   • URL parsing
//   • JSON string extraction and escaping
//   • WAV container building and PCM extraction
//   • Multipart form body building (for Whisper upload)
//   • Sentence splitting (LLM token stream → TTS chunks)
//   • AudioQueue (thread-safe int16_t sample ring buffer)
//   • VAD helper (RMS energy computation)

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

// ── Audio constants ───────────────────────────────────────────────────────────
static constexpr int kAudioRate    = 16000;
static constexpr int kFrameMs      = 20;
static constexpr int kFrameSamples = kAudioRate * kFrameMs / 1000;  // 320

// ── VAD constants ─────────────────────────────────────────────────────────────
static constexpr double kVadThreshold = 500.0;
static constexpr int    kSilenceMs    = 600;
static constexpr int    kMinSpeechMs  = 200;

// ── URL parsing ───────────────────────────────────────────────────────────────

struct HttpUrl {
    std::string host;
    int         port = 80;
    std::string path;
};

inline HttpUrl ParseUrl(const std::string &url) {
    HttpUrl r;
    std::string s = url;
    if (s.substr(0, 7) == "http://") s = s.substr(7);
    auto slash = s.find('/');
    std::string hostport = (slash == std::string::npos) ? s : s.substr(0, slash);
    r.path = (slash == std::string::npos) ? "/" : s.substr(slash);
    auto colon = hostport.find(':');
    if (colon == std::string::npos) {
        r.host = hostport;
    } else {
        r.host = hostport.substr(0, colon);
        r.port = std::stoi(hostport.substr(colon + 1));
    }
    return r;
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

// Extract the value of a string field: {"key":"value"} → value.
// Handles basic escape sequences inside the value.
inline std::string JsonGetString(const std::string &json, const std::string &key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return {};
    ++pos;
    std::string val;
    while (pos < json.size()) {
        char c = json[pos++];
        if (c == '"') break;
        if (c == '\\' && pos < json.size()) {
            char esc = json[pos++];
            switch (esc) {
                case '"':  val += '"';  break;
                case '\\': val += '\\'; break;
                case 'n':  val += '\n'; break;
                case 'r':  val += '\r'; break;
                case 't':  val += '\t'; break;
                default:   val += esc;  break;
            }
        } else {
            val += c;
        }
    }
    return val;
}

// Escape a string for embedding in a JSON string literal.
inline std::string JsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ── Multipart form body builder ───────────────────────────────────────────────

inline std::string BuildMultipart(const std::string &boundary,
                                  const std::string &field_name,
                                  const std::string &filename,
                                  const std::string &mime,
                                  const std::vector<uint8_t> &data) {
    std::string body;
    body += "--" + boundary + "\r\n";
    body += "Content-Disposition: form-data; name=\"" + field_name
          + "\"; filename=\"" + filename + "\"\r\n";
    body += "Content-Type: " + mime + "\r\n\r\n";
    body.append(reinterpret_cast<const char *>(data.data()), data.size());
    body += "\r\n--" + boundary + "--\r\n";
    return body;
}

// ── WAV container ─────────────────────────────────────────────────────────────

// Build a minimal PCM WAV file from s16le mono samples.
inline std::vector<uint8_t> BuildWav(const std::vector<int16_t> &samples,
                                     int sample_rate = kAudioRate) {
    uint32_t data_size   = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    uint32_t riff_size   = 36 + data_size;
    uint16_t channels    = 1;
    uint16_t bits        = 16;
    uint32_t byte_rate   = sample_rate * channels * bits / 8;
    uint16_t block_align = channels * bits / 8;

    std::vector<uint8_t> wav;
    wav.reserve(44 + data_size);
    auto w2 = [&](uint16_t v) { wav.push_back(v & 0xFF); wav.push_back(v >> 8); };
    auto w4 = [&](uint32_t v) { w2(v & 0xFFFF); w2(v >> 16); };
    auto ws = [&](const char *s) { for (; *s; ++s) wav.push_back(*s); };
    ws("RIFF"); w4(riff_size); ws("WAVE");
    ws("fmt "); w4(16); w2(1); w2(channels); w4(static_cast<uint32_t>(sample_rate));
    w4(byte_rate); w2(block_align); w2(bits);
    ws("data"); w4(data_size);
    const uint8_t *p = reinterpret_cast<const uint8_t *>(samples.data());
    wav.insert(wav.end(), p, p + data_size);
    return wav;
}

// Extract the sample rate embedded in a WAV header (bytes 24–27).
// Returns 0 if the buffer is too short.
inline uint32_t WavSampleRate(const std::string &wav_data) {
    if (wav_data.size() < 28) return 0;
    uint32_t sr;
    memcpy(&sr, wav_data.data() + 24, 4);
    return sr;
}

// Parse a WAV container and return the raw s16le PCM samples from the "data"
// chunk.  Handles WAV files with extra metadata chunks between "fmt " and
// "data" (e.g. files produced by some TTS engines).
inline std::vector<int16_t> ExtractWavPcm(const std::string &wav_data) {
    if (wav_data.size() < 44) return {};
    size_t offset = 12;  // skip RIFF header (12 bytes)
    while (offset + 8 <= wav_data.size()) {
        std::string id = wav_data.substr(offset, 4);
        uint32_t sz;
        memcpy(&sz, wav_data.data() + offset + 4, 4);
        if (id == "data") {
            offset += 8;
            if (offset + sz > wav_data.size()) sz = wav_data.size() - offset;
            size_t n_samples = sz / 2;
            std::vector<int16_t> out(n_samples);
            memcpy(out.data(), wav_data.data() + offset, n_samples * 2);
            return out;
        }
        offset += 8 + sz;
    }
    return {};
}

// ── Sentence splitter ─────────────────────────────────────────────────────────
// Splits the accumulated LLM token stream into sentences on '.', '!', '?',
// '\n'.  Returns complete sentences; leaves the incomplete tail in `buf`.
// Sentences are trimmed of leading/trailing whitespace and quote characters.
// A sentence must contain at least 2 word characters to be emitted (avoids
// emitting bare punctuation, stray quotes, etc.).
inline std::vector<std::string> SplitSentences(std::string &buf) {
    // Strip leading whitespace and quotes; strip trailing quotes only
    // (keep terminal punctuation and newlines that end the sentence).
    static constexpr char kTrimLeading[] = " \t\r\n\"'`";
    static constexpr char kTrimTrailing[] = "\"'`";

    auto emit = [&](std::string s) -> std::string {
        size_t a = s.find_first_not_of(kTrimLeading);
        if (a == std::string::npos) return {};
        size_t b = s.find_last_not_of(kTrimTrailing);
        s = s.substr(a, b - a + 1);
        // Require at least 2 word (non-space, non-punctuation) characters.
        int words = 0;
        for (char c : s)
            if (std::isalnum(static_cast<unsigned char>(c))) ++words;
        return words >= 2 ? s : std::string{};
    };

    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i < buf.size(); i++) {
        char c = buf[i];
        if (c == '.' || c == '!' || c == '?' || c == '\n') {
            std::string s = emit(buf.substr(start, i + 1 - start));
            if (!s.empty()) out.push_back(std::move(s));
            start = i + 1;
        }
    }
    buf = buf.substr(start);
    return out;
}

// ── VAD helper ────────────────────────────────────────────────────────────────

// Compute RMS energy of a PCM frame.
inline double RmsEnergy(const int16_t *samples, int n) {
    if (n <= 0) return 0.0;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += static_cast<double>(samples[i]) * samples[i];
    return std::sqrt(sum / n);
}

// VAD state machine.  Feed 20 ms PCM frames via ProcessFrame(); when an
// utterance is complete it is moved into `ready` and the method returns true.
struct Vad {
    // Tunable parameters — defaults match the original compile-time constants.
    double vad_threshold = kVadThreshold;  // RMS level for voiced frame
    int    silence_ms    = kSilenceMs;     // silence duration to end utterance
    int    min_speech_ms = kMinSpeechMs;   // minimum utterance length

    std::vector<int16_t> speech_buf;
    std::vector<int16_t> ready;          // filled when ProcessFrame returns true
    int  voiced_frames = 0;
    int  silence_frames = 0;
    bool in_speech      = false;

    // Returns true when a complete utterance has been moved into `ready`.
    bool ProcessFrame(const int16_t *samples, int n) {
        double rms    = RmsEnergy(samples, n);
        bool   voiced = (rms >= vad_threshold);

        if (voiced) {
            if (!in_speech) {
                in_speech      = true;
                silence_frames = 0;
                voiced_frames  = 0;
            }
            speech_buf.insert(speech_buf.end(), samples, samples + n);
            ++voiced_frames;
            silence_frames = 0;
        } else if (in_speech) {
            speech_buf.insert(speech_buf.end(), samples, samples + n);
            ++silence_frames;
            int silence_needed = silence_ms / kFrameMs;
            if (silence_frames >= silence_needed) {
                int min_frames = min_speech_ms / kFrameMs;
                if (voiced_frames >= min_frames) {
                    ready      = std::move(speech_buf);
                    speech_buf.clear();
                    in_speech      = false;
                    silence_frames = 0;
                    voiced_frames  = 0;
                    return true;
                }
                speech_buf.clear();
                in_speech      = false;
                silence_frames = 0;
                voiced_frames  = 0;
            }
        }
        return false;
    }

    void Reset() {
        speech_buf.clear();
        ready.clear();
        voiced_frames  = 0;
        silence_frames = 0;
        in_speech      = false;
    }
};

// ── AudioQueue ────────────────────────────────────────────────────────────────
// Thread-safe int16_t sample queue.  onFrameRequested drains kFrameSamples at
// a time; the pipeline thread pushes TTS PCM in bulk.
class AudioQueue {
public:
    void Push(const int16_t *samples, size_t n) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t i = 0; i < n; i++) q_.push_back(samples[i]);
        }
        cv_.notify_one();
    }

    // Fill exactly `n` samples into `out`.  Pads with silence on underflow.
    void Pop(int16_t *out, size_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t avail = q_.size();
        size_t copy  = std::min(avail, n);
        for (size_t i = 0; i < copy; i++) { out[i] = q_.front(); q_.pop_front(); }
        for (size_t i = copy; i < n; i++) out[i] = 0;
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

    void Clear() {
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear();
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<int16_t>     q_;
};
