// test_chatbot.cpp — unit tests for chatbot_lib.hpp and wyoming_lib.hpp
//
// Tests cover every pure function in chatbot_lib.hpp:
//   • ParseUrl           — URL parsing
//   • JsonGetString      — JSON string field extraction
//   • JsonEscape         — JSON string escaping
//   • BuildMultipart     — multipart form body construction
//   • BuildWav           — WAV container construction
//   • ExtractWavPcm      — WAV PCM extraction
//   • WavSampleRate      — WAV sample-rate field extraction
//   • SplitSentences     — LLM token stream → TTS sentence chunks
//   • RmsEnergy          — VAD energy computation
//   • Vad                — full VAD state-machine
//   • AudioQueue         — thread-safe sample queue
//
// And end-to-end tests for the Wyoming protocol client (wyoming_lib.hpp):
//   • WyomingTranscribe  — STT via mock Wyoming faster-whisper server
//   • WyomingSynthesize  — TTS via mock Wyoming piper server
//   • Pipeline           — VAD → mock STT → mock LLM → mock TTS → AudioQueue

#include "chatbot_lib.hpp"
#include "wyoming_lib.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <arpa/inet.h>
#include <cmath>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <vector>

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

// ═══════════════════════════════════════════════════════════════════════════════
// ParseUrl
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ParseUrl, HostOnly) {
    auto u = ParseUrl("http://localhost");
    EXPECT_EQ(u.host, "localhost");
    EXPECT_EQ(u.port, 80);
    EXPECT_EQ(u.path, "/");
}

TEST(ParseUrl, HostAndPort) {
    auto u = ParseUrl("http://127.0.0.1:9000");
    EXPECT_EQ(u.host, "127.0.0.1");
    EXPECT_EQ(u.port, 9000);
    EXPECT_EQ(u.path, "/");
}

TEST(ParseUrl, HostPortAndPath) {
    auto u = ParseUrl("http://example.com:8080/api/chat");
    EXPECT_EQ(u.host, "example.com");
    EXPECT_EQ(u.port, 8080);
    EXPECT_EQ(u.path, "/api/chat");
}

TEST(ParseUrl, PathOnly) {
    auto u = ParseUrl("http://host/some/path?q=1");
    EXPECT_EQ(u.host, "host");
    EXPECT_EQ(u.port, 80);
    EXPECT_EQ(u.path, "/some/path?q=1");
}

TEST(ParseUrl, NoScheme) {
    // No http:// prefix — treat the whole string as host:port/path.
    auto u = ParseUrl("myhost:1234/foo");
    EXPECT_EQ(u.host, "myhost");
    EXPECT_EQ(u.port, 1234);
    EXPECT_EQ(u.path, "/foo");
}

TEST(ParseUrl, DefaultPortWhenNoPort) {
    auto u = ParseUrl("http://gpu-box/inference");
    EXPECT_EQ(u.port, 80);
    EXPECT_EQ(u.path, "/inference");
}

// ═══════════════════════════════════════════════════════════════════════════════
// JsonGetString
// ═══════════════════════════════════════════════════════════════════════════════

TEST(JsonGetString, SimpleString) {
    EXPECT_EQ(JsonGetString(R"({"text":"hello world"})", "text"), "hello world");
}

TEST(JsonGetString, MultipleFields) {
    std::string json = R"({"model":"llama3","content":"hi there","done":false})";
    EXPECT_EQ(JsonGetString(json, "model"),   "llama3");
    EXPECT_EQ(JsonGetString(json, "content"), "hi there");
}

TEST(JsonGetString, EscapeSequences) {
    std::string json = R"({"text":"line1\nline2\ttab\"quote\\\\"})";
    std::string val  = JsonGetString(json, "text");
    EXPECT_EQ(val, "line1\nline2\ttab\"quote\\\\");
}

TEST(JsonGetString, MissingKey) {
    EXPECT_EQ(JsonGetString(R"({"foo":"bar"})", "baz"), "");
}

TEST(JsonGetString, EmptyJson) {
    EXPECT_EQ(JsonGetString("", "text"), "");
}

TEST(JsonGetString, EmptyValue) {
    EXPECT_EQ(JsonGetString(R"({"text":""})", "text"), "");
}

TEST(JsonGetString, WhitespaceAroundColon) {
    // Standard JSON parsers allow whitespace; our parser looks for ':' after key.
    std::string json = R"({"key" : "value"})";
    EXPECT_EQ(JsonGetString(json, "key"), "value");
}

TEST(JsonGetString, NestedJsonInValue) {
    // Value itself is a JSON string with escaped braces — should be returned as-is.
    std::string json = R"({"content":"{\"inner\":\"val\"}"})";
    EXPECT_EQ(JsonGetString(json, "content"), "{\"inner\":\"val\"}");
}

TEST(JsonGetString, OllamaStreamingLine) {
    // Typical line from Ollama /api/chat stream.
    std::string line = R"({"model":"llama3","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false})";
    EXPECT_EQ(JsonGetString(line, "content"), "Hello");
}

TEST(JsonGetString, OllamaStreamingMultiToken) {
    std::string line = R"({"message":{"role":"assistant","content":"world!"}})";
    EXPECT_EQ(JsonGetString(line, "content"), "world!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// JsonEscape
// ═══════════════════════════════════════════════════════════════════════════════

TEST(JsonEscape, PlainAscii) {
    EXPECT_EQ(JsonEscape("hello world"), "hello world");
}

TEST(JsonEscape, DoubleQuote) {
    EXPECT_EQ(JsonEscape("say \"hi\""), "say \\\"hi\\\"");
}

TEST(JsonEscape, Backslash) {
    EXPECT_EQ(JsonEscape("C:\\Users"), "C:\\\\Users");
}

TEST(JsonEscape, Newline) {
    EXPECT_EQ(JsonEscape("line1\nline2"), "line1\\nline2");
}

TEST(JsonEscape, Tab) {
    EXPECT_EQ(JsonEscape("col1\tcol2"), "col1\\tcol2");
}

TEST(JsonEscape, CarriageReturn) {
    EXPECT_EQ(JsonEscape("a\rb"), "a\\rb");
}

TEST(JsonEscape, AllSpecials) {
    EXPECT_EQ(JsonEscape("\"\\\n\r\t"), "\\\"\\\\\\n\\r\\t");
}

TEST(JsonEscape, EmptyString) {
    EXPECT_EQ(JsonEscape(""), "");
}

TEST(JsonEscape, RoundTrip) {
    // JsonEscape then JsonGetString should recover original value.
    std::string original = "Hello \"World\"\nSecond line\t!";
    std::string json     = "{\"text\":\"" + JsonEscape(original) + "\"}";
    EXPECT_EQ(JsonGetString(json, "text"), original);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BuildMultipart
// ═══════════════════════════════════════════════════════════════════════════════

TEST(BuildMultipart, ContainsBoundary) {
    std::vector<uint8_t> data = {1, 2, 3};
    std::string body = BuildMultipart("BOUNDARY", "file", "a.wav", "audio/wav", data);
    EXPECT_NE(body.find("--BOUNDARY"), std::string::npos);
    EXPECT_NE(body.find("--BOUNDARY--"), std::string::npos);
}

TEST(BuildMultipart, ContainsDisposition) {
    std::vector<uint8_t> data = {0xAB};
    std::string body = BuildMultipart("B", "myfield", "myfile.wav", "audio/wav", data);
    EXPECT_NE(body.find("name=\"myfield\""), std::string::npos);
    EXPECT_NE(body.find("filename=\"myfile.wav\""), std::string::npos);
}

TEST(BuildMultipart, ContainsMimeType) {
    std::vector<uint8_t> data = {1};
    std::string body = BuildMultipart("B", "f", "f.wav", "audio/wav", data);
    EXPECT_NE(body.find("Content-Type: audio/wav"), std::string::npos);
}

TEST(BuildMultipart, ContainsPayload) {
    std::vector<uint8_t> data = {0x52, 0x49, 0x46, 0x46};  // "RIFF"
    std::string body = BuildMultipart("B", "f", "f.wav", "audio/wav", data);
    EXPECT_NE(body.find("RIFF"), std::string::npos);
}

TEST(BuildMultipart, EmptyData) {
    std::vector<uint8_t> data;
    std::string body = BuildMultipart("B", "f", "f.wav", "audio/wav", data);
    // Even with empty data the structure must be well-formed.
    EXPECT_NE(body.find("--B\r\n"), std::string::npos);
    EXPECT_NE(body.find("--B--"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BuildWav / ExtractWavPcm / WavSampleRate
// ═══════════════════════════════════════════════════════════════════════════════

TEST(BuildWav, FourCCsPresent) {
    std::vector<int16_t> samples = {100, -100, 200};
    auto wav = BuildWav(samples);
    EXPECT_EQ(wav[0], 'R'); EXPECT_EQ(wav[1], 'I');
    EXPECT_EQ(wav[2], 'F'); EXPECT_EQ(wav[3], 'F');
    EXPECT_EQ(wav[8],  'W'); EXPECT_EQ(wav[9],  'A');
    EXPECT_EQ(wav[10], 'V'); EXPECT_EQ(wav[11], 'E');
}

TEST(BuildWav, Size) {
    std::vector<int16_t> samples(160);  // one 20 ms frame
    auto wav = BuildWav(samples);
    EXPECT_EQ(wav.size(), 44u + 160u * 2u);
}

TEST(BuildWav, SampleRateField) {
    std::vector<int16_t> samples = {0};
    auto wav = BuildWav(samples, 22050);
    uint32_t sr;
    memcpy(&sr, wav.data() + 24, 4);
    EXPECT_EQ(sr, 22050u);
}

TEST(BuildWav, DefaultSampleRate) {
    std::vector<int16_t> samples = {0};
    auto wav = BuildWav(samples);
    uint32_t sr;
    memcpy(&sr, wav.data() + 24, 4);
    EXPECT_EQ(sr, static_cast<uint32_t>(kAudioRate));
}

TEST(BuildWav, EmptySamples) {
    std::vector<int16_t> samples;
    auto wav = BuildWav(samples);
    EXPECT_EQ(wav.size(), 44u);  // header only, no data
}

TEST(ExtractWavPcm, RoundTrip) {
    std::vector<int16_t> original = {1, -1, 32767, -32768, 0, 100, -100};
    auto wav_bytes = BuildWav(original);
    std::string wav_str(reinterpret_cast<const char *>(wav_bytes.data()), wav_bytes.size());
    auto recovered = ExtractWavPcm(wav_str);
    EXPECT_EQ(recovered, original);
}

TEST(ExtractWavPcm, TooShort) {
    EXPECT_THAT(ExtractWavPcm("RIFF"), IsEmpty());
    EXPECT_THAT(ExtractWavPcm(""), IsEmpty());
}

TEST(ExtractWavPcm, NoDataChunk) {
    // Build a WAV but corrupt the "data" FourCC.
    std::vector<int16_t> samples = {1, 2};
    auto wav = BuildWav(samples);
    wav[36] = 'X';  // corrupt "data" → "Xata"
    std::string wav_str(reinterpret_cast<const char *>(wav.data()), wav.size());
    EXPECT_THAT(ExtractWavPcm(wav_str), IsEmpty());
}

TEST(ExtractWavPcm, ExtraChunkBeforeData) {
    // Construct a WAV with an extra "LIST" chunk before "data".
    std::vector<int16_t> samples = {10, 20, 30};
    auto wav = BuildWav(samples);
    // Insert a fake 4-byte "LIST" chunk (8 byte header + 4 bytes payload) after "WAVE".
    std::vector<uint8_t> extra_chunk = {'L','I','S','T', 4,0,0,0, 'I','N','F','O'};
    std::vector<uint8_t> patched;
    // RIFF header is bytes 0–11; insert after byte 11.
    patched.insert(patched.end(), wav.begin(), wav.begin() + 12);
    patched.insert(patched.end(), extra_chunk.begin(), extra_chunk.end());
    patched.insert(patched.end(), wav.begin() + 12, wav.end());
    // Fix RIFF size field (bytes 4–7).
    uint32_t new_riff_size = static_cast<uint32_t>(patched.size()) - 8;
    memcpy(patched.data() + 4, &new_riff_size, 4);
    std::string patched_str(reinterpret_cast<const char *>(patched.data()), patched.size());
    auto recovered = ExtractWavPcm(patched_str);
    EXPECT_EQ(recovered, samples);
}

TEST(WavSampleRate, ReturnsCorrectRate) {
    std::vector<int16_t> s = {0};
    auto wav = BuildWav(s, 22050);
    std::string wav_str(reinterpret_cast<const char *>(wav.data()), wav.size());
    EXPECT_EQ(WavSampleRate(wav_str), 22050u);
}

TEST(WavSampleRate, TooShort) {
    EXPECT_EQ(WavSampleRate("short"), 0u);
    EXPECT_EQ(WavSampleRate(""), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SplitSentences
// ═══════════════════════════════════════════════════════════════════════════════

TEST(SplitSentences, SingleSentencePeriod) {
    std::string buf = "Hello world.";
    auto parts = SplitSentences(buf);
    EXPECT_THAT(parts, ElementsAre("Hello world."));
    EXPECT_EQ(buf, "");  // no remainder
}

TEST(SplitSentences, MultipleSentences) {
    std::string buf = "First sentence. Second sentence! Third?";
    auto parts = SplitSentences(buf);
    EXPECT_THAT(parts, ElementsAre("First sentence.", "Second sentence!", "Third?"));
    EXPECT_EQ(buf, "");
}

TEST(SplitSentences, RemainderLeft) {
    std::string buf = "Complete sentence. Incomplete";
    auto parts = SplitSentences(buf);
    EXPECT_THAT(parts, ElementsAre("Complete sentence."));
    // The space after the period belongs to the next sentence and is preserved
    // in the remainder; trimming happens only when a sentence is emitted.
    EXPECT_EQ(buf, " Incomplete");
}

TEST(SplitSentences, Newline) {
    std::string buf = "Line one\nLine two\n";
    auto parts = SplitSentences(buf);
    EXPECT_THAT(parts, ElementsAre("Line one\n", "Line two\n"));
    EXPECT_EQ(buf, "");
}

TEST(SplitSentences, LeadingWhitespaceTrimmed) {
    std::string buf = "Sentence one.  Sentence two.";
    auto parts = SplitSentences(buf);
    ASSERT_EQ(parts.size(), 2u);
    // Leading space on "  Sentence two." must be stripped.
    EXPECT_EQ(parts[1], "Sentence two.");
}

TEST(SplitSentences, TooShortSentencesIgnored) {
    // Bare punctuation without content is < 3 chars; must not be emitted.
    std::string buf = "A. B! C?";
    auto parts = SplitSentences(buf);
    // "A." → 2 chars, "B!" → 2 chars, "C?" → 2 chars — all filtered out.
    EXPECT_THAT(parts, IsEmpty());
}

TEST(SplitSentences, EmptyBuffer) {
    std::string buf = "";
    auto parts = SplitSentences(buf);
    EXPECT_THAT(parts, IsEmpty());
    EXPECT_EQ(buf, "");
}

TEST(SplitSentences, NoTerminator) {
    std::string buf = "Just some words";
    auto parts = SplitSentences(buf);
    EXPECT_THAT(parts, IsEmpty());
    EXPECT_EQ(buf, "Just some words");  // all left as remainder
}

TEST(SplitSentences, IncrementalTokenStream) {
    // Simulate Ollama tokens arriving one word at a time.
    std::string buf;
    std::vector<std::string> all;
    for (const char *token : {"Hello", " there", ".", " How", " are", " you", "?"}) {
        buf += token;
        for (auto &s : SplitSentences(buf)) all.push_back(s);
    }
    EXPECT_THAT(all, ElementsAre("Hello there.", "How are you?"));
    EXPECT_EQ(buf, "");
}

// ═══════════════════════════════════════════════════════════════════════════════
// RmsEnergy
// ═══════════════════════════════════════════════════════════════════════════════

TEST(RmsEnergy, ZeroSamples) {
    std::vector<int16_t> s = {0, 0, 0, 0};
    EXPECT_DOUBLE_EQ(RmsEnergy(s.data(), s.size()), 0.0);
}

TEST(RmsEnergy, ConstantValue) {
    // RMS of N identical values v = |v|.
    std::vector<int16_t> s(100, 1000);
    EXPECT_DOUBLE_EQ(RmsEnergy(s.data(), s.size()), 1000.0);
}

TEST(RmsEnergy, KnownValue) {
    // Two samples: 3 and 4 → RMS = sqrt((9+16)/2) = sqrt(12.5)
    std::vector<int16_t> s = {3, 4};
    EXPECT_NEAR(RmsEnergy(s.data(), s.size()), std::sqrt(12.5), 1e-9);
}

TEST(RmsEnergy, NegativeSamples) {
    // RMS of [-v] = v.
    std::vector<int16_t> s(50, -2000);
    EXPECT_DOUBLE_EQ(RmsEnergy(s.data(), s.size()), 2000.0);
}

TEST(RmsEnergy, SingleSample) {
    int16_t v = 500;
    EXPECT_DOUBLE_EQ(RmsEnergy(&v, 1), 500.0);
}

TEST(RmsEnergy, EmptyFrame) {
    EXPECT_DOUBLE_EQ(RmsEnergy(nullptr, 0), 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vad
// ═══════════════════════════════════════════════════════════════════════════════

namespace {
// Generate a PCM frame filled with the given amplitude.
std::vector<int16_t> MakeFrame(int16_t amplitude, int n = kFrameSamples) {
    return std::vector<int16_t>(n, amplitude);
}

// Feed `count` silent frames to the VAD (amplitude = 0).
[[maybe_unused]] void FeedSilence(Vad &vad, int count) {
    auto frame = MakeFrame(0);
    for (int i = 0; i < count; i++)
        vad.ProcessFrame(frame.data(), kFrameSamples);
}

// Feed `count` voiced frames to the VAD.
void FeedVoice(Vad &vad, int count, int16_t amp = 10000) {
    auto frame = MakeFrame(amp);
    for (int i = 0; i < count; i++)
        vad.ProcessFrame(frame.data(), kFrameSamples);
}
}  // namespace

TEST(Vad, SilenceNeverFires) {
    Vad vad;
    int silence_frames = (kSilenceMs / kFrameMs) * 4;
    for (int i = 0; i < silence_frames; i++) {
        auto frame = MakeFrame(0);
        EXPECT_FALSE(vad.ProcessFrame(frame.data(), kFrameSamples));
    }
}

TEST(Vad, ShortUtteranceIgnored) {
    // Speech shorter than kMinSpeechMs must not produce an utterance.
    Vad vad;
    int min_voiced_frames = kMinSpeechMs / kFrameMs;  // e.g. 10
    FeedVoice(vad, min_voiced_frames - 1);
    // Now drain with silence.
    int silence_needed = kSilenceMs / kFrameMs;
    bool fired = false;
    auto frame = MakeFrame(0);
    for (int i = 0; i < silence_needed + 2; i++)
        fired |= vad.ProcessFrame(frame.data(), kFrameSamples);
    EXPECT_FALSE(fired);
}

TEST(Vad, UtteranceFiresAfterSilence) {
    Vad vad;
    int voiced_frames  = kMinSpeechMs / kFrameMs + 5;
    int silence_frames = kSilenceMs   / kFrameMs;

    FeedVoice(vad, voiced_frames);

    bool fired = false;
    auto silent_frame = MakeFrame(0);
    for (int i = 0; i < silence_frames + 1; i++) {
        if (vad.ProcessFrame(silent_frame.data(), kFrameSamples)) {
            fired = true;
            break;
        }
    }
    EXPECT_TRUE(fired);
    EXPECT_FALSE(vad.ready.empty());
}

TEST(Vad, ReadyBufferContainsSpeech) {
    Vad vad;
    int voiced_frames  = kMinSpeechMs / kFrameMs + 5;
    int silence_frames = kSilenceMs   / kFrameMs + 1;

    FeedVoice(vad, voiced_frames);
    bool fired = false;
    auto silent_frame = MakeFrame(0);
    for (int i = 0; i < silence_frames; i++) {
        if (vad.ProcessFrame(silent_frame.data(), kFrameSamples)) {
            fired = true;
            break;
        }
    }
    ASSERT_TRUE(fired);
    // The ready buffer must contain at least the voiced frames.
    EXPECT_GE(vad.ready.size(), static_cast<size_t>(voiced_frames * kFrameSamples));
}

TEST(Vad, StateReset) {
    Vad vad;
    FeedVoice(vad, kMinSpeechMs / kFrameMs + 5);
    vad.Reset();
    EXPECT_FALSE(vad.in_speech);
    EXPECT_EQ(vad.silence_frames, 0);
    EXPECT_TRUE(vad.speech_buf.empty());
    EXPECT_TRUE(vad.ready.empty());
}

TEST(Vad, MultipleUtterances) {
    Vad vad;
    int voiced  = kMinSpeechMs / kFrameMs + 5;
    int silence = kSilenceMs   / kFrameMs + 1;
    int fires   = 0;

    for (int turn = 0; turn < 3; turn++) {
        FeedVoice(vad, voiced);
        auto silent_frame = MakeFrame(0);
        for (int i = 0; i < silence; i++) {
            if (vad.ProcessFrame(silent_frame.data(), kFrameSamples))
                ++fires;
        }
    }
    EXPECT_EQ(fires, 3);
}

TEST(Vad, BelowThresholdIsNotVoiced) {
    // Amplitude just below kVadThreshold should not trigger speech.
    Vad vad;
    int16_t quiet_amp = static_cast<int16_t>(kVadThreshold - 1.0);
    // The RMS of a constant frame equals the amplitude, so this is < threshold.
    int voiced_frames  = kMinSpeechMs / kFrameMs + 5;
    int silence_frames = kSilenceMs   / kFrameMs + 1;
    auto quiet_frame = MakeFrame(quiet_amp);
    for (int i = 0; i < voiced_frames; i++)
        vad.ProcessFrame(quiet_frame.data(), kFrameSamples);
    bool fired = false;
    auto silent_frame = MakeFrame(0);
    for (int i = 0; i < silence_frames; i++)
        fired |= vad.ProcessFrame(silent_frame.data(), kFrameSamples);
    EXPECT_FALSE(fired);
}

TEST(Vad, ExactlyAtThresholdIsVoiced) {
    // Amplitude exactly at kVadThreshold should be treated as voiced.
    Vad vad;
    // A constant frame has RMS == amplitude, so use exactly kVadThreshold.
    // Cast carefully: kVadThreshold = 500.0, fits in int16_t.
    int16_t threshold_amp = static_cast<int16_t>(kVadThreshold);
    int voiced_frames  = kMinSpeechMs / kFrameMs + 5;
    int silence_frames = kSilenceMs   / kFrameMs + 1;
    auto voiced_frame = MakeFrame(threshold_amp);
    for (int i = 0; i < voiced_frames; i++)
        vad.ProcessFrame(voiced_frame.data(), kFrameSamples);
    bool fired = false;
    auto silent_frame = MakeFrame(0);
    for (int i = 0; i < silence_frames; i++) {
        if (vad.ProcessFrame(silent_frame.data(), kFrameSamples)) {
            fired = true; break;
        }
    }
    EXPECT_TRUE(fired);
}

// ═══════════════════════════════════════════════════════════════════════════════
// AudioQueue
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioQueue, PushAndPopExact) {
    AudioQueue q;
    int16_t src[] = {1, 2, 3, 4, 5};
    q.Push(src, 5);
    int16_t dst[5] = {};
    q.Pop(dst, 5);
    EXPECT_THAT(std::vector<int16_t>(dst, dst + 5),
                ElementsAre(1, 2, 3, 4, 5));
    EXPECT_EQ(q.Size(), 0u);
}

TEST(AudioQueue, SizeTracked) {
    AudioQueue q;
    int16_t s[] = {10, 20, 30};
    q.Push(s, 3);
    EXPECT_EQ(q.Size(), 3u);
    int16_t dst[1];
    q.Pop(dst, 1);
    EXPECT_EQ(q.Size(), 2u);
}

TEST(AudioQueue, UnderflowPadsWithZero) {
    AudioQueue q;
    int16_t src[] = {100, 200};
    q.Push(src, 2);
    int16_t dst[5] = {-1, -1, -1, -1, -1};
    q.Pop(dst, 5);
    EXPECT_EQ(dst[0], 100);
    EXPECT_EQ(dst[1], 200);
    EXPECT_EQ(dst[2], 0);   // padded
    EXPECT_EQ(dst[3], 0);
    EXPECT_EQ(dst[4], 0);
}

TEST(AudioQueue, EmptyPopAllZero) {
    AudioQueue q;
    int16_t dst[4] = {1, 2, 3, 4};
    q.Pop(dst, 4);
    EXPECT_THAT(std::vector<int16_t>(dst, dst + 4), ElementsAre(0, 0, 0, 0));
}

TEST(AudioQueue, Clear) {
    AudioQueue q;
    int16_t s[] = {1, 2, 3, 4, 5};
    q.Push(s, 5);
    EXPECT_EQ(q.Size(), 5u);
    q.Clear();
    EXPECT_EQ(q.Size(), 0u);
    int16_t dst[1] = {99};
    q.Pop(dst, 1);
    EXPECT_EQ(dst[0], 0);
}

TEST(AudioQueue, MultiplePushesOrdered) {
    AudioQueue q;
    int16_t a[] = {1, 2};
    int16_t b[] = {3, 4};
    q.Push(a, 2);
    q.Push(b, 2);
    int16_t dst[4];
    q.Pop(dst, 4);
    EXPECT_THAT(std::vector<int16_t>(dst, dst + 4), ElementsAre(1, 2, 3, 4));
}

TEST(AudioQueue, ConcurrentPushPop) {
    // One producer thread pushes 10 000 samples; main thread pops 10 000.
    // Test that all data arrives in order with no corruption.
    AudioQueue q;
    constexpr int kN = 10000;
    std::vector<int16_t> produced(kN);
    for (int i = 0; i < kN; i++) produced[i] = static_cast<int16_t>(i & 0x7FFF);

    std::thread producer([&] {
        constexpr int kChunk = 160;
        for (int off = 0; off < kN; off += kChunk) {
            int n = std::min(kChunk, kN - off);
            q.Push(produced.data() + off, n);
        }
    });

    std::vector<int16_t> consumed(kN);
    int got = 0;
    while (got < kN) {
        int want = std::min(160, kN - got);
        // Spin until there is enough data.
        while (static_cast<int>(q.Size()) < want)
            std::this_thread::yield();
        q.Pop(consumed.data() + got, want);
        got += want;
    }
    producer.join();

    EXPECT_EQ(consumed, produced);
}

TEST(AudioQueue, LargeFrameRoundTrip) {
    // Simulate one second of audio (8000 samples) in one push.
    AudioQueue q;
    std::vector<int16_t> src(kAudioRate);
    for (int i = 0; i < kAudioRate; i++) src[i] = static_cast<int16_t>(i);
    q.Push(src.data(), src.size());
    EXPECT_EQ(q.Size(), static_cast<size_t>(kAudioRate));

    std::vector<int16_t> dst(kAudioRate, 0);
    q.Pop(dst.data(), kAudioRate);
    EXPECT_EQ(dst, src);
    EXPECT_EQ(q.Size(), 0u);
}

// (LooksNumeric, BuildOptionsJson, LoadYaml removed — superseded by
//  nlohmann_json and yaml-cpp; tested via integration rather than unit tests.)

// ═══════════════════════════════════════════════════════════════════════════════
// Wyoming protocol helpers
// ═══════════════════════════════════════════════════════════════════════════════
//
// MockServer spins up a TCP listener on an OS-assigned loopback port.
// Each call to Serve() accepts one connection, runs the provided handler,
// and closes the connection.

namespace {

class MockServer {
public:
    // Creates and binds the listening socket on 127.0.0.1:0.
    // Throws std::runtime_error on failure.
    MockServer() {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) throw std::runtime_error("socket");
        int one = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;
        if (bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0)
            throw std::runtime_error("bind");
        if (listen(listen_fd_, 1) < 0)
            throw std::runtime_error("listen");

        socklen_t len = sizeof(addr);
        getsockname(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);
    }

    ~MockServer() {
        if (listen_fd_ >= 0) close(listen_fd_);
    }

    int port() const { return port_; }

    // Accept one connection, run handler(client_fd), close client_fd.
    void Serve(std::function<void(int)> handler) {
        int client = accept(listen_fd_, nullptr, nullptr);
        if (client < 0) return;
        handler(client);
        close(client);
    }

private:
    int listen_fd_ = -1;
    int port_      = 0;
};

// Receive exactly n bytes into buf; returns true on success.
[[maybe_unused]] static bool RecvAll(int fd, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, static_cast<char *>(buf) + got, n - got, 0);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// ParseHostPort
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ParseHostPort, WithPort) {
    auto [h, p] = ParseHostPort("myhost:1234", 9999);
    EXPECT_EQ(h, "myhost");
    EXPECT_EQ(p, 1234);
}

TEST(ParseHostPort, WithoutPort) {
    auto [h, p] = ParseHostPort("myhost", 9999);
    EXPECT_EQ(h, "myhost");
    EXPECT_EQ(p, 9999);
}

TEST(ParseHostPort, IPv4WithPort) {
    auto [h, p] = ParseHostPort("127.0.0.1:5555", 80);
    EXPECT_EQ(h, "127.0.0.1");
    EXPECT_EQ(p, 5555);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WyomingTranscribe — end-to-end with mock STT server
// ═══════════════════════════════════════════════════════════════════════════════

// Mock Wyoming faster-whisper server that accepts one audio-chunk + audio-stop
// and replies with a "transcript" event.
static void ServeMockStt(int client_fd, const std::string &transcript) {
    // Drain all incoming frames until audio-stop.
    for (int i = 0; i < 64; i++) {
        std::string dj;
        std::vector<uint8_t> pl;
        std::string type = WyomingRecv(client_fd, dj, pl);
        if (type.empty() || type == "audio-stop") break;
    }
    // Send transcript.
    std::string data = "{\"text\":\"" + transcript + "\"}";
    WyomingSend(client_fd, "transcript", data);
}

TEST(WyomingTranscribe, ReturnsTranscript) {
    MockServer server;
    std::string expected = "Hello world";

    std::thread t([&] { server.Serve([&](int fd) { ServeMockStt(fd, expected); }); });

    // Build a minimal voiced PCM utterance.
    std::vector<int16_t> pcm(kFrameSamples, 1000);

    std::string result = WyomingTranscribe(pcm, "127.0.0.1", server.port());
    t.join();

    EXPECT_EQ(result, expected);
}

TEST(WyomingTranscribe, EmptyPcmReturnsEmpty) {
    // Empty PCM must not open a connection at all.
    std::vector<int16_t> pcm;
    std::string result = WyomingTranscribe(pcm, "127.0.0.1", 0 /*unreachable*/);
    EXPECT_TRUE(result.empty());
}

TEST(WyomingTranscribe, SendsCorrectSampleRate) {
    // Verify the audio-chunk header carries kAudioRate.
    MockServer server;
    std::string received_rate;

    std::thread t([&] {
        server.Serve([&](int fd) {
            std::string dj;
            std::vector<uint8_t> pl;
            std::string type = WyomingRecv(fd, dj, pl);
            // Extract "rate" from the data JSON of the audio-chunk.
            if (type == "audio-chunk") {
                auto pos = dj.find("\"rate\":");
                if (pos != std::string::npos)
                    received_rate = dj.substr(pos + 7, dj.find(',', pos + 7) - pos - 7);
            }
            // Drain audio-stop and reply.
            for (int i = 0; i < 4; i++) {
                std::string dj2; std::vector<uint8_t> pl2;
                if (WyomingRecv(fd, dj2, pl2) == "audio-stop") break;
            }
            WyomingSend(fd, "transcript", "{\"text\":\"ok\"}");
        });
    });

    std::vector<int16_t> pcm(kFrameSamples, 500);
    WyomingTranscribe(pcm, "127.0.0.1", server.port());
    t.join();

    EXPECT_EQ(received_rate, std::to_string(kAudioRate));
}

TEST(WyomingTranscribe, SendsRawPcmPayload) {
    // The audio-chunk payload must contain the exact raw s16le bytes.
    MockServer server;
    std::vector<uint8_t> received_payload;

    std::thread t([&] {
        server.Serve([&](int fd) {
            std::string dj;
            std::vector<uint8_t> pl;
            std::string type = WyomingRecv(fd, dj, pl);
            if (type == "audio-chunk") received_payload = pl;
            // Drain audio-stop and reply.
            for (int i = 0; i < 4; i++) {
                std::string dj2; std::vector<uint8_t> pl2;
                if (WyomingRecv(fd, dj2, pl2) == "audio-stop") break;
            }
            WyomingSend(fd, "transcript", "{\"text\":\"ok\"}");
        });
    });

    std::vector<int16_t> pcm = {100, -200, 300, -400};
    WyomingTranscribe(pcm, "127.0.0.1", server.port());
    t.join();

    ASSERT_EQ(received_payload.size(), pcm.size() * sizeof(int16_t));
    // Compare byte-by-byte.
    const uint8_t *expected = reinterpret_cast<const uint8_t *>(pcm.data());
    for (size_t i = 0; i < received_payload.size(); i++)
        EXPECT_EQ(received_payload[i], expected[i]) << "at byte " << i;
}

TEST(WyomingTranscribe, ErrorEventReturnsEmpty) {
    // If the server sends "error" instead of "transcript", result must be "".
    MockServer server;
    std::thread t([&] {
        server.Serve([&](int fd) {
            // Drain all frames.
            for (int i = 0; i < 8; i++) {
                std::string dj; std::vector<uint8_t> pl;
                if (WyomingRecv(fd, dj, pl) == "audio-stop") break;
            }
            WyomingSend(fd, "error", "{\"text\":\"internal error\"}");
        });
    });

    std::vector<int16_t> pcm(kFrameSamples, 1000);
    std::string result = WyomingTranscribe(pcm, "127.0.0.1", server.port());
    t.join();

    EXPECT_TRUE(result.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// WyomingSynthesize — end-to-end with mock TTS server
// ═══════════════════════════════════════════════════════════════════════════════

// Mock Wyoming piper server: accepts a synthesize request and sends back a
// small PCM chunk at the given sample rate.
static void ServeMockTts(int client_fd,
                          const std::vector<int16_t> &pcm_to_send,
                          int rate = kAudioRate) {
    // Drain synthesize request.
    {
        std::string dj; std::vector<uint8_t> pl;
        WyomingRecv(client_fd, dj, pl);  // "synthesize"
    }
    // audio-start
    std::string start_data = "{\"rate\":" + std::to_string(rate)
                           + ",\"width\":2,\"channels\":1}";
    WyomingSend(client_fd, "audio-start", start_data);
    // audio-chunk
    const void *raw = pcm_to_send.data();
    size_t raw_len  = pcm_to_send.size() * sizeof(int16_t);
    WyomingSend(client_fd, "audio-chunk", "", raw, raw_len);
    // audio-stop
    WyomingSend(client_fd, "audio-stop", "");
}

TEST(WyomingSynthesize, ReturnsPcm) {
    MockServer server;
    std::vector<int16_t> expected = {100, 200, 300, -100, -200};

    std::thread t([&] {
        server.Serve([&](int fd) { ServeMockTts(fd, expected); });
    });

    auto result = WyomingSynthesize("Hello.", "127.0.0.1", server.port());
    t.join();

    EXPECT_EQ(result, expected);
}

TEST(WyomingSynthesize, EmptyTextReturnsEmpty) {
    auto result = WyomingSynthesize("", "127.0.0.1", 0 /*unreachable*/);
    EXPECT_TRUE(result.empty());
}

TEST(WyomingSynthesize, SendsSynthesizeRequest) {
    // Verify the request contains the text and channels field.
    MockServer server;
    std::string received_text;

    std::thread t([&] {
        server.Serve([&](int fd) {
            // Read the synthesize request (don't call ServeMockTts which would
            // try to read a second synthesize frame).
            std::string dj; std::vector<uint8_t> pl;
            std::string type = WyomingRecv(fd, dj, pl);
            if (type == "synthesize") received_text = JsonGetString(dj, "text");
            // Reply with an empty audio sequence.
            WyomingSend(fd, "audio-start",
                        "{\"rate\":" + std::to_string(kAudioRate) + ",\"channels\":1}");
            WyomingSend(fd, "audio-stop", "");
        });
    });

    WyomingSynthesize("Test sentence.", "127.0.0.1", server.port());
    t.join();

    EXPECT_EQ(received_text, "Test sentence.");
}

TEST(WyomingSynthesize, DownsamplesWhenRateDiffers) {
    // If the server returns audio at 2× kAudioRate, the result should be ½ the length.
    MockServer server;
    int double_rate = kAudioRate * 2;
    // 8 samples at double rate → 4 samples at kAudioRate after 2× decimation.
    std::vector<int16_t> hi_rate_pcm = {10, 20, 30, 40, 50, 60, 70, 80};

    std::thread t([&] {
        server.Serve([&](int fd) { ServeMockTts(fd, hi_rate_pcm, double_rate); });
    });

    auto result = WyomingSynthesize("Text.", "127.0.0.1", server.port());
    t.join();

    // Every other sample is kept (step = 2).
    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 30);
    EXPECT_EQ(result[2], 50);
    EXPECT_EQ(result[3], 70);
}

TEST(WyomingSynthesize, MultipleChunksAreConcat) {
    // TTS servers may send many small audio-chunks; all must be concatenated.
    MockServer server;
    std::vector<int16_t> chunk_a = {1, 2, 3};
    std::vector<int16_t> chunk_b = {4, 5, 6};

    std::thread t([&] {
        server.Serve([&](int fd) {
            // Drain synthesize.
            { std::string dj; std::vector<uint8_t> pl; WyomingRecv(fd, dj, pl); }
            // audio-start
            WyomingSend(fd, "audio-start",
                        "{\"rate\":" + std::to_string(kAudioRate) + ",\"channels\":1}");
            // Two audio-chunk messages.
            WyomingSend(fd, "audio-chunk", "",
                        chunk_a.data(), chunk_a.size() * sizeof(int16_t));
            WyomingSend(fd, "audio-chunk", "",
                        chunk_b.data(), chunk_b.size() * sizeof(int16_t));
            WyomingSend(fd, "audio-stop", "");
        });
    });

    auto result = WyomingSynthesize("Two chunks.", "127.0.0.1", server.port());
    t.join();

    EXPECT_THAT(result, ElementsAre(1, 2, 3, 4, 5, 6));
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end pipeline: VAD → mock STT → mock LLM → mock TTS → AudioQueue
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Pipeline, VADToSTTToLLMToTTSToAudioQueue) {
    // ── Mock STT server ──────────────────────────────────────────────────────
    MockServer stt_server;
    const std::string kTranscript = "What time is it?";

    std::thread stt_thread([&] {
        stt_server.Serve([&](int fd) { ServeMockStt(fd, kTranscript); });
    });

    // ── Mock LLM (inline — no network needed) ───────────────────────────────
    // Simulates the sentence-splitting step: given a user utterance, emit one
    // sentence that the TTS mock will handle.
    const std::string kReply = "It is three o'clock.";
    auto mock_llm = [&](const std::string & /*user_text*/,
                        std::function<void(const std::string &)> on_sentence) {
        on_sentence(kReply);
    };

    // ── Mock TTS server (one connection per sentence) ────────────────────────
    MockServer tts_server;
    // Synthesized response: a simple 1-frame PCM burst.
    std::vector<int16_t> tts_pcm(kFrameSamples, 4000);

    // Collect sentences from the LLM so we know how many TTS connections to serve.
    std::vector<std::string> sentences;

    // ── Simulate a voiced utterance via VAD ─────────────────────────────────
    Vad vad;
    AudioQueue audio_out;
    bool utterance_ready = false;

    // Feed voiced frames (amplitude well above kVadThreshold).
    int voiced_frames = kMinSpeechMs / kFrameMs + 5;
    auto voiced_frame = std::vector<int16_t>(kFrameSamples, 10000);
    for (int i = 0; i < voiced_frames; i++)
        vad.ProcessFrame(voiced_frame.data(), kFrameSamples);

    // Feed silence until VAD fires.
    auto silent_frame = std::vector<int16_t>(kFrameSamples, 0);
    for (int i = 0; i < kSilenceMs / kFrameMs + 2; i++) {
        if (vad.ProcessFrame(silent_frame.data(), kFrameSamples)) {
            utterance_ready = true;
            break;
        }
    }
    ASSERT_TRUE(utterance_ready) << "VAD did not fire";

    // ── Run the pipeline stages ──────────────────────────────────────────────
    // Stage 1: STT — transcribe the VAD utterance.
    std::string transcript = WyomingTranscribe(
        vad.ready, "127.0.0.1", stt_server.port());
    EXPECT_EQ(transcript, kTranscript);

    // Stage 2: LLM — generate response sentences from the transcript.
    mock_llm(transcript, [&](const std::string &s) { sentences.push_back(s); });
    ASSERT_FALSE(sentences.empty());

    // Start TTS mock server — exactly one connection per sentence.
    std::thread tts_thread([&] {
        for (size_t i = 0; i < sentences.size(); i++)
            tts_server.Serve([&](int fd) { ServeMockTts(fd, tts_pcm); });
    });

    // Stage 3: TTS — synthesize each sentence and push into AudioQueue.
    for (const auto &sentence : sentences) {
        auto pcm = WyomingSynthesize(sentence, "127.0.0.1", tts_server.port());
        EXPECT_FALSE(pcm.empty());
        audio_out.Push(pcm.data(), pcm.size());
    }

    // ── Verify AudioQueue received the synthesized audio ─────────────────────
    EXPECT_GT(audio_out.Size(), 0u);
    std::vector<int16_t> out(audio_out.Size());
    audio_out.Pop(out.data(), out.size());
    // All samples must come from the TTS mock (amplitude 4000).
    for (auto s : out) EXPECT_EQ(s, 4000);

    stt_thread.join();
    tts_thread.join();
}
