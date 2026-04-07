#pragma once
// wyoming_lib.hpp — Wyoming speech-protocol client helpers for sloperator
//
// Provides the network-I/O layer that sits between chatbot_lib.hpp (pure
// utility functions) and main.cpp (SIP / PJSIP entry point).
//
// All functions accept explicit host/port parameters instead of global state
// so they can be tested with in-process mock servers.
//
// Covers:
//   • ParseHostPort       — split "host:port" strings
//   • ConnectTcp          — blocking TCP connect → fd
//   • SendAll             — write loop that tolerates partial sends
//   • WyomingSend         — emit one Wyoming protocol frame
//   • WyomingRecv         — receive one Wyoming protocol frame
//   • WyomingTranscribe   — run STT via Wyoming faster-whisper server
//   • WyomingSynthesize   — run TTS via Wyoming piper server

#include "chatbot_lib.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// ── Transport helpers ─────────────────────────────────────────────────────────

// Parse "host:port" → {host, port}. Uses default_port when no colon is found.
inline std::pair<std::string, int> ParseHostPort(const std::string &addr,
                                                  int default_port) {
    auto colon = addr.rfind(':');
    if (colon == std::string::npos) return {addr, default_port};
    return {addr.substr(0, colon), std::stoi(addr.substr(colon + 1))};
}

// Create a TCP connection to host:port.  Returns the file descriptor, or -1.
inline int ConnectTcp(const std::string &host, int port) {
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

// Write exactly len bytes to fd.  Returns false on short write or error.
inline bool SendAll(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        buf += n; len -= n;
    }
    return true;
}

// ── Wyoming protocol ──────────────────────────────────────────────────────────

// Send one Wyoming frame: header JSON line, optional data JSON, optional payload.
inline bool WyomingSend(int fd,
                         const std::string &type,
                         const std::string &data_json,
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
    if (payload && payload_len > 0 &&
        !SendAll(fd, static_cast<const char *>(payload), payload_len))
        return false;
    return true;
}

// Receive one Wyoming frame.  Returns the event type string, or "" on error.
// Fills data_json (if present) and payload (if present).
inline std::string WyomingRecv(int fd,
                                std::string &data_json,
                                std::vector<uint8_t> &payload) {
    data_json.clear();
    payload.clear();

    // Read newline-terminated header.
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
                got += static_cast<size_t>(n);
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
                got += static_cast<size_t>(n);
            }
        }
    }

    return type;
}

// ── Wyoming service calls ─────────────────────────────────────────────────────

// Transcribe PCM samples via a Wyoming faster-whisper server.
// Sends audio-chunk + audio-stop; reads until a "transcript" event arrives.
// Returns the transcript text, or "" on failure.
inline std::string WyomingTranscribe(const std::vector<int16_t> &pcm,
                                      const std::string &host, int port) {
    if (pcm.empty()) return {};
    int fd = ConnectTcp(host, port);
    if (fd < 0) return {};

    std::string audio_data = "{\"rate\":" + std::to_string(kAudioRate)
                           + ",\"width\":2,\"channels\":1,\"timestamp\":null}";
    const void *raw     = pcm.data();
    size_t      raw_len = pcm.size() * sizeof(int16_t);
    if (!WyomingSend(fd, "audio-chunk", audio_data, raw, raw_len)) {
        close(fd); return {};
    }
    if (!WyomingSend(fd, "audio-stop", "{\"timestamp\":null}")) {
        close(fd); return {};
    }

    std::string result;
    for (int i = 0; i < 32; i++) {
        std::string       dj;
        std::vector<uint8_t> pl;
        std::string type = WyomingRecv(fd, dj, pl);
        if (type.empty()) break;
        if (type == "transcript") { result = JsonGetString(dj, "text"); break; }
        if (type == "error") break;
    }
    close(fd);
    return result;
}

// Synthesize text to s16le PCM via a Wyoming piper server.
// Collects audio-chunk payloads until audio-stop; downsamples if needed.
// Returns an empty vector on failure.
inline std::vector<int16_t> WyomingSynthesize(const std::string &text,
                                               const std::string &host, int port) {
    if (text.empty()) return {};
    int fd = ConnectTcp(host, port);
    if (fd < 0) return {};

    std::string synth_data = "{\"text\":\"" + JsonEscape(text) + "\",\"channels\":1}";
    if (!WyomingSend(fd, "synthesize", synth_data)) { close(fd); return {}; }

    std::vector<int16_t> pcm;
    int piper_rate = kAudioRate;
    for (int i = 0; i < 4096; i++) {
        std::string       dj;
        std::vector<uint8_t> pl;
        std::string type = WyomingRecv(fd, dj, pl);
        if (type.empty()) break;
        if (type == "audio-start") {
            auto rate_pos = dj.find("\"rate\":");
            if (rate_pos != std::string::npos)
                piper_rate = std::stoi(dj.substr(rate_pos + 7));
        } else if (type == "audio-chunk") {
            size_t n    = pl.size() / 2;
            size_t base = pcm.size();
            pcm.resize(base + n);
            memcpy(pcm.data() + base, pl.data(), pl.size());
        } else if (type == "audio-stop") {
            break;
        } else if (type == "error") {
            break;
        }
    }
    close(fd);

    // Downsample if piper uses a different sample rate (e.g. 22050 → 16000).
    if (piper_rate != kAudioRate && !pcm.empty()) {
        double ratio = static_cast<double>(piper_rate) / kAudioRate;
        std::vector<int16_t> resampled;
        resampled.reserve(static_cast<size_t>(pcm.size() / ratio) + 1);
        // Use a floating-point source position to avoid integer rounding
        // error accumulation over large buffers.
        for (double pos = 0.0; static_cast<size_t>(pos) < pcm.size(); pos += ratio)
            resampled.push_back(pcm[static_cast<size_t>(pos)]);
        return resampled;
    }
    return pcm;
}
