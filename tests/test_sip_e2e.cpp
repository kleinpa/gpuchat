// test_sip_e2e.cpp — SIP end-to-end test with Opus codec
//
// Goal: exercise the PJSIP + Opus codec linking path that the unit tests in
// test_chatbot.cpp never touch.  Codec linking bugs (missing symbols, wrong
// library order, PJMEDIA_HAS_G711_CODEC=0 side-effects) only surface when
// pjmedia-codec is actually loaded and asked to negotiate Opus.
//
// What this test does:
//   1. Bring up two PJSUA2 Endpoint instances on separate loopback ports
//      (callee on an OS-assigned port, caller on another).
//   2. Caller dials callee with an SDP that lists only Opus.
//   3. Callee auto-answers.
//   4. Both sides attach a minimal AudioMediaPort so RTP frames actually flow.
//   5. After a short hold (~500 ms of RTP), caller hangs up.
//   6. Test verifies:
//      - Media status reached ACTIVE (codec negotiation succeeded).
//      - The negotiated codec name contains "opus" (case-insensitive).
//      - No PJSIP exception was thrown during the call.
//
// LLM / STT / TTS are completely absent — this is a codec + SIP stack test.
//
// NOTE: Two PJSUA2 Endpoint singletons cannot coexist in the same process.
// We work around this by using two pjsua_transport_id / two accounts inside a
// *single* Endpoint, one for the callee account (auto-answers) and one for the
// caller account (originates the call).

#include <pjsua2.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Returns a free loopback port by binding a temporary socket to :0.
static int FreeLoopbackPort() {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return 0;
    struct sockaddr_in a{};
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port        = 0;
    ::bind(fd, reinterpret_cast<struct sockaddr *>(&a), sizeof(a));
    socklen_t len = sizeof(a);
    ::getsockname(fd, reinterpret_cast<struct sockaddr *>(&a), &len);
    int port = ntohs(a.sin_port);
    ::close(fd);
    return port;
}

// ── Minimal audio port: push/pop silence ──────────────────────────────────────

class SilentAudioPort : public pj::AudioMediaPort {
public:
    SilentAudioPort() {
        pj::MediaFormatAudio fmt;
        fmt.type          = PJMEDIA_TYPE_AUDIO;
        fmt.clockRate     = 48000;
        fmt.channelCount  = 2;
        fmt.bitsPerSample = 16;
        fmt.frameTimeUsec = 20 * 1000;  // 20 ms frames
        createPort("silent", fmt);
    }

    void onFrameRequested(pj::MediaFrame &frame) override {
        frame.type = PJMEDIA_FRAME_TYPE_AUDIO;
        if (frame.buf.size() > 0)
            std::fill(frame.buf.begin(), frame.buf.end(), 0);
        ++frames_sent;
    }

    void onFrameReceived(pj::MediaFrame &frame) override {
        if (frame.type == PJMEDIA_FRAME_TYPE_AUDIO)
            ++frames_received;
    }

    std::atomic<int> frames_sent{0};
    std::atomic<int> frames_received{0};
};

// ── State shared between PJSUA2 callbacks and the test body ──────────────────

struct CallState {
    std::mutex              mu;
    std::condition_variable cv;

    bool   media_active   = false;
    bool   disconnected   = false;
    std::string codec_name;   // negotiated codec, filled on media-active

    void NotifyMediaActive(const std::string &codec) {
        std::lock_guard<std::mutex> lk(mu);
        media_active = true;
        codec_name   = codec;
        cv.notify_all();
    }

    void NotifyDisconnected() {
        std::lock_guard<std::mutex> lk(mu);
        disconnected = true;
        cv.notify_all();
    }

    bool WaitForMediaActive(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, timeout, [this] { return media_active; });
    }

    bool WaitForDisconnected(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, timeout, [this] { return disconnected; });
    }
};

// ── Callee call: auto-answers, attaches SilentAudioPort ───────────────────────

class CalleeCall : public pj::Call {
public:
    CalleeCall(pj::Account &acc, int call_id, CallState &state)
        : pj::Call(acc, call_id), state_(state) {}

    void onCallState(pj::OnCallStateParam & /*prm*/) override {
        pj::CallInfo ci = getInfo();
        if (ci.state == PJSIP_INV_STATE_DISCONNECTED)
            state_.NotifyDisconnected();
    }

    void onCallMediaState(pj::OnCallMediaStateParam & /*prm*/) override {
        pj::CallInfo ci = getInfo();
        for (unsigned i = 0; i < ci.media.size(); i++) {
            if (ci.media[i].type != PJMEDIA_TYPE_AUDIO) continue;

            auto *aud = dynamic_cast<pj::AudioMedia *>(getMedia(i));

            if (ci.media[i].status == PJSUA_CALL_MEDIA_ACTIVE) {
                if (!aud) continue;
                if (!port_) {
                    port_ = std::make_unique<SilentAudioPort>();
                    aud->startTransmit(*port_);
                    port_->startTransmit(*aud);
                }

                // Extract codec name from the stream info.
                std::string codec;
                pjsua_call_info raw{};
                pjsua_call_get_info(getId(), &raw);
                if (raw.media_cnt > 0) {
                    pjsua_stream_info si{};
                    if (pjsua_call_get_stream_info(getId(), 0, &si) == PJ_SUCCESS
                        && si.type == PJMEDIA_TYPE_AUDIO) {
                        codec = std::string(si.info.aud.fmt.encoding_name.ptr,
                                           si.info.aud.fmt.encoding_name.slen);
                    }
                }
                state_.NotifyMediaActive(codec);
            } else {
                // Media no longer active — stop transmit before port_ is destroyed.
                if (port_ && aud) {
                    try { aud->stopTransmit(*port_); } catch (...) {}
                    try { port_->stopTransmit(*aud); } catch (...) {}
                }
                port_.reset();
            }
        }
    }

private:
    CallState  &state_;
    std::unique_ptr<SilentAudioPort> port_;
};

// ── Callee account ────────────────────────────────────────────────────────────

class CalleeAccount : public pj::Account {
public:
    explicit CalleeAccount(CallState &state) : state_(state) {}

    void onIncomingCall(pj::OnIncomingCallParam &prm) override {
        call_ = std::make_unique<CalleeCall>(*this, prm.callId, state_);
        pj::CallOpParam op;
        op.statusCode = PJSIP_SC_OK;
        call_->answer(op);
    }

private:
    CallState &state_;
    std::unique_ptr<CalleeCall> call_;
};

// ── Caller call ───────────────────────────────────────────────────────────────

class CallerCall : public pj::Call {
public:
    CallerCall(pj::Account &acc, CallState &state)
        : pj::Call(acc), state_(state) {}

    void onCallMediaState(pj::OnCallMediaStateParam & /*prm*/) override {
        pj::CallInfo ci = getInfo();
        for (unsigned i = 0; i < ci.media.size(); i++) {
            if (ci.media[i].type != PJMEDIA_TYPE_AUDIO) continue;

            auto *aud = dynamic_cast<pj::AudioMedia *>(getMedia(i));

            if (ci.media[i].status == PJSUA_CALL_MEDIA_ACTIVE) {
                if (!aud || port_) continue;
                port_ = std::make_unique<SilentAudioPort>();
                aud->startTransmit(*port_);
                port_->startTransmit(*aud);
            } else {
                if (port_ && aud) {
                    try { aud->stopTransmit(*port_); } catch (...) {}
                    try { port_->stopTransmit(*aud); } catch (...) {}
                }
                port_.reset();
            }
        }
    }

private:
    CallState  &state_;
    std::unique_ptr<SilentAudioPort> port_;
};

// ── The test ──────────────────────────────────────────────────────────────────

TEST(SipE2E, OpusCodecNegotiation) {
    // Pick two free ports for the SIP UDP transports.
    int callee_sip_port = FreeLoopbackPort();
    int caller_sip_port = FreeLoopbackPort();
    ASSERT_GT(callee_sip_port, 0);
    ASSERT_GT(caller_sip_port, 0);
    ASSERT_NE(callee_sip_port, caller_sip_port);

    // ── Initialise a single PJSUA2 Endpoint ──────────────────────────────────
    pj::Endpoint ep;
    ASSERT_NO_THROW(ep.libCreate());

    // Mirror main.cpp's EpConfig exactly.
    pj::EpConfig ep_cfg;
    ep_cfg.logConfig.level        = 0;  // suppress PJSIP log noise in test output
    ep_cfg.logConfig.consoleLevel = 0;
    ep_cfg.medConfig.noVad        = true;
    ep_cfg.medConfig.ecTailLen    = 0;   // no echo canceller
    ep_cfg.medConfig.clockRate    = 48000;

    ASSERT_NO_THROW(ep.libInit(ep_cfg));

    // ── Transport: callee ─────────────────────────────────────────────────────
    pj::TransportConfig callee_tp_cfg;
    callee_tp_cfg.port = static_cast<unsigned>(callee_sip_port);
    callee_tp_cfg.boundAddress = "127.0.0.1";
    pj::TransportId callee_tp = -1;
    ASSERT_NO_THROW(callee_tp = ep.transportCreate(PJSIP_TRANSPORT_UDP, callee_tp_cfg));
    (void)callee_tp;

    // ── Transport: caller ─────────────────────────────────────────────────────
    pj::TransportConfig caller_tp_cfg;
    caller_tp_cfg.port = static_cast<unsigned>(caller_sip_port);
    caller_tp_cfg.boundAddress = "127.0.0.1";
    pj::TransportId caller_tp = -1;
    ASSERT_NO_THROW(caller_tp = ep.transportCreate(PJSIP_TRANSPORT_UDP, caller_tp_cfg));
    (void)caller_tp;

    try {
        ep.libStart();
    } catch (pj::Error &e) {
        FAIL() << "ep.libStart() threw: " << e.info();
    }

    // Mirror main.cpp: setNullDev() then codecSetPriority(), both after libStart().
    ASSERT_NO_THROW(ep.audDevManager().setNullDev());

    // ── Enumerate registered codecs (diagnostic) ─────────────────────────────
    {
        pj::CodecInfoVector2 codecs = ep.codecEnum2();
        bool found_opus = false;
        for (const auto &ci : codecs) {
            if (ci.codecId.find("opus") != std::string::npos ||
                ci.codecId.find("OPUS") != std::string::npos)
                found_opus = true;
        }
        if (!found_opus) {
            std::string codec_list;
            for (const auto &ci : codecs) codec_list += " " + ci.codecId;
            FAIL() << "Opus codec not registered after libStart()."
                   << " Available codecs:" << codec_list
                   << "\nThis likely indicates a missing or broken libopus linkage.";
        }
    }

    // ── Codec priorities: keep only Opus ─────────────────────────────────────
    // Must be called after libStart() — the codec registry is populated by
    // the media endpoint during start.  This is the path most likely to expose
    // a bad Opus link: if libopus.a is missing or has unresolved symbols,
    // the encoder/decoder factory will fail to register and this call will
    // throw or the subsequent SDP negotiation will exclude Opus entirely.
    //
    // Use the same codec ID as main.cpp: opus/48000/2.
    ASSERT_NO_THROW(ep.codecSetPriority("opus/48000/2", 255));
    // Deprioritise everything else so SDP lists only Opus.
    for (const auto &id : {"PCMU/8000",  "PCMA/8000",
                            "G722/16000", "G729/8000",
                            "speex/8000", "speex/16000",
                            "iLBC/8000",  "GSM/8000"}) {
        try { ep.codecSetPriority(id, 0); } catch (...) { /* not built in */ }
    }

    // Use a nested scope so all accounts and calls are destroyed before
    // ep.libDestroy().  SilentAudioPort's destructor calls
    // pjmedia_conf_remove_port, which requires the conference bridge to still
    // be alive — that bridge is torn down inside libDestroy().
    bool media_ok  = false;
    std::string codec_name;
    {
        // ── Register callee account (local, no registrar) ─────────────────────
        pj::AccountConfig callee_acfg;
        callee_acfg.idUri = "sip:callee@127.0.0.1:" + std::to_string(callee_sip_port);
        callee_acfg.regConfig.registrarUri = "";  // no registration needed
        callee_acfg.sipConfig.proxies      = {};

        CallState call_state;
        CalleeAccount callee_acc(call_state);
        ASSERT_NO_THROW(callee_acc.create(callee_acfg));

        // ── Register caller account ───────────────────────────────────────────
        pj::AccountConfig caller_acfg;
        caller_acfg.idUri = "sip:caller@127.0.0.1:" + std::to_string(caller_sip_port);
        caller_acfg.regConfig.registrarUri = "";
        caller_acfg.sipConfig.proxies      = {};

        pj::Account caller_acc;
        ASSERT_NO_THROW(caller_acc.create(caller_acfg));

        // ── Make the call ─────────────────────────────────────────────────────
        CallerCall caller_call(caller_acc, call_state);

        pj::CallOpParam call_op(true);
        call_op.opt.audioCount = 1;
        call_op.opt.videoCount = 0;

        std::string callee_uri =
            "sip:callee@127.0.0.1:" + std::to_string(callee_sip_port);

        ASSERT_NO_THROW(caller_call.makeCall(callee_uri, call_op));

        // ── Wait for media to become active (up to 5 s) ───────────────────────
        media_ok = call_state.WaitForMediaActive(std::chrono::seconds(5));
        if (media_ok) {
            // Let a few RTP frames flow (≈ 500 ms) so encoder/decoder are exercised.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            codec_name = call_state.codec_name;
        }

        // ── Hang up ───────────────────────────────────────────────────────────
        try {
            pj::CallOpParam hangup_op;
            hangup_op.statusCode = PJSIP_SC_OK;
            caller_call.hangup(hangup_op);
        } catch (...) {}

        call_state.WaitForDisconnected(std::chrono::seconds(5));

        // Shut down accounts while the endpoint (and conference bridge) is still
        // alive.  This triggers call / port cleanup before libDestroy().
        callee_acc.shutdown();
        caller_acc.shutdown();

        // Give PJSIP a moment to process the shutdown events.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // CallerCall, CallerAccount, CalleeAccount (and their owned
        // SilentAudioPort objects) are destroyed here at end of scope —
        // while the conference bridge is still live.
    }

    // ── Assertions (outside the scope so ep.libDestroy runs last) ────────────
    EXPECT_TRUE(media_ok) << "Media never became active — possible Opus link failure";
    if (media_ok) {
        std::string lc = codec_name;
        std::transform(lc.begin(), lc.end(), lc.begin(), ::tolower);
        EXPECT_NE(lc.find("opus"), std::string::npos)
            << "Unexpected codec: " << codec_name;
    }

    ep.libDestroy();
}
