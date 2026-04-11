load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"], exclude = ["**/.git/**"]),
    visibility = ["//visibility:private"],
)

# PJSIP 2.14.1 — stripped down to SIP signalling + G.722 wideband audio only.
#
# What we keep vs remove:
#   KEEP    G.722 wideband (16 kHz)  — the only codec used (payload type 9)
#   KEEP    pjmedia conference bridge + custom AudioMediaPort
#   KEEP    null audio device  — modem-server calls setNullDev(); no ALSA needed
#   REMOVE  ALSA / PortAudio   — --disable-sound → no libasound2 runtime dep
#   REMOVE  Speex AEC          -- we set ecTailLen = 0, so no echo canceller
#   REMOVE  GSM / Speex / iLBC / L16 / G.711 / G.722.1 codecs
#   REMOVE  OpenSSL / TLS
#   REMOVE  Video pipeline
#
# NOTE: --disable-libyuv and --disable-libwebrtc are intentionally omitted.
# Those flags cause `make install` to fail because build.mak unconditionally
# lists their .a paths in APP_THIRD_PARTY_LIB_FILES regardless of the flags.
# Using only --disable-video prevents use of those libs while keeping them
# compilable so the install step succeeds.
#
# PJSIP installs static libs with a platform-specific arch suffix, e.g.
#   libpjsua2-x86_64-unknown-linux-gnu.a
# The postfix_script creates symlinks without the arch suffix so that
# out_static_libs can use portable names.
configure_make(
    name = "pjsip",
    lib_source = ":all_srcs",
    configure_in_place = True,
    configure_options = [
        # ── codec stripping ─────────────────────────────────────────────────
        "--disable-speex-aec",    # Speex echo canceller (we set ecTailLen=0)
        "--disable-gsm-codec",    # GSM (not used)
        "--disable-speex-codec",  # Speex (disabled; not registered, cannot set priority)
        "--disable-ilbc-codec",   # iLBC (disabled; not registered, cannot set priority)
        "--disable-l16-codec",    # Linear PCM 16-bit (not used)
        # G.722 is KEPT — it is our only codec (16 kHz wideband, PT=9)
        "--disable-g7221-codec",  # G.722.1 (not used)
        # ── audio device stripping ───────────────────────────────────────────
        # Disables all sound backends (ALSA, PortAudio).
        # pjsua_set_null_snd_dev() remains fully functional: it uses
        # pjmedia_master_port_create() internally, not the audio device layer.
        "--disable-sound",
        # ── other reductions ─────────────────────────────────────────────────
        "--disable-video",
        "--disable-openssl",
        # Prevent configure from detecting system OpenSSL even when it is
        # installed (e.g. on Ubuntu CI runners). Without these, configure
        # enables EVP-based SHA256 auth and SRTP RNG, producing undefined
        # references to EVP_*/RAND_bytes/ERR_* at link time.
        "ac_cv_lib_ssl_SSL_CTX_new=no",
        "ac_cv_header_openssl_ssl_h=no",
        "CFLAGS=-O2",
        # Include -std=c++17 here to match the --action_env=CXXFLAGS=-std=c++17
        # set in .bazelrc. Without it, this override would strip the C++ standard
        # flag from PJSUA2's C++ compilation units.
        # NOTE: do not append additional flags with spaces here — rules_foreign_cc
        # passes each configure_options entry as a separate shell word, so a value
        # like "CXXFLAGS=-std=c++17 -O2" would be split and -O2 would be passed as
        # a standalone argument, causing aconfigure to error with
        # "unrecognized option: -O2".
        "CXXFLAGS=-std=c++17",
        # Clear LDFLAGS: rules_foreign_cc injects gcc-style linker flags
        # (e.g. -Wl,-S, -fuse-ld=gold) that /usr/bin/ld rejects when PJSIP
        # links its internal test binaries directly with ld.
        "LDFLAGS=",
    ],
    # Use 'lib' instead of 'all' to build only library directories,
    # avoiding test binaries that fail when ld receives gcc-style LDFLAGS.
    targets = ["dep", "lib", "install"],
    # The default install_prefix is the rule name ("pjsip"), which clashes with
    # PJSIP's own "pjsip/" build subdirectory, causing "make install" to cp
    # files to themselves. Use a name that does not exist as a PJSIP source dir.
    install_prefix = "pjproject_install",
    # make install only copies the main PJSIP libs; bundled third-party libs
    # (srtp, resample, webrtc, etc.) remain in third_party/lib/.  Copy them
    # to the install prefix so out_static_libs can find them.
    postfix_script = "cp $$BUILD_TMPDIR/third_party/lib/*.a $$BUILD_TMPDIR/$$INSTALL_PREFIX/lib/",
    #   libpjsua2-x86_64-unknown-linux-gnu.a
    # We target linux/x86_64 only, so hardcode the suffix directly.
    out_static_libs = [
        # pjsua2 C++ wrapper (needed by main.cpp and modem_client.cpp)
        "libpjsua2-x86_64-unknown-linux-gnu.a",
        # pjsua C API
        "libpjsua-x86_64-unknown-linux-gnu.a",
        # SIP stack
        "libpjsip-ua-x86_64-unknown-linux-gnu.a",
        "libpjsip-simple-x86_64-unknown-linux-gnu.a",
        "libpjsip-x86_64-unknown-linux-gnu.a",
        # Media: codec framework + conference bridge + audio device manager
        "libpjmedia-codec-x86_64-unknown-linux-gnu.a",
        "libpjmedia-videodev-x86_64-unknown-linux-gnu.a",
        "libpjmedia-x86_64-unknown-linux-gnu.a",
        "libpjmedia-audiodev-x86_64-unknown-linux-gnu.a",
        # NAT traversal (required by pjmedia transport layer)
        "libpjnath-x86_64-unknown-linux-gnu.a",
        # Utility and base libraries
        "libpjlib-util-x86_64-unknown-linux-gnu.a",
        "libpj-x86_64-unknown-linux-gnu.a",
        # Bundled third-party libraries (copied by postfix_script from
        # third_party/lib/ since make install doesn't install them).
        "libsrtp-x86_64-unknown-linux-gnu.a",
        "libresample-x86_64-unknown-linux-gnu.a",
        "libwebrtc-x86_64-unknown-linux-gnu.a",
    ],
    # No -lasound: --disable-sound means modem-server has no ALSA runtime dep.
    # @libuuid//:libuuid is listed here so it propagates transitively to any
    # cc_binary (e.g. modem-server) that links against pjsip. libpj uses
    # uuid_generate/uuid_unparse from libuuid when the system uuid-dev is
    # present during configure; the Bazel dep satisfies those symbols at link
    # time without requiring a system libuuid.so at runtime.
    deps = ["@libuuid//:libuuid"],
    linkopts = [
        "-lm",
        "-lrt",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)
