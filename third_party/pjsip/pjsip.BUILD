load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all_srcs",
    srcs = glob(
        ["**"],
        exclude = ["**/.git/**"],
    ),
    visibility = ["//visibility:private"],
)

# PJSIP 2.14.1 — stripped down to SIP signalling + Opus audio only.
#
# What we keep vs remove:
#   KEEP    Opus                 — the only codec; Asterisk upstream speaks Opus
#   REMOVE  G.711 µ-law / A-law  — disabled at compile time (PJMEDIA_HAS_G711_CODEC=0)
#   KEEP    pjmedia conference bridge + custom AudioMediaPort
#   KEEP    null audio device  — modem-server calls setNullDev(); no ALSA needed
#   REMOVE  ALSA / PortAudio   — --disable-sound → no libasound2 runtime dep
#   REMOVE  Speex AEC          -- we set ecTailLen = 0, so no echo canceller
#   REMOVE  GSM / Speex / iLBC / L16 / G.722 / G.722.1 codecs
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
    configure_in_place = True,
    configure_options = [
        # ── codec stripping ─────────────────────────────────────────────────
        "--disable-speex-aec",  # Speex echo canceller (we set ecTailLen=0)
        "--disable-g711-codec",  # G.711 µ-law / A-law
        "--disable-gsm-codec",  # GSM (not used)
        "--disable-speex-codec",  # Speex (disabled; not registered, cannot set priority)
        "--disable-ilbc-codec",  # iLBC (disabled; not registered, cannot set priority)
        "--disable-l16-codec",  # Linear PCM 16-bit (not used)
        "--disable-g722-codec",  # G.722 wideband (not used)
        "--disable-g7221-codec",  # G.722.1 (not used)
        # ── audio device stripping ───────────────────────────────────────────
        # Disables all sound backends (ALSA, PortAudio).
        # pjsua_set_null_snd_dev() remains fully functional: it uses
        # pjmedia_master_port_create() internally, not the audio device layer.
        "--disable-sound",
        # ── other reductions ─────────────────────────────────────────────────
        "--disable-video",
        "--disable-openssl",
        "--disable-resample",
        # Prevent configure from detecting system OpenSSL even when it is
        # installed (e.g. on Ubuntu CI runners). Without these, configure
        # enables EVP-based SHA256 auth and SRTP RNG, producing undefined
        # references to EVP_*/RAND_bytes/ERR_* at link time.
        "ac_cv_lib_ssl_SSL_CTX_new=no",
        "ac_cv_header_openssl_ssl_h=no",
        # ── force Opus detection to use the Bazel-staged @opus dep ───────────
        # PJSIP's configure probes for Opus via pkg-config or by compiling a
        # test binary with -lopus. Neither works because the Bazel-built Opus
        # lives under $EXT_BUILD_DEPS, which configure's subshell doesn't see.
        # Caching these autoconf results skips the probe and lets the compiler
        # find the headers/lib via CFLAGS/LDFLAGS set in the env attribute.
        "ac_cv_lib_opus_opus_encoder_create=yes",
        "ac_cv_header_opus_opus_h=yes",
        # PJSIP's configure gates PJMEDIA_HAS_OPUS_CODEC on opus_repacketizer_get_size
        # (not opus_encoder_create).  Cache it so configure sets the flag to 1.
        # Without this, config_auto.h emits PJMEDIA_HAS_OPUS_CODEC=0 and no codecs
        # register at runtime even though everything links fine.
        "ac_cv_lib_opus_opus_repacketizer_get_size=yes",
        # Include -std=c++17 here to match the --action_env=CXXFLAGS=-std=c++17
        # set in .bazelrc. Without it, this override would strip the C++ standard
        # flag from PJSUA2's C++ compilation units.
        # NOTE: do not append additional flags with spaces here — rules_foreign_cc
        # passes each configure_options entry as a separate shell word, so a value
        # like "CXXFLAGS=-std=c++17 -O2" would be split and -O2 would be passed as
        # a standalone argument, causing aconfigure to error with
        # "unrecognized option: -O2".
        "CXXFLAGS=-std=c++17",
        # CFLAGS and LDFLAGS are set via the env attribute below so they can
        # reference $$EXT_BUILD_DEPS (rules_foreign_cc supports multi-word values
        # in env but not in configure_options, which are each a single shell word).
    ],
    # CFLAGS: -O2 optimization. Opus headers come from @opus//:opus staged into
    # $EXT_BUILD_DEPS/include by rules_foreign_cc.
    # LDFLAGS: overrides the gcc-style linker flags rules_foreign_cc injects
    # (e.g. -Wl,-S, -fuse-ld=gold) that /usr/bin/ld rejects when PJSIP links
    # internal test binaries directly with ld.
    env = {
        "CFLAGS": "-O2 -I$$EXT_BUILD_DEPS/include -DPJMEDIA_HAS_G711_CODEC=0",
        "LDFLAGS": "-L$$EXT_BUILD_DEPS/lib",
    },
    # The default install_prefix is the rule name ("pjsip"), which clashes with
    # PJSIP's own "pjsip/" build subdirectory, causing "make install" to cp
    # files to themselves. Use a name that does not exist as a PJSIP source dir.
    install_prefix = "pjproject_install",
    lib_source = ":all_srcs",
    linkopts = [
        "-lm",
        "-lrt",
        "-lpthread",
    ],
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
        "libwebrtc-x86_64-unknown-linux-gnu.a",
    ],
    # make install only copies the main PJSIP libs; bundled third-party libs
    # (srtp, webrtc, etc.) remain in third_party/lib/.  Copy them to the install
    # prefix so out_static_libs can find them.
    # libopus.a is NOT in out_static_libs: rules_foreign_cc propagates @opus//:opus's
    # CcInfo to the final binary, so the linker picks it up automatically.
    postfix_script = "cp $$BUILD_TMPDIR/third_party/lib/*.a $$BUILD_TMPDIR/$$INSTALL_PREFIX/lib/",
    # Use 'lib' instead of 'all' to build only library directories,
    # avoiding test binaries that fail when ld receives gcc-style LDFLAGS.
    targets = [
        "dep",
        "lib",
        "install",
    ],
    visibility = ["//visibility:public"],
    # Use opus_headers only — not @opus//:opus — so that rules_foreign_cc stages the
    # Opus public headers into $EXT_BUILD_DEPS/include/ for PJSIP's make build, but
    # does NOT propagate @opus//:opus's full CcInfo (which includes SIMD sub-targets
    # silk_x86_rtcd/celt_x86_rtcd that have unresolved _c fallback symbols at link time).
    #
    # The opus_headers target provides the headers with includes=["include"], so they
    # land at $EXT_BUILD_DEPS/include/opus.h etc.  PJSIP's opus.c is patched
    # (opus_include_path.patch) to use <opus.h> instead of <opus/opus.h>.
    #
    # libopus.a itself must be linked by the final binary.  Add @opus//:opus
    # directly to each cc_binary/cc_test that depends on @pjsip — see
    # BUILD.bazel and tests/BUILD.bazel.
    #
    # NOTE: we use @opus//:opus (not @opus//:opus_headers) because
    # opus_headers has visibility=private and cannot be referenced from the
    # @pjsip external repo.  @opus//:opus is public and its deps include
    # opus_headers, so the same headers land in $EXT_BUILD_DEPS/include/.
    deps = [
        "@libuuid",
        "@opus//:opus",
    ],
)
