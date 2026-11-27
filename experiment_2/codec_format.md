# LVC2 prototype

One self-contained stream per time slice; video size is sum of all five complete files. Magic LVC2, uint32 little-endian JSON header length, canonical compact JSON header, concatenated rANS payloads.
Header: version2, kind, channels, model checkpoint SHA256, original HW, coding/hyper shapes, stream lengths, rate_index0, padding0 (convolution implicit padding with final crop), payload SHA256. I2 main only; I3 main then hyper (decoder decodes hyper first).
Model weights are preshared and excluded from video rate; size/hash reported separately. CDF buffers are saved with weights; decoder preserves those CDFs. CPU independent-process exactness tested. Cross-device CDF determinism, hostile-input robustness beyond basic limits, deployment portability and formal security review NOT RUN.
Basic detection: magic, header length, version/model mismatch, payload length/checksum, output HW. Prototype is for trusted experiment files; not a hardened network parser.
