#!/usr/bin/env python3
"""A minimal LAN client for the TinyTitan decode service.

This exists to answer one question that in-process tests cannot: **does the decode service's protocol
actually cross a real network between two machines?** The Swift tests in
`tests/TinyTitanDecodeService/DecodeTCPSocketTests.swift` prove the codec and the socket carry frames inside
one process; this proves the same frames survive a real TCP hop.

The frame format is taken from `DecodeFrameCodec` and is deliberately not re-derived here:
a 4-byte little-endian UInt32 payload length, then that many bytes of JSON. The 4 MiB ceiling is
`DecodeFrameCodec.maximumPayloadBytes` and is enforced on send so this client cannot send something the
service would reject as oversized.

Usage:
    python3 lan_decode_probe.py <host> <port> [--command JSON] [--timeout SECONDS]

The default command is `{"shutdown": {}}`, which is Swift's synthesized Codable form for the payload-free
`.shutdown` case. Pass `--command` to send something else; the raw JSON is used verbatim.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys

MAXIMUM_PAYLOAD_BYTES = 4 * 1024 * 1024


def encode_frame(payload: bytes) -> bytes:
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError(
            f"payload is {len(payload)} bytes, above the {MAXIMUM_PAYLOAD_BYTES} the service accepts"
        )
    return struct.pack("<I", len(payload)) + payload


def read_exactly(sock: socket.socket, count: int) -> bytes:
    """Read exactly `count` bytes or raise.

    A single `recv` is allowed to return short — more so over a network than over loopback — so a caller that
    reads once is testing buffering rather than the transport.
    """
    collected = bytearray()
    while len(collected) < count:
        chunk = sock.recv(count - len(collected))
        if not chunk:
            raise EOFError(f"peer closed after {len(collected)} of {count} bytes")
        collected.extend(chunk)
    return bytes(collected)


def read_frame(sock: socket.socket) -> bytes:
    (length,) = struct.unpack("<I", read_exactly(sock, 4))
    if length > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError(f"service announced {length} bytes, above the {MAXIMUM_PAYLOAD_BYTES} ceiling")
    return read_exactly(sock, length)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe a decode service across the LAN.")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument(
        "--command",
        default='{"shutdown": {}}',
        help='raw JSON to send as one frame (default: the payload-free .shutdown command)',
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--expect-reply",
        action="store_true",
        help="fail unless at least one frame comes back before the timeout",
    )
    arguments = parser.parse_args(argv)

    payload = arguments.command.encode("utf-8")
    # Validate locally so a malformed command is reported here rather than as a silent protocol error there.
    json.loads(payload)

    with socket.create_connection((arguments.host, arguments.port), timeout=arguments.timeout) as sock:
        sock.settimeout(arguments.timeout)
        print(f"connected to {arguments.host}:{arguments.port}", flush=True)
        sock.sendall(encode_frame(payload))
        print(f"sent {len(payload)} byte(s): {arguments.command}", flush=True)

        replies = 0
        try:
            while True:
                frame = read_frame(sock)
                replies += 1
                print(f"reply {replies}: {frame.decode('utf-8', 'replace')}", flush=True)
                if not arguments.expect_reply:
                    break
        except (EOFError, socket.timeout, TimeoutError) as end:
            # A clean exit for a probe that sent `.shutdown`: the service closing is the success signal, and
            # so is a timeout when no reply was expected. `--expect-reply` is what turns absence into failure.
            print(f"stream ended: {type(end).__name__}: {end}", flush=True)

    if arguments.expect_reply and replies == 0:
        print("FAIL: --expect-reply was given but no frame arrived", file=sys.stderr)
        return 1
    print(f"OK: sent one frame over the LAN, {replies} reply/replies", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
