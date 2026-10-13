"""
Wire format shared by server.py and remote.py.

Every message is a 4-byte big-endian length followed by the payload. After
connecting, the client sends a JSON hello and the server answers with a JSON
reply. From then on the client sends JPEG frames, and the server answers each
one with a JSON detections message.

    client -> {"hands": true, "faces": true, "num_hands": 2}
    server <- {"ok": true, "hands": true, "faces": true, "delegate": "GPU", "status": [...]}
    client -> <jpeg>
    server <- {"hands": [{"landmarks": [[x, y], ...], "label": "Left", "score": 0.9}],
               "faces": [{"box": [x, y, w, h], "score": 0.9}], "ms": 4.2}
"""

import json
import socket
import struct

from detectors import Detections, Face, Hand

DEFAULT_PORT = 8765
MAX_MSG = 8 * 1024 * 1024        # far bigger than any frame we send; guards against junk

_LEN = struct.Struct(">I")


def send_msg(sock, payload):
    sock.sendall(_LEN.pack(len(payload)) + payload)


def recv_msg(sock):
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if n > MAX_MSG:
        raise ConnectionError(f"message too large ({n} bytes)")
    return _recv_exact(sock, n)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def send_json(sock, obj):
    send_msg(sock, json.dumps(obj, separators=(",", ":")).encode())


def recv_json(sock):
    return json.loads(recv_msg(sock))


def detections_to_dict(d):
    return {"hands": [{"landmarks": [[round(x, 5), round(y, 5)] for x, y in h.landmarks],
                       "label": h.label, "score": round(h.score, 3)} for h in d.hands],
            "faces": [{"box": [round(v, 5) for v in f.box], "score": round(f.score, 3)}
                      for f in d.faces],
            "ms": round(d.ms, 2)}


def detections_from_dict(obj):
    return Detections(
        hands=[Hand(landmarks=[tuple(p) for p in h["landmarks"]], label=h["label"],
                    score=h["score"]) for h in obj.get("hands", [])],
        faces=[Face(box=tuple(f["box"]), score=f["score"]) for f in obj.get("faces", [])],
        ms=obj.get("ms", 0.0))


def parse_addr(addr):
    """'host', 'host:port' or '[v6addr]:port' -> (host, port)."""
    host, sep, port = addr.rpartition(":")
    if not sep or host.endswith(":"):      # no port, or a bare IPv6 address
        return addr.strip("[]"), DEFAULT_PORT
    return host.strip("[]"), int(port)


def tune(sock):
    # Small request/response messages: don't let Nagle hold them back
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
