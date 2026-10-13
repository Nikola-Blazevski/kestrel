"""Tiny MJPEG server so the annotated feed can be viewed headless from any browser."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>doc-cam-pi</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px monospace;display:flex;
flex-direction:column;align-items:center}img{max-width:100%;height:auto}
#d{padding:8px}</style></head><body>
<img src="/stream.mjpg"><div id="d"></div>
<script>
async function poll(){try{const r=await fetch('/data.json');const j=await r.json();
const m=j.mag;document.getElementById('d').textContent=m?
`${m.sensor}  HDG ${m.heading.toFixed(1)}°  X ${m.x.toFixed(1)}  Y ${m.y.toFixed(1)}  Z ${m.z.toFixed(1)} µT  |B| ${m.magnitude.toFixed(1)} µT  FPS ${j.fps.toFixed(1)}`
:`no magnetometer  FPS ${j.fps.toFixed(1)}`;}catch(e){}setTimeout(poll,500)}poll();
</script></body></html>""".encode("utf-8")


class StreamHub:
    def __init__(self, quality=75):
        self.quality = quality
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self.clients = 0
        self.data = {}

    def publish(self, frame, data=None):
        if data is not None:
            self.data = data
        if self.clients == 0:          # don't spend CPU encoding for nobody
            return
        ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self._cond.notify_all()

    def wait(self, last_seq, timeout=2.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._seq, self._jpeg


def start_server(hub, port, host="0.0.0.0"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html", PAGE)
            elif self.path == "/data.json":
                self._send(200, "application/json", json.dumps(hub.data).encode())
            elif self.path.startswith("/stream.mjpg"):
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            hub.clients += 1
            seq = 0
            try:
                while True:
                    seq, jpeg = hub.wait(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode() +
                                     b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.clients -= 1

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
