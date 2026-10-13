"""Serve the offline GRF replay GUI using only Python's standard library."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import webbrowser


class ReplayHandler(SimpleHTTPRequestHandler):
    """Serve MP4 byte ranges so browser scrubbing and decision jumps work."""

    def send_head(self):
        if not self.path.split("?", 1)[0].endswith(".mp4"):
            return super().send_head()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404, "Video not found")
            return None
        size = path.stat().st_size
        header = self.headers.get("Range")
        start, end = 0, size - 1
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                self.send_error(416, "Invalid byte range")
                return None
            if match.group(1):
                start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
            else:
                start = max(0, size - int(match.group(2)))
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        stream = path.open("rb")
        stream.seek(start)
        self._bytes_remaining = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        if not hasattr(self, "_bytes_remaining"):
            return super().copyfile(source, outputfile)
        while self._bytes_remaining:
            chunk = source.read(min(256 * 1024, self._bytes_remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            self._bytes_remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the viewer in a browser")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    handler = partial(ReplayHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/viewer/"
    print(f"GRF match viewer: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
