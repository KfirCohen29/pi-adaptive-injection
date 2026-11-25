"""Local HTTP server for serving mock resources and logging incoming requests."""

import functools
import pathlib
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

class LoggingHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that logs GET/POST requests to an external log file."""
    def __init__(self, *args, log_file: pathlib.Path, **kwargs):
        """Initialize the handler with a target request log file path.

        Args:
            *args: Positional arguments forwarded to `SimpleHTTPRequestHandler`.
            log_file: Path to the append-only text file recording HTTP requests.
            **kwargs: Keyword arguments forwarded to `SimpleHTTPRequestHandler`.
        """
        self.log_file = log_file
        super().__init__(*args, **kwargs)

    def _record_request(self, body: str = "") -> None:
        """Append a formatted single-line summary of the current HTTP request to disk.

        Args:
            body: Decoded request payload body (empty for GET requests).
        """
        entry = f"{self.command} {self.path} | {body}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def do_GET(self) -> None:
        """Log incoming GET requests and serve static files if `self.directory` exists."""
        self._record_request()
        if pathlib.Path(self.directory).exists():
            super().do_GET()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Read, decode, and log incoming POST request bodies, returning HTTP 200."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        self._record_request(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default stderr HTTP access logs."""
        pass

if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1])
    web_dir = root / "mock_web"
    log_file = root / "http_log.txt"
    handler = functools.partial(LoggingHandler, directory=str(web_dir), log_file=log_file)
    server = HTTPServer(("127.0.0.1", 9932), handler)
    server.serve_forever()