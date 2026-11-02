"""The on-board explainer: a llama-server that exists only while someone needs it. Stdlib only.

start() takes a frozen decision snapshot and an audience. The page gets explain_prompt.facts_text at once, then a background thread
starts the server, prewarms it, streams the answer over the existing /state polling, and runs explain_check on the result. An answer
that fails the check, or anything else that goes wrong, leaves the facts on the page with one plain line saying why. Nothing here can
take the dashboard down: every failure ends as text. The thread and the process are gone again idle_s after the last answer.

There is one job at a time and every viewer sees the same one. Only the audience enum and the snapshot reach the model.

Why raw sockets and select below: each wait has to notice a stop button, a dead process, a hang and the temperature, and http.client
cannot resume a read after a timeout.
"""
from __future__ import annotations

import glob
import http.client
import json
import math
import os
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import explain_check as ec
import explain_prompt as ep

HOST = "127.0.0.1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO / "device" / "llm" / "bin" / "llama-server"
DEFAULT_MODEL = REPO / "data" / "llm" / "twinread-explain-qwen3-v1-Q4_0-embQ8.gguf"
DEFAULT_PORT = 1790
DEFAULT_THREADS = 3
DEFAULT_IDLE_S = 120.0
# Limits on the hottest thermal zone; device/README.md gives the reasoning and results/uno_q/onboard/explainer/ the measurements.
START_MAX_C = 68.0
STOP_MAX_C = 72.0

START_TIMEOUT_S = 60.0     # measured 3.4 to 4.5 s to load
PREPARE_TIMEOUT_S = 180.0  # measured about 20 s to prewarm
FIRST_WORD_TIMEOUT_S = 120.0  # measured 9 to 11 s
TOKEN_GAP_TIMEOUT_S = 30.0    # measured about 0.3 s between tokens
ANSWER_TIMEOUT_S = 300.0   # measured 31 to 37 s for a whole answer
HEAT_EVERY_S = 2.0
SELECT_S = 0.25
IDLE_POLL_S = 2.0          # a warm, idle server is looked at this often, so a crash is reaped and shown as off
STOP_GRACE_S = 3.0

ACTIVE = ("starting", "preparing", "thinking", "writing", "checking")
READING_ITSELF = "this is the reading itself."
# What each failed check means, most important first. The reader gets the first that applies.
CHECK_WORDS = (("numbers_match", "got a number wrong"), ("class_correct", "named the wrong tissue class"),
               ("no_wrong_class", "named a tissue class that is not in this result"), ("heads_correct", "got the two heads wrong"),
               ("status_correct", "got the review flag wrong"), ("no_forbidden", "said something it cannot know from this result"),
               ("complete", "stopped before the end"), ("length_ok", "came out too long or too short"),
               ("format_ok", "came out in the wrong format"), ("framed_as_model", "left out that this is the model's reading"),
               ("vocabulary_ok", "used a word outside the list it is allowed"))


class _Cancelled(Exception):
    """The stop button or a slide change: the page already says so."""


class _Fallback(Exception):
    """The answer cannot be used. The message is the one plain line the reader sees beside the facts."""


class _ServerDied(Exception):
    pass


class _Hung(Exception):
    pass


class _TooLong(Exception):
    """The stream ran past max_tokens: it is cut there, whatever the server does."""


def hottest_c():
    """The hottest thermal zone in C, or None where there are no zones."""
    temps = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(p) as f:
                temps.append(int(f.read()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def plain_failure(result: dict) -> str:
    """The reader's one line for an answer that failed explain_check."""
    for name, words in CHECK_WORDS:
        if not result["checks"].get(name, True):
            return f"The language model's wording {words}, so {READING_ITSELF}"
    return f"The language model's wording failed the check, so {READING_ITSELF}"


class _Response:
    """An HTTP response arriving in pieces: status line, headers, then a plain or chunked body. feed() returns the body bytes so far."""

    def __init__(self):
        self.status, self.done = None, False
        self._head, self._buf = b"", b""
        self._chunked, self._length, self._got, self._left, self._crlf = False, None, 0, 0, 0

    def feed(self, data: bytes) -> bytes:
        if self.status is None:
            self._head += data
            end = self._head.find(b"\r\n\r\n")
            if end < 0:
                return b""
            head, data = self._head[:end].decode("latin-1"), self._head[end + 4:]
            lines = head.split("\r\n")
            self.status = int(lines[0].split()[1])
            fields = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
            self._chunked = "chunked" in fields.get("transfer-encoding", "").lower()
            if not self._chunked and "content-length" in fields:
                self._length = int(fields["content-length"])
                self.done = self._length == 0
        if not self._chunked:
            self._got += len(data)
            if self._length is not None and self._got >= self._length:
                self.done = True
            return data
        return self._dechunk(data)

    def _dechunk(self, data: bytes) -> bytes:
        self._buf += data
        out = bytearray()
        while not self.done:
            if self._crlf:  # the line end after a chunk's data
                n = min(self._crlf, len(self._buf))
                self._buf, self._crlf = self._buf[n:], self._crlf - n
                if self._crlf:
                    break
            if self._left:
                part = self._buf[:self._left]
                out += part
                self._buf, self._left = self._buf[len(part):], self._left - len(part)
                if self._left:
                    break
                self._crlf = 2
                continue
            i = self._buf.find(b"\r\n")
            if i < 0:
                break
            size = int(self._buf[:i].split(b";")[0], 16)
            self._buf = self._buf[i + 2:]
            if size == 0:
                self.done = True
            self._left = size
        return bytes(out)


class Job:
    """One explanation. Fields change only under the Explainer's lock."""

    def __init__(self, job_id, snapshot, audience, body, facts, about, picture):
        self.id, self.snapshot, self.audience, self.body, self.facts = job_id, snapshot, audience, body, facts
        self.about, self.picture = about, picture
        self.t0 = time.monotonic()
        self.phase, self.source, self.text, self.note, self.checked = "starting", "reading", facts, None, None
        self.raw, self.tokens, self.cancel = "", 0, False
        self.stats = {}
        self.rejected, self.reported = None, False

    @property
    def active(self) -> bool:
        return self.phase in ACTIVE

    def elapsed(self) -> float:
        return round(time.monotonic() - self.t0, 2)

    def public(self) -> dict:
        out = {"id": self.id, "phase": self.phase, "audience": self.audience, "source": self.source, "text": self.text, "note": self.note,
               "checked": self.checked, "about": self.about, "elapsed_s": int(self.elapsed()), "words": len(self.raw.split())}
        if not self.active:
            out["stats"] = self.stats
        return out


class Explainer:
    """start, stop, dismiss and state may be called from any thread and never wait on the model."""

    def __init__(self, binary=DEFAULT_BINARY, model=DEFAULT_MODEL, port=DEFAULT_PORT, threads=DEFAULT_THREADS, idle_s=DEFAULT_IDLE_S,
                 start_max_c=START_MAX_C, stop_max_c=STOP_MAX_C, temp=hottest_c, on_change=None, start_timeout_s=START_TIMEOUT_S,
                 prepare_timeout_s=PREPARE_TIMEOUT_S, first_word_timeout_s=FIRST_WORD_TIMEOUT_S, token_gap_timeout_s=TOKEN_GAP_TIMEOUT_S,
                 answer_timeout_s=ANSWER_TIMEOUT_S, heat_every_s=HEAT_EVERY_S):
        self.binary, self.model, self.port, self.threads = Path(binary).resolve(), Path(model).resolve(), port, threads  # the server runs from REPO
        self.idle_s, self.start_max_c, self.stop_max_c = float(idle_s), float(start_max_c), float(stop_max_c)
        self._temp, self._on_change = temp, on_change
        self.start_timeout_s, self.prepare_timeout_s = start_timeout_s, prepare_timeout_s
        self.first_word_timeout_s, self.token_gap_timeout_s = first_word_timeout_s, token_gap_timeout_s
        self.answer_timeout_s, self.heat_every_s = answer_timeout_s, heat_every_s
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._job, self._queued, self._thread = None, None, None
        self._seq, self._closing = 0, False
        self._server, self._warm, self._proc, self._log = "off", False, None, None
        self._idle_deadline = 0.0
        self._heat_t, self._peak_c = 0.0, None

    # -- what the page may ask --------------------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        job = self._job
        return job is not None and job.active

    def state(self) -> dict:
        with self._lock:
            job = self._job
            out = {"server": self._server, "start_max_c": self.start_max_c, "stop_max_c": self.stop_max_c, "job": job.public() if job else None}
            if self._server == "ready" and not (job and job.active):
                out["unload_in_s"] = max(0, math.ceil(self._idle_deadline - time.monotonic()))
        return out

    def picture(self, job_id=None) -> bytes:
        with self._lock:
            job = self._job
            return job.picture if job and (job_id is None or job.id == job_id) else b""

    def missing_files(self) -> list:
        out = []
        if not (self.binary.is_file() and os.access(self.binary, os.X_OK)):
            out.append(self.binary.name)
        if not self.model.is_file():
            out.append(self.model.name)
        return out

    def hint(self, temp_c=None):
        """Said beside the button before anyone presses it: what will happen instead of an answer. None if nothing is in the way."""
        if self.missing_files():
            return "The language model's files are not on this board, so Explain shows the reading itself."
        if temp_c is not None and temp_c > self.start_max_c:
            return f"The board is at {temp_c:.1f} °C. The language model starts below {self.start_max_c:g} °C, so for now Explain shows the reading itself."
        return None

    def _refusal(self):
        missing = self.missing_files()
        if missing:
            return f"The language model's files are not on this board (missing {' and '.join(missing)}), so {READING_ITSELF}"
        t = self._temp()
        if t is not None and t > self.start_max_c:
            return f"The board is at {t:.1f} °C and the language model starts only below {self.start_max_c:g} °C, so {READING_ITSELF}"
        return None

    # -- what the buttons do ----------------------------------------------------------------------------------------------
    def start(self, snapshot, audience="doctor", about=None, picture=b"") -> bool:
        """False if an answer is already being written. Raises ep.NoDecision or ValueError, with nothing changed, on bad input."""
        facts = ep.facts_text(snapshot)
        body = ep.chat_request(snapshot, audience)
        refusal = self._refusal()
        with self._cv:
            if self._closing or (self._job is not None and self._job.active):
                return False
            self._seq += 1
            job = Job(self._seq, snapshot, audience, body, facts, about or {}, picture)
            self._job = job
            self._idle_deadline = time.monotonic() + self.idle_s
            if refusal:
                self._finish_locked(job, "fallback", note=refusal)
            else:
                self._queued = job
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name="explainer", daemon=True)
                self._thread.start()
            self._cv.notify_all()
        if refusal:
            self._report(job)
        return True

    def stop(self) -> bool:
        """The stop button: the page gets the facts at once. The server stays warm for a next click if it had finished preparing."""
        with self._cv:
            job = self._job
            if job is None or not job.active:
                return False
            job.cancel = True
            self._finish_locked(job, "stopped", note=f"Stopped, so {READING_ITSELF}")
            self._idle_deadline = time.monotonic() + self.idle_s
            self._cv.notify_all()
        self._report(job)
        self._changed()
        return True

    def dismiss(self):
        """Close the panel, stopping any answer in progress: a slide changed by hand, or the close button."""
        with self._cv:
            job = self._job
            if job is None:
                return
            was_active = job.active
            if was_active:
                job.cancel = True
                self._finish_locked(job, "stopped", note=f"Stopped, so {READING_ITSELF}")
                self._idle_deadline = time.monotonic() + self.idle_s
            self._job = None
            self._cv.notify_all()
        if was_active:
            self._report(job)
            self._changed()

    def shutdown(self):
        """The app is exiting: no server may outlive it."""
        with self._cv:
            self._closing = True
            job = self._job
            if job is not None and job.active:
                job.cancel = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(STOP_GRACE_S + 5.0)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(5)

    # -- the worker thread ------------------------------------------------------------------------------------------------
    def _worker(self):
        clean = False
        try:
            while True:
                job = self._next_job()
                if job is None:
                    clean = True
                    return
                self._run_job(job)
        except Exception as e:  # noqa: BLE001 - a dead worker must not leave a job at "starting" for ever
            print(json.dumps({"event": "explain_worker_died", "error": repr(e)}), flush=True)
        finally:
            if not clean:
                with self._cv:
                    if self._thread is threading.current_thread():
                        self._thread = None
                    self._queued = None
                    if self._job is not None and self._job.active:
                        self._finish_locked(self._job, "fallback", note=f"The language model failed, so {READING_ITSELF}")

    def _next_job(self):
        """The next job, or None once idle_s has passed with nothing to do: the server is stopped first, so the thread that started
        it is the last thing to go (a server's PDEATHSIG follows that thread)."""
        while True:
            with self._cv:
                while self._queued is None and not self._closing and time.monotonic() < self._idle_deadline:
                    self._cv.wait(max(0.0, min(IDLE_POLL_S, self._idle_deadline - time.monotonic())))
                    self._note_dead_server_locked()
                if self._queued is not None and not self._closing:
                    job, self._queued = self._queued, None
                    if job.cancel:
                        continue
                    return job
            self._safe_stop()
            with self._cv:
                if not self._closing and (self._queued is not None or time.monotonic() < self._idle_deadline):
                    continue  # a click landed while the server was stopping: its panel gets its full time
                if self._job is not None and not self._job.active:
                    self._job = None
                self._thread = None
                return None

    def _note_dead_server_locked(self):
        proc = self._proc
        if proc is not None and proc.poll() is not None:
            print(json.dumps({"event": "explain_server_gone", "returncode": proc.returncode}), flush=True)
            self._close_log()
            self._proc, self._warm, self._server = None, False, "off"

    def _run_job(self, job):
        self._heat_t, self._peak_c = 0.0, self._temp()
        try:
            self._ensure_server(job)
            self._generate(job)
        except _Cancelled:
            if not self._warm:  # it never finished preparing, so no next click can use it as it is, and it is burning CPU
                self._safe_stop()
        except _Fallback as e:
            self._to_facts(job, str(e))
        except Exception as e:  # noqa: BLE001 - the dashboard must outlive anything that happens here
            print(json.dumps({"event": "explain_error", "error": repr(e)}), flush=True)
            self._to_facts(job, f"The language model failed, so {READING_ITSELF}")  # first: no unchecked text may stay on the page
            self._safe_stop(now=True)
        finally:
            with self._cv:
                self._idle_deadline = time.monotonic() + self.idle_s
                job.stats["peak_c"] = self._peak_c
            self._report(job)
            self._changed()

    def _finish_locked(self, job, phase, note=None):
        if job.active:
            job.stats["done_s"] = job.elapsed()
        job.phase, job.note = phase, note
        if phase != "done":
            job.source, job.text = "reading", job.facts

    def _to_facts(self, job, note):
        with self._cv:
            if job.active:
                self._finish_locked(job, "fallback", note=note)

    def _report(self, job):
        with self._cv:
            if job.reported or job.active:
                return
            job.reported = True
        print(json.dumps({"event": "explain", "job": job.id, "audience": job.audience, "status": job.snapshot.get("status"), "phase": job.phase,
                          "checked": job.checked, "note": job.note, "stats": job.stats, "text": job.text, "rejected": job.rejected}), flush=True)

    def _changed(self):
        if self._on_change is not None:
            self._on_change()

    # -- the server -------------------------------------------------------------------------------------------------------
    def command(self) -> list:
        """The whole command line. setpriv makes the kernel kill the server when the thread that started it dies, inside or outside
        systemd; llama-server itself binds to loopback only."""
        t = str(self.threads)
        guard = ["setpriv", "--pdeathsig", "KILL"] if shutil.which("setpriv") else []
        return guard + ["nice", "-n", "10", str(self.binary), "-m", str(self.model), "-c", "1024", "-t", t, "-tb", t, "-np", "1",
                        "--host", HOST, "--port", str(self.port), "--no-webui", "--cache-ram", "0", "--ctx-checkpoints", "0",
                        "--reasoning", "off"]

    def _set(self, job=None, phase=None, server=None):
        with self._cv:
            if job is not None and phase is not None:
                if job.cancel or not job.active:
                    raise _Cancelled
                job.phase = phase
            if server is not None:
                self._server = server

    def _port_taken(self) -> bool:
        try:
            with socket.create_connection((HOST, self.port), timeout=0.3):
                return True
        except OSError:
            return False

    def _ensure_server(self, job):
        with self._cv:
            self._note_dead_server_locked()
        if self._proc is not None and self._warm:
            job.stats["cold"] = False
            return
        job.stats["cold"] = True
        self._stop_server()
        if self._port_taken():
            raise _Fallback(f"Port {self.port} is already in use, so the language model cannot start and {READING_ITSELF}")
        t0 = time.monotonic()
        self._set(job, "starting", "starting")
        try:
            self._log = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(self.command(), cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log,
                                          env=dict(os.environ, OMP_WAIT_POLICY="passive"))
        except OSError as e:
            self._close_log()
            self._set(server="off")
            raise _Fallback(f"The language model could not be started ({e.strerror}), so {READING_ITSELF}") from e
        try:
            self._wait_healthy(job, t0)
            job.stats["load_s"] = round(time.monotonic() - t0, 2)
            t1 = time.monotonic()
            self._set(job, "preparing", "preparing")
            for _ in self._exchange(job, ep.prewarm_request(), self.prepare_timeout_s, self.prepare_timeout_s, self.prepare_timeout_s):
                pass
            job.stats["prepare_s"] = round(time.monotonic() - t1, 2)
        except _ServerDied as e:
            code = self._stop_server()
            raise _Fallback(f"The language model did not start (exit code {code}), so {READING_ITSELF}") from e
        except _Hung as e:
            self._stop_server(now=True)
            raise _Fallback(f"The language model did not get ready in time, so {READING_ITSELF}") from e
        self._warm = True
        self._set(server="ready")

    def _wait_healthy(self, job, t0):
        while True:
            self._watch(job)
            try:
                c = http.client.HTTPConnection(HOST, self.port, timeout=1)
                c.request("GET", "/health")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    return
            except (OSError, http.client.HTTPException):
                pass
            if time.monotonic() - t0 > self.start_timeout_s:
                raise _Hung
            time.sleep(0.2)

    def _stop_server(self, now=False):
        """Stop and reap the server. Returns its exit code, or None if there was none."""
        proc = self._proc
        if proc is None:
            return None
        self._set(server="stopping")
        if proc.poll() is None:
            if now:
                proc.kill()
            else:
                proc.terminate()
            try:
                proc.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:  # a stopped or hung process does not act on SIGTERM
                proc.kill()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:  # stuck in the kernel: nothing more can be done from here, and the page must go on
                    print(json.dumps({"event": "explain_server_will_not_die", "pid": proc.pid}), flush=True)
        self._close_log()
        self._proc, self._warm = None, False
        self._set(server="off")
        return proc.returncode

    def _safe_stop(self, now=False):
        """_stop_server for the paths that must go on whatever it does."""
        try:
            self._stop_server(now=now)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"event": "explain_stop_failed", "error": repr(e)}), flush=True)
            with self._cv:
                self._proc, self._warm, self._server = None, False, "off"

    def _close_log(self):
        log, self._log = self._log, None
        if log is None:
            return
        try:
            log.seek(max(0, os.fstat(log.fileno()).st_size - 300))
            tail = log.read().decode("utf-8", "replace")
            print(json.dumps({"event": "explain_server_log_tail", "tail": tail}), flush=True)
        except (OSError, ValueError):
            pass
        log.close()

    def _watch(self, job):
        """Everything that can end a wait, checked often: the stop button, a dead server, and the temperature."""
        if job.cancel or self._closing:
            raise _Cancelled
        if self._proc is None or self._proc.poll() is not None:
            raise _ServerDied
        now = time.monotonic()
        if now - self._heat_t >= self.heat_every_s:
            self._heat_t = now
            t = self._temp()
            if t is not None:
                self._peak_c = t if self._peak_c is None else max(self._peak_c, t)
                if t > self.stop_max_c:
                    self._stop_server(now=True)
                    raise _Fallback(f"The board reached {t:.1f} °C, over the {self.stop_max_c:g} °C limit, so the language model was stopped and "
                                    f"{READING_ITSELF}")

    def _exchange(self, job, body, first_s, gap_s, total_s):
        """POST one request; yields pieces of the response body. Every wait is short, so _watch runs throughout."""
        payload = json.dumps(body).encode()
        head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {HOST}:{self.port}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
        try:
            sock = socket.create_connection((HOST, self.port), timeout=5)
        except OSError as e:
            raise _ServerDied from e
        with sock:
            resp = _Response()
            try:
                sock.sendall(head + payload)
                sock.setblocking(False)
                t0 = last = time.monotonic()
                got_any = False
                while not resp.done:
                    self._watch(job)
                    now = time.monotonic()
                    limit = min(t0 + total_s, last + (gap_s if got_any else first_s))
                    if now >= limit:
                        raise _Hung
                    if not select.select([sock], [], [], min(SELECT_S, limit - now))[0]:
                        continue
                    data = sock.recv(65536)
                    if not data:
                        break
                    last, got_any = time.monotonic(), True
                    piece = resp.feed(data)
                    if resp.status is not None and resp.status != 200:
                        raise _Fallback(f"The language model refused the request (HTTP {resp.status}), so {READING_ITSELF}")
                    if piece:
                        yield piece
            except OSError as e:  # ECONNRESET: the server went away mid-answer
                raise _ServerDied from e
            if not resp.done and resp.status is None:
                raise _ServerDied

    def _generate(self, job):
        self._set(job, "thinking")
        buf, finish, timings, ended = b"", None, {}, False
        stream = self._exchange(job, job.body, self.first_word_timeout_s, self.token_gap_timeout_s, self.answer_timeout_s)
        try:
            for piece in stream:
                buf += piece
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        ended = True
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise _Fallback(f"The language model reported an error, so {READING_ITSELF}")
                    choice = (event.get("choices") or [{}])[0]
                    finish = choice.get("finish_reason") or finish
                    timings = event.get("timings") or timings
                    self._token(job, (choice.get("delta") or {}).get("content") or "")
        except _ServerDied as e:
            self._stop_server()
            raise _Fallback(f"The language model stopped part way through, so {READING_ITSELF}") from e
        except _Hung as e:
            self._stop_server(now=True)
            raise _Fallback(f"The language model stopped answering, so {READING_ITSELF}") from e
        except _TooLong:
            finish, ended = "max_tokens", True
        finally:
            stream.close()  # hangs up at once: the server cancels the answer when its client goes away
        if not ended and finish is None:
            self._stop_server()
            raise _Fallback(f"The language model stopped part way through, so {READING_ITSELF}")
        self._check(job, finish, timings)

    def _token(self, job, piece):
        with self._cv:
            if job.cancel or not job.active:
                raise _Cancelled
            if not piece:
                return
            job.raw += piece
            job.tokens += 1
            if job.raw.strip():  # the first words replace the facts; until then the facts stay
                if job.source != "model":
                    job.stats["first_word_s"] = job.elapsed()
                job.source, job.text, job.phase = "model", job.raw.strip(), "writing"
            if job.tokens >= job.body["max_tokens"]:
                raise _TooLong

    def _check(self, job, finish, timings):
        with self._cv:
            if job.cancel or not job.active:
                raise _Cancelled
            job.phase = "checking"
            text = job.raw.strip()
        result = ec.check_snapshot(text, job.snapshot)
        failed = [k for k, ok in result["checks"].items() if not ok]
        with self._cv:
            if job.cancel or not job.active:
                raise _Cancelled
            job.stats.update(words=len(text.split()), tokens=job.tokens, finish=finish, passed=result["passed"], failed_checks=failed,
                             invented_claim=result["invented_claim"], gen_tps=timings.get("predicted_per_second"),
                             prompt_n=timings.get("prompt_n"), cache_n=timings.get("cache_n"), prompt_tps=timings.get("prompt_per_second"))
            if result["passed"]:
                job.source, job.text, job.checked = "model", text, True
                self._finish_locked(job, "done")
            else:
                job.rejected, job.checked = text, False
                self._finish_locked(job, "fallback", note=plain_failure(result))

