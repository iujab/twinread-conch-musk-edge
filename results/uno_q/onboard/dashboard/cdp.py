"""Drive real headless Chromium over CDP: open a page, evaluate, click, screenshot, and collect console errors."""
import asyncio, base64, json, os, subprocess, tempfile, time, urllib.request
import websockets

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome")

class Page:
    def __init__(self, ws, proc, profile):
        self.ws, self.proc, self.profile, self._id = ws, proc, profile, 0
        self.errors, self._pending = [], {}
        self._reader = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self._pending:
                self._pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append(str(d.get("exception", {}).get("description") or d.get("text")))
            elif m.get("method") == "Runtime.consoleAPICalled" and m["params"]["type"] in ("error", "warning"):
                self.errors.append(" ".join(str(a.get("value")) for a in m["params"]["args"]))
            elif m.get("method") == "Log.entryAdded" and m["params"]["entry"]["level"] == "error":
                e = m["params"]["entry"]
                self.errors.append(f"{e.get('text')} {e.get('url', '')}")

    async def cmd(self, method, **params):
        self._id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[self._id] = fut
        await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        m = await asyncio.wait_for(fut, 20)
        if "error" in m:
            raise RuntimeError(f"{method}: {m['error']}")
        return m["result"]

    async def size(self, w, h, mobile=False):
        await self.cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=mobile)

    async def goto(self, url, settle=1.2):
        await self.cmd("Page.navigate", url=url)
        await asyncio.sleep(settle)

    async def js(self, expr):
        r = await self.cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    async def key(self, key="Tab", code=9):
        for t in ("keyDown", "keyUp"):
            await self.cmd("Input.dispatchKeyEvent", type=t, key=key, code=key, windowsVirtualKeyCode=code)

    async def shot(self, path, full=False):
        kw = {"captureBeyondViewport": True} if full else {}
        r = await self.cmd("Page.captureScreenshot", format="png", **kw)
        open(path, "wb").write(base64.b64decode(r["data"]))

    async def close(self):
        self._reader.cancel()
        await self.ws.close()
        self.proc.terminate()
        self.proc.wait(5)


async def open_page(port=9333):
    profile = tempfile.mkdtemp(prefix="cdp-")
    proc = subprocess.Popen([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--window-size=1280,720", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1))
            ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
            break
        except Exception:
            time.sleep(0.1)
    ws = await websockets.connect(ws_url, max_size=None)
    p = Page(ws, proc, profile)
    for dom in ("Page", "Runtime", "Log"):
        await p.cmd(f"{dom}.enable")
    return p
