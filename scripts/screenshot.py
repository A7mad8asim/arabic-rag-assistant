"""Take the README screenshot: drive the running app in headless Edge (Chrome DevTools Protocol) at 2x.

    streamlit run app.py --server.port 8502      (in another terminal, with Ollama running)
    python scripts/screenshot.py docs/screenshot.png

Needs Microsoft Edge (or Chrome via --browser) and `pip install websockets`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request

import websockets

QUESTIONS = ["How many hotel gyms were there in Doha in 2023?", "كم بلغت قيمة صادرات قطر إلى الصين في 2022؟"]
W, H = 1400, 1440


async def drive(ws_url: str, app_url: str, out: str) -> None:
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        n = 0

        async def cmd(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        async def js(expr):
            r = await cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
            return r.get("result", {}).get("value")

        async def wait_for(expr, seconds):
            for _ in range(seconds):
                await asyncio.sleep(1)
                if await js(expr):
                    return True
            raise TimeoutError(expr)

        await cmd("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
        await cmd("Page.navigate", url=app_url)
        await wait_for("document.querySelectorAll('[data-testid=stSidebar] button').length > 3", 60)
        for i, q in enumerate(QUESTIONS, 1):
            await asyncio.sleep(3)  # let the previous rerun settle, or the click is lost
            await js(f"[...document.querySelectorAll('button')].find(b => b.innerText.trim() === {json.dumps(q)}).click()")
            await wait_for(f"document.querySelectorAll('[data-testid=stChatMessage]').length >= {2 * i} && "
                           "!document.querySelector('[data-testid=stSpinner]')", 300)
        await asyncio.sleep(2)
        # Open the first answer's source card so the screenshot shows what a citation leads to.
        await js("document.querySelector('[data-testid=stChatMessage] [data-testid=stExpander] summary').click()")
        await asyncio.sleep(2)
        await js("window.scrollTo(0, 0); document.querySelector('[data-testid=stMain]').scrollTo(0, 0)")
        await asyncio.sleep(1)
        shot = await cmd("Page.captureScreenshot", format="png")
        with open(out, "wb") as f:
            f.write(base64.b64decode(shot["data"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--url", default="http://localhost:8502")
    p.add_argument("--browser", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    args = p.parse_args()
    proc = subprocess.Popen([args.browser, "--headless=new", "--remote-debugging-port=9333", "--hide-scrollbars",
                             f"--user-data-dir={tempfile.mkdtemp(prefix='arag-shot')}", "--no-first-run",
                             f"--window-size={W},{H}", "about:blank"])
    try:
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                page = next(t for t in pages if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.3)
        asyncio.run(drive(page["webSocketDebuggerUrl"], args.url, args.out))
        print("saved", args.out)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
