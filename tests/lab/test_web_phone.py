"""The web phone: its JavaScript tests, its messages against Python's, the page
server, and the echo robot command it is checked with."""

import asyncio
import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cli.phone_page import make_server
from common.wire.caller_line import CallerLine, QueueMic
from common.wire.protocol import PIECE_BYTES, Cleared, End, Key, Played, Start, encode

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


@needs_node
def test_the_javascript_tests_pass():
    result = subprocess.run(
        [NODE, "--test", "lab/web_phone/phone-dsp.test.js"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_python_and_javascript_write_exactly_the_same_messages():
    """The two sides of the wire are written in two languages; this pins them together."""
    script = (
        "import('./lab/web_phone/phone-dsp.js').then(m => console.log(JSON.stringify(["
        "m.encodeStart('web-1'), m.encodePlayed('q1', 1240), m.encodeCleared(37, 980, ['q1', 'q2']),"
        "m.encodeCleared(0, 0, []), m.encodeKey('#', 20), m.encodeEnd('bye')])))"
    )
    result = subprocess.run([NODE, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        encode(Start("web-1")),
        encode(Played("q1", 1240)),
        encode(Cleared(37, 980, ("q1", "q2"))),
        encode(Cleared(0, 0, ())),
        encode(Key("#", 20)),
        encode(End("bye")),
    ]


@pytest.fixture
def page_url():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.headers.get_content_type(), response.read().decode("utf-8")


def test_the_page_and_its_scripts_are_served_with_the_right_types(page_url):
    kind, html = fetch(page_url + "/")
    assert kind == "text/html" and '<script type="module" src="phone.js">' in html
    for script in ("phone.js", "phone-dsp.js"):
        kind, body = fetch(f"{page_url}/{script}")
        assert kind == "text/javascript", script  # browsers refuse modules sent as anything else
    assert 'registerProcessor("hearken-phone"' in fetch(page_url + "/phone-dsp.js")[1]
    assert 'addModule("phone-dsp.js")' in fetch(page_url + "/phone.js")[1]  # the same file is the worklet
    with pytest.raises(urllib.error.HTTPError):
        fetch(page_url + "/not-there.js")


def test_the_page_server_only_listens_on_this_computer():
    server = make_server(port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


async def test_the_echo_robot_command_answers_a_call():
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "cli.dummy_robot", "--port", "0", "--delay", "0.2",
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        first_line = (await asyncio.wait_for(process.stdout.readline(), 30)).decode().strip()
        assert first_line.startswith("echo robot: ws://127.0.0.1:")
        url = first_line.removeprefix("echo robot: ")
        marked = [bytes([n]) * PIECE_BYTES for n in range(1, 4)]
        heard = []

        class Ear:
            def heard(self, piece, caller_ms):
                if piece in marked:
                    heard.append(piece)

        mic = QueueMic()
        mic.say(marked)
        line = CallerLine(url, "cli-check", mic, Ear())
        call = asyncio.create_task(line.run())
        await asyncio.sleep(1.5)
        await line.hang_up("done")
        assert await asyncio.wait_for(call, 10) == "caller:done"
        assert heard == marked
    finally:
        process.terminate()
        await process.wait()
