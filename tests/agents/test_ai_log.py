"""The AI request log (18 Nov 2026): one diary row per AI request, with the key's .env name
(never the key), the time, how it ended, and the service's own reason. Fake networks only."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agents.common.ai_log import AI_REQUEST, AiLog, gemini_details, timed
from agents.agent_c.thinker.step_brain import GroqStepBrain

SECRET = "sk-very-secret-key-123"


class Rows:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def decision(self, what: str, **fields) -> None:
        assert what == AI_REQUEST
        self.rows.append(fields)


def gemini_answer(prompt=120, out=8, thinking=0, headers=None):
    """The shape of a google-genai answer, as much of it as the log reads."""
    usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=thinking,
                            traffic_type=SimpleNamespace(name="ON_DEMAND"))
    return SimpleNamespace(usage_metadata=usage, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
                           model_version="gemini-3.5-flash-lite", sdk_http_response=SimpleNamespace(headers=headers or {}))


async def test_a_finished_request_says_its_key_time_and_what_the_service_said():
    rows = Rows()
    log = AiLog(rows.decision, secrets=[SECRET])
    async with log.request("gemini", "gemini-3.5-flash-lite", "GEMINI_API_KEY_4", "thinker decide") as req:
        await asyncio.sleep(0.02)
        req.add(gemini_details(gemini_answer(headers={"Server-Timing": "gfet4t7; dur=812"})))
    [row] = rows.rows
    assert (row["key"], row["purpose"], row["outcome"]) == ("GEMINI_API_KEY_4", "thinker decide", "ok")
    assert row["ms"] >= 15 and row["tokens_in"] == 120 and row["finish"] == "STOP"
    assert row["server_timing"] == "gfet4t7; dur=812" and row["traffic"] == "ON_DEMAND"
    assert SECRET not in str(row)


async def test_our_time_limit_shows_as_cut_off_not_as_the_services_error():
    rows = Rows()
    log = AiLog(rows.decision)

    async def slow():
        async with log.request("gemini", "m", "GEMINI_API_KEY", "talker job_open"):
            await asyncio.sleep(5)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(slow(), 0.05)
    [row] = rows.rows
    assert row["outcome"] == "cut off" and 40 <= row["ms"] < 1000


async def test_an_error_keeps_the_services_reason_and_never_a_key():
    rows = Rows()
    log = AiLog(rows.decision, secrets=[SECRET])

    class Overloaded(Exception):
        code, status = 503, "UNAVAILABLE"

    with pytest.raises(Overloaded):
        async with log.request("gemini", "m", "GEMINI_API_KEY_3", "yes-judge"):
            raise Overloaded(f"503 UNAVAILABLE. The model is overloaded. (key {SECRET})")
    [row] = rows.rows
    assert (row["outcome"], row["status"], row["status_name"]) == ("error", 503, "UNAVAILABLE")
    assert "overloaded" in row["message"] and SECRET not in row["message"]


async def test_time_spent_on_our_side_mid_reply_is_not_the_ais():
    # The Talker hands each sentence to the voice, which waits while the caller talks: that is
    # not the AI being slow.
    rows = Rows()
    async with timed(AiLog(rows.decision), "gemini", "m", "K", "talker job_open") as req:
        req.first_piece()
        async with req.our_time():
            await asyncio.sleep(0.1)
    [row] = rows.rows
    assert row["our_wait_ms"] >= 90 and row["ms"] < 60 and "first_ms" in row


async def test_no_log_means_no_rows_and_the_request_still_runs():
    async with timed(None, "gemini", "m", "K", "x") as req:
        req.add({"tokens_in": 1})


async def test_groq_rows_show_its_own_queue_time_and_a_refused_key():
    rows = Rows()

    def network(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"].endswith("used-up"):
            return httpx.Response(429, headers={"retry-after": "7", "x-ratelimit-remaining-tokens": "0"},
                                  json={"error": {"message": "Rate limit reached ... tokens per day"}})
        return httpx.Response(200, headers={"x-groq-region": "dls"}, json={
            "choices": [{"message": {"content": '{"exit": "lookup"}'}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "queue_time": 0.412, "total_time": 0.6}})

    clients = [httpx.AsyncClient(base_url="https://groq.test", headers={"Authorization": f"Bearer {key}"},
                                 transport=httpx.MockTransport(network)) for key in ("used-up", "fresh")]
    brain = GroqStepBrain(clients, "openai/gpt-oss-120b", names=["GROQ_API_KEY_8", "GROQ_API_KEY_9"],
                          log=AiLog(rows.decision))
    assert await brain.ask("decide", {"caller_words": ["status batao"]}) == {"exit": "lookup"}
    refused, answered = rows.rows
    assert (refused["key"], refused["outcome"], refused["status"]) == ("GROQ_API_KEY_8", "error", 429)
    assert refused["retry_after"] == "7" and "tokens per day" in refused["message"]
    assert (answered["key"], answered["outcome"], answered["service_queue_ms"]) == ("GROQ_API_KEY_9", "ok", 412)
    assert answered["x_groq_region"] == "dls" and answered["tokens_in"] == 900


def test_key_names_map_each_key_to_its_env_name_and_keys_never_print_them(monkeypatch):
    from agents.agent_c.robot import Keys
    from common import env

    monkeypatch.setenv("HEARKEN_GEMINI_KEY", "GEMINI_API_KEY_4")
    monkeypatch.setenv("GEMINI_API_KEY_4", "g-four")
    monkeypatch.setenv("HEARKEN_GROQ_KEYS", "GROQ_API_KEY_8,GROQ_API_KEY_9")
    monkeypatch.setenv("GROQ_API_KEY_8", "q-eight")
    monkeypatch.setenv("GROQ_API_KEY_9", "q-nine")
    names = env.key_names()
    assert names["g-four"] == "GEMINI_API_KEY_4" and names["q-nine"] == "GROQ_API_KEY_9"
    keys = Keys(gemini="g-four", sarvam="s", groq_pool=("q-eight",), names=names)
    assert keys.name("q-eight") == "GROQ_API_KEY_8" and keys.name("unknown") == ""
    assert "GROQ_API_KEY_8" not in repr(keys) and "q-eight" not in repr(keys) and "g-four" not in repr(keys)


# ---- the Talker and the yes-judge on Groq (18 Nov 2026) ------------------------------------

def _groq(network):
    return httpx.AsyncClient(base_url="https://groq.test", transport=httpx.MockTransport(network))


async def test_the_groq_talker_says_its_reply_and_never_the_text_beside_an_exit():
    # 18 Nov 2026: asked for end_call, qwen also wrote "The caller said: ..." beside the call.
    from agents.agent_c.talker.talker_ai import REPLY, GroqTalkerAI

    answers = iter([
        {"choices": [{"message": {"content": "Ji, aapka order kitchen mein hai. Thodi der mein aayega."}}]},
        {"choices": [{"message": {"content": 'The caller said: "bye"',
                                  "tool_calls": [{"function": {"name": "end_call", "arguments": "{}"}}]}}]}])
    seen: list[dict] = []

    def network(request):
        seen.append(__import__("json").loads(request.content))
        return httpx.Response(200, json=next(answers))

    ai, said = GroqTalkerAI(_groq(network), "qwen/qwen3.8-27b", extra={"reasoning_effort": "none"}), []

    async def speak(sentence):
        said.append(sentence)

    assert await ai.reply("job_open", {"caller_said": "kab aayega?"}, (), speak) == REPLY
    assert said == ["Ji, aapka order kitchen mein hai.", "Thodi der mein aayega."]
    assert await ai.reply("no_job", {"caller_words": ["bye"]}, ("open_job", "end_call"), speak) == "end_call"
    assert len(said) == 2  # nothing more was said
    assert seen[1]["tools"][1]["function"]["name"] == "end_call" and seen[0]["reasoning_effort"] == "none"


async def test_the_groq_talker_moves_to_the_next_key_on_a_limit():
    from agents.agent_c.talker.talker_ai import FailoverTalkerAI, GroqTalkerAI

    used = httpx.Response(429, json={"error": {"message": "Rate limit reached for model ... tokens per minute"}})
    full = httpx.Response(200, json={"choices": [{"message": {"content": "Ji."}}]})
    ais = [GroqTalkerAI(_groq(lambda r: used), "m"), GroqTalkerAI(_groq(lambda r: full), "m")]
    switched, said = [], []

    async def speak(sentence):
        said.append(sentence)

    talker = FailoverTalkerAI(ais, on_switch=switched.append)
    await talker.reply("job_open", {}, (), speak)
    assert said == ["Ji."] and len(switched) == 1


async def test_the_groq_judge_gives_the_one_word_and_logs_its_key():
    from agents.agent_c.guard.judge import CHANGED, read_label, groq_ask
    from common.google import ClientRing

    rows = Rows()

    def network(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "changed"}}]})

    ring = ClientRing([_groq(network)], ["GROQ_API_KEY_9"])
    ask = groq_ask(ring, "qwen/qwen3.8-27b", "label it", AiLog(rows.decision), {"reasoning_effort": "none"})
    assert read_label(await ask("Address badal kar Flat do si kar doon?", "Flat 2C, Rose Apartments")) == CHANGED
    assert rows.rows[0]["key"] == "GROQ_API_KEY_9" and rows.rows[0]["purpose"] == "yes-judge"


# ---- the address check on Jev (19 Nov 2026) ---------------------------------------------------

async def test_the_jev_address_check_gives_its_probability_and_logs_its_key():
    from agents.agent_c.thinker.address_check import JevAddressCheck

    rows, seen = Rows(), []

    def network(request):
        seen.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"same_address": {"type": "noul", "noul": 0.12}},
                                         "usage": {"input_tokens": 500, "output_tokens": 70}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(network))
    check = JevAddressCheck(client, "jev-latest", 2.0, "JEV_API_KEY", AiLog(rows.decision))
    p = await check.same_address(["House 74,", "Sector 15.", "Gurgaon."], "House 74, Sector 15", "4530")
    assert p == 0.12
    assert seen[0]["state"]["robot_wrote_address"] == "House 74, Sector 15"
    assert seen[0]["questions"]["same_address"]["type"] == "noul"
    assert rows.rows[0]["key"] == "JEV_API_KEY" and rows.rows[0]["probability"] == 0.12


async def test_the_address_check_never_blocks_a_change_by_failing():
    # No answer (a 5xx, or too slow) is "no answer": the change goes on as before the check.
    from agents.agent_c.thinker.address_check import JevAddressCheck

    errors = []

    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    broken = JevAddressCheck(httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(529, json={}))),
                             "jev-latest", 2.0, on_error=errors.append)
    late = JevAddressCheck(httpx.AsyncClient(transport=httpx.MockTransport(slow)), "jev-latest", 0.05,
                           on_error=errors.append)
    assert await broken.same_address(["x"], "y", "4530") is None
    assert await late.same_address(["x"], "y", "4530") is None
    assert len(errors) == 2 and "goes on unchecked" in errors[0]
