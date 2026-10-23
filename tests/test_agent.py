import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("bm25s")

from rdp import Store, check_run, connect, start_run
from research.agent import extract_answer, run_agent
from research.run_eval import contains_score, normalize
from research.tools import Architecture, _schema


class EchoArch(Architecture):
    name = "echo"
    tools = (_schema("lookup", "Look something up.", query="Query"),)

    def __init__(self):
        super().__init__(corpus=None)

    def tool_lookup(self, query: str) -> str:
        return f"result for {query}"


def tool_call(cid, name, args):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None, prompt=10, completion=5):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))],
              usage=NS(prompt_tokens=prompt, completion_tokens=completion))


class ScriptedClient:
    """Stands in for the OpenAI client: returns scripted replies, records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_tool_call_then_answer():
    client = ScriptedClient([
        reply(calls=[tool_call("c1", "lookup", {"query": "alien"}),
                     tool_call("c2", "lookup", {"query": "scott"})]),
        reply("Found it.\nAnswer: Ridley Scott", prompt=40, completion=8),
    ])
    r = run_agent(client, "m", EchoArch(), "Who directed Alien?")
    assert (r.final_answer, r.stop_reason) == ("Ridley Scott", "answered")
    assert [t["role"] for t in r.turns] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [c["turn_idx"] for c in r.tool_calls] == [1, 1]
    assert r.tool_calls[0]["result"] == "result for alien"
    assert (r.tokens_in, r.tokens_out, r.n_model_calls) == (50, 13, 2)
    # second request carries the assistant tool-call message and both tool results
    roles = [m["role"] for m in client.requests[1]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assert client.requests[1][3]["tool_call_id"] == "c1"


def test_max_steps_and_errors_do_not_raise():
    loop = ScriptedClient([reply(calls=[tool_call(f"c{i}", "lookup", {"query": "x"})])
                           for i in range(3)])
    r = run_agent(loop, "m", EchoArch(), "q", max_steps=3)
    assert (r.stop_reason, r.final_answer, len(r.tool_calls)) == ("max_steps", None, 3)

    r = run_agent(ScriptedClient([TimeoutError("slow")]), "m", EchoArch(), "q")
    assert r.stop_reason == "error:TimeoutError" and r.turns[-1]["role"] == "error"


def test_bad_tool_name_is_reported_to_model():
    client = ScriptedClient([reply(calls=[tool_call("c1", "nope", {})]), reply("Answer: x")])
    r = run_agent(client, "m", EchoArch(), "q")
    assert r.tool_calls[0]["result"].startswith("Error: unknown tool")


def test_episode_round_trips_through_platform_and_passes_checks(tmp_path):
    store = Store(tmp_path)
    client = ScriptedClient([reply(calls=[tool_call("c1", "lookup", {"query": "a"})]),
                             reply("Answer: b")])
    r = run_agent(client, "m", EchoArch(), "q")
    with start_run("eval", model="m", store=store) as run:
        ep = run.log_episode(r.turns, r.tool_calls, architecture="echo",
                             final_answer=r.final_answer, tokens_in=r.tokens_in,
                             tokens_out=r.tokens_out, stop_reason=r.stop_reason)
        run.log_grade(ep, "contains", 1.0)
    assert all(c.passed for c in check_run(store, run.run_id))
    con = connect(store)
    assert con.execute("SELECT stop_reason, n_turns FROM episodes").fetchone() == ("answered", 4)
    ref = con.execute("SELECT result_ref FROM tool_calls").fetchone()[0]
    assert store.get_blob(ref) == b"result for a"


@pytest.mark.parametrize("text, expected", [
    ("Some reasoning.\nAnswer: Hugh Hudson", "Hugh Hudson"),
    ("**Answer:** My Love", "My Love"),
    ("Answer: **My Love**", "My Love"),
    ("answer: first\nmore\nAnswer: second", "second"),
    ("Just Hugh Hudson", "Just Hugh Hudson"),
    (None, None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_contains_score():
    assert normalize("The Beatles’ “Abbey Road”!") == "beatles' abbey road"
    assert contains_score("It was Tony Hatch and Jackie Trent.", ["Tony Hatch and Jackie Trent"])
    assert not contains_score("Tony Hatch", ["Tony Hatch and Jackie Trent"])
    assert contains_score("I don't know", ["I don't know"])
    assert contains_score("This is not in the collection.", ["I don't know"])
    assert not contains_score("$5 million", ["I don't know"])
    assert not contains_score(None, ["x"])


def test_cli_show_handles_tool_only_assistant_turns(tmp_path, capsys):
    from rdp.cli import main

    store = Store(tmp_path)
    client = ScriptedClient([reply(calls=[tool_call("c1", "lookup", {"query": "a"})]),
                             reply("Answer: b")])
    r = run_agent(client, "m", EchoArch(), "q")
    with start_run("eval", store=store) as run:
        ep = run.log_episode(r.turns, r.tool_calls, stop_reason=r.stop_reason)
        run.log_grade(ep, "contains", 1.0)
    assert main(["--root", str(tmp_path), "show", ep]) == 0
    out = capsys.readouterr().out
    assert "lookup" in out and "result for a" in out and "Answer: b" in out
