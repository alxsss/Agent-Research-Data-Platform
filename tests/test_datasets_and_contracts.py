import json

import pytest

from rdp import ContractViolation, Registry, Store, check_run, no_overlap, start_run


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_register_versions_idempotency_and_metadata(tmp_path):
    reg = Registry(Store(tmp_path / "store"))
    f = write_jsonl(tmp_path / "a.jsonl", [{"q": "x"}, {"q": "y"}])
    v1 = reg.register("evalset", f)
    assert v1 == "evalset:v1"
    assert reg.register("evalset", f) == v1          # same content -> same version
    assert reg.get(v1).row_count == 2
    write_jsonl(f, [{"q": "x"}])
    assert reg.register("evalset", f) == "evalset:v2"
    assert [r[0] for r in reg.list()] == ["evalset:v1", "evalset:v2"]  # returns timestamps
    assert reg.get("evalset:latest").version_id == "evalset:v2"
    assert reg.lineage("evalset:latest")[0][0] == "evalset:v2"
    with pytest.raises(KeyError):
        reg.resolve("nope:latest")


def test_cli_datasets_lists_registered_versions(tmp_path, capsys):
    from rdp.cli import main

    reg = Registry(Store(tmp_path / "store"))
    reg.register("evalset", write_jsonl(tmp_path / "a.jsonl", [{"q": "x"}]))
    assert main(["--root", str(tmp_path / "store"), "datasets"]) == 0
    assert capsys.readouterr().out.startswith("evalset:v1\t1\t")


def test_verify_detects_modified_files(tmp_path):
    reg = Registry(Store(tmp_path / "store"))
    f = write_jsonl(tmp_path / "a.jsonl", [{"q": "x"}])
    v = reg.register("evalset", f)
    assert reg.verify(v)
    write_jsonl(f, [{"q": "changed"}])
    assert not reg.verify(v)


def test_lineage_chain_and_unknown_parent(tmp_path):
    reg = Registry(Store(tmp_path / "store"))
    a = reg.register("a", write_jsonl(tmp_path / "a.jsonl", [{"x": 1}]))
    b = reg.register("b", write_jsonl(tmp_path / "b.jsonl", [{"x": 2}]), parents=[a],
                     producer_run_id="run-1")
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "w.bin").write_bytes(b"\x00\x01")
    (d / "meta.json").write_text("{}")  # mixed formats: hash it, don't try to read it
    c = reg.register("c", d, parents=[b])
    assert reg.get(c).row_count is None
    assert reg.lineage(c) == [(c, 0, None), (b, 1, "run-1"), (a, 2, None)]
    with pytest.raises(KeyError):
        reg.register("z", tmp_path / "a.jsonl", parents=["nope:v1"])


def test_no_overlap_blocks_contaminated_dataset(tmp_path):
    reg = Registry(Store(tmp_path / "store"))
    evalset = write_jsonl(tmp_path / "eval.jsonl", [{"question": "Who directed  Alien?"}])
    train = write_jsonl(tmp_path / "train.jsonl",
                        [{"question": "who directed alien?"}, {"question": "other"}])
    with pytest.raises(ContractViolation) as e:
        reg.register("sft", train, checks=[no_overlap(evalset, key="question")])
    assert e.value.results[0].n_violations == 1
    assert reg.list() == []                            # nothing recorded on failure

    write_jsonl(train, [{"question": "other"}])
    assert reg.register("sft", train, checks=[no_overlap(evalset, key="question")])


def test_check_run_passes_on_clean_run_and_catches_ungraded(tmp_path):
    store = Store(tmp_path)
    with start_run("eval", store=store) as run:
        ep = run.log_episode(turns=[{"role": "user", "content": "q", "token_count": 3},
                                    {"role": "assistant", "content": "a"}],
                             tool_calls=[{"turn_idx": 1, "tool_name": "search", "result": "r"}],
                             tokens_in=3, tokens_out=1)
        run.log_grade(ep, "exact", 1.0)
        run.log_episode(turns=[{"role": "user", "content": "q2"}])   # never graded
    results = {r.name: r for r in check_run(store, run.run_id)}
    assert not results["every_episode_has_a_grade"].passed
    assert results["every_episode_has_a_grade"].n_violations == 1
    assert all(r.passed for n, r in results.items() if n != "every_episode_has_a_grade")


def test_check_run_catches_dangling_tool_call(tmp_path):
    store = Store(tmp_path)
    with start_run("eval", store=store) as run:
        ep = run.log_episode(turns=[{"role": "user", "content": "q"}],
                             tool_calls=[{"turn_idx": 5, "tool_name": "search"}])
        run.log_grade(ep, "exact", 0.0)
    results = {r.name: r for r in check_run(store, run.run_id)}
    assert not results["tool_calls_reference_existing_turn"].passed


def test_demo_end_to_end(tmp_path):
    from rdp.cli import main

    store_root = tmp_path / "store"
    assert main(["--root", str(store_root), "demo", "--episodes", "10"]) == 0
    reg = Registry(Store(store_root))
    chain = [v for v, _, _ in reg.lineage("checkpoint:v1")]
    assert chain == ["checkpoint:v1", "sft_trajectories:v1", "evalset:v1"]
    assert main(["--root", str(store_root), "runs"]) == 0
