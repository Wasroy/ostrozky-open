import csv
import json
import os

from rescuesim.check_agent import check_agent
from rescuesim.evaluate import METRIC_KEYS, main as evaluate_main


def test_hover_stub_passes_check_agent(root, capsys):
    assert check_agent(os.path.join(root, "examples", "hover_agent"))
    assert "Agent API OK" in capsys.readouterr().out


def test_bad_agent_fails_check_agent(tmp_path):
    pkg = tmp_path / "bad_agent"
    pkg.mkdir()
    (pkg / "agent.py").write_text("class Agent:\n    def reset(self):\n        pass\n"
                                  "    def act(self, obs):\n        return [0.0, 0.0, 2.0]\n")
    assert not check_agent(str(pkg), verbose=False)


def test_evaluate_smoke(root, tmp_path):
    out = tmp_path / "eval"
    evaluate_main(["--agent", os.path.join(root, "examples", "hover_agent"), "--terrains", "empty",
                   "--poses", "2", "--runs", "2", "--workers", "1", "--timeout", "1", "--out", str(out), "--log"])
    rows = list(csv.DictReader(open(out / "flights.csv")))
    assert len(rows) == 4 and {r["outcome"] for r in rows} == {"timeout"}
    m = json.load(open(out / "metrics.json"))
    assert set(METRIC_KEYS) <= set(m["overall"]) and m["overall"]["flights"] == 4
    assert m["overall"]["timeout_rate"] == 1.0
    assert len(os.listdir(out / "logs")) == 4
