"""Judge every call of a run, and keep the marks in each call's score.json
(Part 5, L10 and L11). A call the judge can't mark stays unjudged, with why;
its other marks still stand.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lab.judge.judge import Ask, BadAnswer, judge, packet_from_call
from lab.scorer.call_record import load_call
from lab.scorer.score import CallScore, read_score, write_score
from lab.settings import LabSettings


async def judge_run(run_dir: Path, ask: Ask, settings: LabSettings, model: str) -> list[CallScore]:
    judged = []
    for folder in sorted((run_dir / "calls").iterdir()):
        if not (folder / "score.json").exists():
            continue  # a lab problem: rerun, never marked
        score = read_score(folder)
        try:
            marks = await judge(packet_from_call(load_call(folder, settings.sound_level)), ask)
        except BadAnswer as exc:
            score.notes.append(f"judge answer refused: {exc}")
        else:
            score.judge = marks.as_dict(model)
        score.decide()
        write_score(folder, score)
        judged.append(score)
    (run_dir / "judged.json").write_text(json.dumps([asdict(s) for s in judged], indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    return judged
