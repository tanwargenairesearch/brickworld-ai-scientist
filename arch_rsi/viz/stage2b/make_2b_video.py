"""VISUALIZATION ONLY: assembles the Stage 2b article video from its three parts. Nothing here scores or selects.

The story, in the order the user asked for on 10 November 2026 (the whole process, not one stage):
  1. what the system is: the three-loop diagram, all three loops shown equally (10 s);
  2. the problem: a target design, and the start and one searched network building it (problem_scene.py's video);
  3. how it searches: one search in one frame, with the build, its evaluation and the progress chart
     (search_build.py's video);
  4. whether the experiments guided the search: every search in every arm, the answer and the limits
     (search_chart_2b.py's video).

The search in parts 2 and 3 is fixed by the rule in README.md, written before the result: among g0 searches whose pick
isn't the start and was judged compliant, sorted by 8-seed primary score, the one at 0-based position floor(m/2). This
script recomputes that choice from results.json and stops if a rendered part isn't for that search.

usage (project root), after search_chart_2b.py --video, search_build.py --video and problem_scene.py --video:
  nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2b/make_2b_video.py
"""
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMP = ROOT / "arch_rsi" / "campaigns" / "stage2b"
SYSTEM_SECONDS, FPS = 10, 12


def chosen_search() -> dict:
    res = json.loads((CAMP / "results.json").read_text())
    assert res["primary_family"]["status"] == "final", "results.json isn't final"
    assert not (ROOT / "arch_rsi" / "stage2" / "FINAL_OPENED_W").exists()
    elig = [s for s in res["per_arm"]["g0"]["searches"] if s["pick"] != "start" and s["verdict"] == "compliant"]
    elig = sorted(elig, key=lambda s: -s["primary_score"])
    assert elig, "no eligible g0 search"
    s = elig[len(elig) // 2]
    return {"search": s["search"], "pick": s["pick"], "primary_score": s["primary_score"], "eligible": len(elig),
            "position": len(elig) // 2, "gate_passed": res["primary_family"]["gate_passed"]}


def seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def main():
    ch = chosen_search()
    bdir = HERE / f"build_g0_s{ch['search']}"
    built = json.loads((bdir / "search_build_data.json").read_text())
    assert Path(built["search"]).resolve() == (CAMP / "W" / "g0" / f"s{ch['search']}").resolve(), built["search"]
    spec = importlib.util.spec_from_file_location("make_system", HERE.parent / "stages" / "system" / "make_system.py")
    SYS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SYS)
    assert not ch["gate_passed"], "the status chips below are for a failed gate"
    sys_png = HERE / "scene1_system.png"
    SYS.render().save(sys_png)
    prob = json.loads((HERE / "problem_data.json").read_text())
    assert prob["search"] == ch["search"] and prob["pick"] == ch["pick"], prob
    parts = [HERE / "problem.mp4", bdir / "search_build.mp4", HERE / "search_2b.mp4"]
    out = HERE / "stage2b.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(SYSTEM_SECONDS), "-r", str(FPS), "-i", str(sys_png),
         "-i", str(parts[0]), "-i", str(parts[1]), "-i", str(parts[2]), "-filter_complex",
         "".join(f"[{i}:v]fps={FPS},format=yuv420p[p{i}];" for i in range(4)) + "[p0][p1][p2][p3]concat=n=4:v=1:a=0[v]",
         "-map", "[v]", "-c:v", "libx264", "-threads", "2", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    data = {"chosen_search": ch, "rule": "README.md: eligible g0 searches by primary score, position floor(m/2)",
            "parts_seconds": {"system": SYSTEM_SECONDS, "problem": round(seconds(parts[0]), 1),
                              "one_search": round(seconds(parts[1]), 1),
                              "all_searches_answer_limits": round(seconds(parts[2]), 1)},
            "video_seconds": round(seconds(out), 1)}
    assert abs(data["video_seconds"] - sum(data["parts_seconds"].values())) < 0.5, data
    (HERE / "stage2b_video_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
