"""Frozen learned-stopping pipeline: dev train -> gates -> formal -> local delivery.

This supervisor never changes code, configuration, thresholds, or seeds.  It
attaches to the manually started development run, then follows the frozen run
order.  State is its only progress interface.  It intentionally does not read
formal result contents until every required formal row is complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:/Users/35884/Documents/Spacecraft")
TOPIC = ROOT / "过程文件/停止头"
REC = TOPIC / "记录"
SCRIPTS = TOPIC / "脚本"
STAGING = TOPIC / "交付展开"
STATE_PATH = REC / "STOPPING_EXECUTION.json"
LOCK_PATH = REC / "supervisor.lock"
REPO = Path(r"D:/py/DRL2")
PYTHON = REPO / ".venv/Scripts/python.exe"
COMMIT = "2e5c236f7412caea3b203519619ef7a48bb16df9"
DEV_SEED = 262440
FORMAL_SEEDS = (262430, 262431, 262432)
DEV_RANGE = "266000-266047"
FORMAL_RANGE = "267000-267047"

ENV = os.environ.copy()
ENV.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"})


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args: str) -> str:
    result = subprocess.run(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), *args],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def require_frozen_tree() -> None:
    if git("rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("固定提交不符，禁止继续")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("跟踪工作树不干净，禁止继续")
    if git("diff", "a714c61", "HEAD", "--stat", "--", "env", "controllers", "dynamics"):
        raise RuntimeError("物理或控制目录相对a714c61有变化，禁止继续")


def is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    return str(pid) in result.stdout and "No tasks" not in result.stdout


def files_complete(directory: Path, first: int) -> bool:
    expected = {f"seed_{seed}.json" for seed in range(first, first + 48)}
    actual = {path.name for path in directory.glob("seed_*.json")}
    return actual == expected and not list(directory.glob("*.lock")) and not list(directory.glob("*.tmp"))


class Pipeline:
    def __init__(self) -> None:
        self.state = read_json(STATE_PATH) if STATE_PATH.exists() else {
            "schema": "stopping_pipeline_v1", "created_at": now(), "commit": COMMIT,
            "repo": str(REPO), "python": str(PYTHON), "phase": "initializing", "events": [], "workers": [],
        }
        self.processes: list[tuple[subprocess.Popen, dict]] = []

    def save(self, phase: str | None = None) -> None:
        if phase and phase != self.state.get("phase"):
            self.state["phase"] = phase
            self.state.setdefault("events", []).append({"at": now(), "phase": phase})
            print(f"{now()} {phase}", flush=True)
        self.state["updated_at"] = now()
        self.state["supervisor_pid"] = os.getpid()
        write_json(STATE_PATH, self.state)

    def event(self, name: str, **details: object) -> None:
        self.state.setdefault("events", []).append({"at": now(), "event": name, **details})
        self.save()

    def fail(self, message: str) -> None:
        self.state["error"] = message
        self.save("program_fault_stop")
        print(f"STOPPING_PIPELINE_ERROR: {message}", file=sys.stderr, flush=True)

    def launch(self, name: str, args: list[str], kind: str) -> tuple[subprocess.Popen, dict]:
        stdout = REC / f"{name}.stdout.log"
        stderr = REC / f"{name}.stderr.log"
        if stdout.exists() or stderr.exists():
            raise RuntimeError(f"运行日志已存在，拒绝重复启动：{name}")
        command = [str(PYTHON), "-u", "-B", *args]
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=out, stderr=err,
                                       creationflags=0x08000000)
        worker = {"name": name, "kind": kind, "pid": process.pid, "started_at": now(),
                  "command": command, "stdout": str(stdout), "stderr": str(stderr), "exit_code": None}
        self.state["workers"].append(worker)
        self.processes.append((process, worker))
        self.save()
        return process, worker

    def monitor(self, phase: str, seconds: int = 60) -> None:
        self.save(phase)
        while self.processes:
            for process, worker in list(self.processes):
                code = process.poll()
                if code is not None:
                    worker["exit_code"] = code
                    worker["exit_observed_at"] = now()
                    self.processes.remove((process, worker))
                    self.save()
                    if code:
                        raise RuntimeError(f"{worker['name']}异常退出，退出码{code}；已保留日志与锁现场")
            if self.processes:
                time.sleep(seconds)
                self.save()

    def prepare(self) -> None:
        require_frozen_tree()
        testlog = REC / "pytest_preflight.log"
        if not testlog.exists() or "30 passed in " not in testlog.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError("开发前指定测试回执不完整")
        launch = REC / "DEV_LAUNCH.json"
        if not launch.exists():
            raise RuntimeError("缺少用户手动启动开发训练的回执")
        launch_data = read_json(launch)
        if launch_data.get("commit") != COMMIT or launch_data.get("budget") != "30000 outer decisions; MPC suffix excluded and recorded separately":
            raise RuntimeError("开发训练启动回执与冻结方案不符")
        self.state["preflight"] = {"tests": "30 passed", "dev_launch": str(launch), "prepared_at": now()}
        self.save("await_dev_training")

    def wait_for_dev(self) -> None:
        run_dir = REPO / "logs/stop_dev_262440"
        manifest_path = run_dir / "manifest.json"
        launch = read_json(REC / "DEV_LAUNCH.json")
        while True:
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                self.state["dev_progress"] = {key: manifest.get(key) for key in (
                    "status", "requested_outer_decisions", "actual_outer_decisions", "continue_transitions",
                    "handoff_episodes", "suffix_simulated_decisions", "gradient_updates")}
                self.save("await_dev_training")
                if manifest.get("status") == "completed":
                    if manifest.get("actual_outer_decisions") != 30000 or not (run_dir / "final_model.zip").exists():
                        raise RuntimeError("开发训练完成记录、预算或最终模型不符")
                    self.event("development_training_completed", model_sha256=sha256(run_dir / "final_model.zip"))
                    return
                if manifest.get("status") not in {"running", "interrupted"}:
                    raise RuntimeError(f"开发训练状态异常：{manifest.get('status')}")
            if not is_alive(launch.get("wrapper_pid")):
                raise RuntimeError("开发训练进程已消失而manifest未完成；保全现场，不自动重启")
            time.sleep(60)

    def development_evaluation(self) -> dict:
        out_dir = REPO / "eval/stopping/dev/stopping"
        if out_dir.exists() and any(out_dir.iterdir()):
            raise RuntimeError("开发评估目录已有资产，拒绝覆盖或重跑")
        (REPO / "eval/stopping_logs").mkdir(parents=True, exist_ok=True)
        common = ["-m", "experiments.v3_stopping", "evaluate", "--run-dir", "logs/stop_dev_262440",
                  "--row", "stopping", "--seeds", DEV_RANGE, "--output-dir", "eval/stopping/dev/stopping"]
        for worker in range(1, 4):
            self.launch(f"dev_eval_{worker}", common, "development_evaluation")
        self.monitor("development_evaluation")
        if not files_complete(out_dir, 266000):
            raise RuntimeError("开发评估未产生完整48开局或仍有锁/临时文件")
        output = REPO / "eval/stopping/dev/devcheck.json"
        if output.exists():
            raise RuntimeError("devcheck结果已存在，拒绝覆盖")
        self.launch("devcheck", ["-m", "experiments.v3_stopping", "devcheck", "--run-dir", "logs/stop_dev_262440",
                                  "--eval-dir", "eval/stopping/dev/stopping", "--output", "eval/stopping/dev/devcheck.json"],
                    "development_gate")
        self.monitor("development_gate")
        result = read_json(output)
        self.state["devcheck"] = result
        self.save("development_gate_decided")
        return result

    def formal_training(self) -> None:
        require_frozen_tree()
        for seed in FORMAL_SEEDS:
            run_dir = REPO / f"logs/stop_{seed}"
            if run_dir.exists():
                raise RuntimeError(f"正式训练目录已存在，拒绝覆盖：{run_dir}")
            self.launch(f"formal_train_{seed}", ["-m", "train.train_stopping", "--steps", "60000", "--seed", str(seed),
                                                   "--run-name", f"stop_{seed}", "--device", "cpu"], "formal_training")
        self.monitor("formal_training")
        models: dict[str, str] = {}
        for seed in FORMAL_SEEDS:
            run_dir = REPO / f"logs/stop_{seed}"
            manifest = read_json(run_dir / "manifest.json")
            if manifest.get("status") != "completed" or manifest.get("actual_outer_decisions") != 60000:
                raise RuntimeError(f"正式训练{seed}未按60000外层决策完整结束")
            model = run_dir / "final_model.zip"
            if not model.exists():
                raise RuntimeError(f"正式训练{seed}缺少最终模型")
            models[str(seed)] = sha256(model)
        self.state["formal_models"] = models
        self.save("formal_training_completed")

    def attach_manual_formal_training(self) -> None:
        """Wait for three user-launched frozen formal runs; never starts training."""
        require_frozen_tree()
        devcheck_path = REPO / "eval/stopping/dev/devcheck.json"
        if not devcheck_path.exists() or not read_json(devcheck_path).get("all_pass"):
            raise RuntimeError("开发D1/D2/D3未完整通过，不能接管正式训练")
        run_dirs = {seed: REPO / f"logs/stop_{seed}" for seed in FORMAL_SEEDS}
        missing = [str(seed) for seed, path in run_dirs.items() if not (path / "manifest.json").exists()]
        if missing:
            raise RuntimeError("尚未找到用户手动启动的正式训练：" + ", ".join(missing))
        self.state["devcheck"] = read_json(devcheck_path)
        while True:
            progress: dict[str, object] = {}
            complete = True
            for seed, run_dir in run_dirs.items():
                manifest = read_json(run_dir / "manifest.json")
                progress[str(seed)] = {key: manifest.get(key) for key in (
                    "status", "requested_outer_decisions", "actual_outer_decisions", "continue_transitions",
                    "handoff_episodes", "suffix_simulated_decisions", "gradient_updates")}
                if manifest.get("status") == "completed":
                    if manifest.get("actual_outer_decisions") != 60000 or not (run_dir / "final_model.zip").exists():
                        raise RuntimeError(f"正式训练{seed}完成记录、预算或最终模型不符")
                elif manifest.get("status") == "running":
                    complete = False
                else:
                    raise RuntimeError(f"正式训练{seed}状态异常：{manifest.get('status')}")
            self.state["formal_training_progress"] = progress
            self.save("await_manual_formal_training")
            if complete:
                break
            time.sleep(60)
        self.state["formal_models"] = {str(seed): sha256(run_dirs[seed] / "final_model.zip") for seed in FORMAL_SEEDS}
        self.save("formal_training_completed")

    def formal_evaluation(self) -> dict:
        require_frozen_tree()
        root = REPO / "eval/stopping/formal"
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("正式评估目录已有资产，拒绝覆盖或重跑")
        (REPO / "eval/stopping_logs").mkdir(parents=True, exist_ok=True)
        jobs: list[tuple[str, list[str]]] = []
        for seed in FORMAL_SEEDS:
            for row in ("stopping", "learned"):
                jobs.append((f"formal_{row}_{seed}", ["-m", "experiments.v3_stopping", "evaluate", "--run-dir",
                    f"logs/stop_{seed}", "--row", row, "--seeds", FORMAL_RANGE, "--output-dir",
                    f"eval/stopping/formal/{row}/{seed}"]))
        pending_pure = [f"formal_pure_{index}" for index in (1, 2)]
        active: list[tuple[subprocess.Popen, dict]] = []
        for name, args in jobs:
            active.append(self.launch(name, args, "formal_evaluation"))
        self.save("formal_evaluation_blinded")
        last_count = 0.0
        while active or pending_pure:
            for process, worker in list(active):
                code = process.poll()
                if code is not None:
                    worker["exit_code"] = code
                    worker["exit_observed_at"] = now()
                    active.remove((process, worker))
                    self.processes.remove((process, worker))
                    self.save()
                    if code:
                        raise RuntimeError(f"{worker['name']}异常退出，退出码{code}；保留现场")
            while pending_pure and len(active) < 6:
                name = pending_pure.pop(0)
                active.append(self.launch(name, ["-m", "experiments.v3_stopping", "evaluate", "--run-dir",
                    "logs/stop_262430", "--row", "pure", "--seeds", FORMAL_RANGE, "--output-dir",
                    "eval/stopping/formal/pure"], "formal_evaluation"))
            if time.monotonic() - last_count >= 300:
                self.state["formal_file_counts"] = {
                    "pure": len(list((root / "pure").glob("seed_*.json"))),
                    **{f"{row}_{seed}": len(list((root / row / str(seed)).glob("seed_*.json")))
                       for row in ("stopping", "learned") for seed in FORMAL_SEEDS},
                }
                last_count = time.monotonic()
            self.save("formal_evaluation_blinded")
            if active:
                time.sleep(60)
        required = [root / "pure"] + [root / row / str(seed) for row in ("stopping", "learned") for seed in FORMAL_SEEDS]
        if not all(files_complete(path, 267000) for path in required):
            raise RuntimeError("正式评估文件集合、锁或临时文件不符")
        readout = root / "readout.json"
        if readout.exists():
            raise RuntimeError("正式官方判读已存在，拒绝覆盖")
        args = ["-m", "experiments.v3_stopping", "readout", "--pure", "eval/stopping/formal/pure"]
        for seed in FORMAL_SEEDS:
            args += ["--stopping", f"{seed}=eval/stopping/formal/stopping/{seed}"]
        for seed in FORMAL_SEEDS:
            args += ["--learned", f"{seed}=eval/stopping/formal/learned/{seed}"]
        args += ["--output", "eval/stopping/formal/readout.json"]
        self.launch("formal_official_readout", args, "formal_readout")
        self.monitor("formal_official_readout")
        result = read_json(readout)
        self.state["formal_readout"] = result
        self.save("formal_readout_completed")
        return result

    def add_file(self, sources: dict[str, Path], arcname: str, source: Path) -> None:
        if arcname in sources:
            raise RuntimeError(f"交付成员名重复：{arcname}")
        if not source.is_file():
            raise RuntimeError(f"交付源文件缺失：{source}")
        sources[arcname] = source

    def add_tree(self, sources: dict[str, Path], prefix: str, source: Path, exclude_models: bool = False) -> None:
        if not source.exists():
            return
        for path in sorted(source.rglob("*")):
            if path.is_file() and not (exclude_models and path.name == "final_model.zip"):
                self.add_file(sources, f"{prefix}/{path.relative_to(source).as_posix()}", path)

    def package(self, development_only: bool, verdict: str) -> dict:
        date = datetime.now().strftime("%Y%m%d")
        tag = f"STOPPING_DEV_{date}" if development_only else f"STOPPING_FORMAL_{date}"
        stage = STAGING / tag
        target = ROOT / "上层交付/待发布" / f"{tag}.zip"
        if stage.exists() or target.exists():
            raise RuntimeError("交付展开目录或待发布ZIP已存在，拒绝覆盖")
        stage.mkdir(parents=True)
        sources: dict[str, Path] = {}
        self.add_file(sources, "frozen/STOPPING_RUN_ORDER_LOWER_20261002.md", REPO / "docs/STOPPING_RUN_ORDER_LOWER_20261002.md")
        self.add_file(sources, "frozen/STOPPING_METHOD_PREREGISTRATION_20261002.md", REPO / "docs/STOPPING_METHOD_PREREGISTRATION_20261002.md")
        self.add_file(sources, "frozen/train_stopping.py", REPO / "train/train_stopping.py")
        self.add_file(sources, "frozen/stopping.py", REPO / "train/stopping.py")
        self.add_file(sources, "frozen/v3_stopping.py", REPO / "experiments/v3_stopping.py")
        self.add_tree(sources, "development/run", REPO / "logs/stop_dev_262440", exclude_models=True)
        self.add_tree(sources, "development/evaluation", REPO / "eval/stopping/dev")
        self.add_tree(sources, "process_records", REC)
        self.add_tree(sources, "tools", SCRIPTS)
        model_hashes = {"stop_dev_262440/final_model.zip": sha256(REPO / "logs/stop_dev_262440/final_model.zip")}
        if not development_only:
            self.add_tree(sources, "formal/evaluation", REPO / "eval/stopping/formal")
            for seed in FORMAL_SEEDS:
                self.add_tree(sources, f"formal/runs/stop_{seed}", REPO / f"logs/stop_{seed}", exclude_models=True)
                model_hashes[f"stop_{seed}/final_model.zip"] = sha256(REPO / f"logs/stop_{seed}/final_model.zip")
        report = stage / "REPORT.md"
        gate = self.state.get("devcheck", {})
        report.write_text(
            "# learned stopping option 独立交付\n\n"
            f"生成时间：{now()}。固定提交：`{COMMIT}`。官方状态：`{verdict}`。\n\n"
            "方法冻结为原V3e二维任务策略加伯努利停止头；Q_H是Pure MPC后段真实回报，gamma=0.999，"
            "训练期交接概率[0.002,0.01]，部署第一次beta>=0.5单向交接。预算是外层决策点，MPC后段仅供标签并单列算量。\n\n"
            f"开发结构检查：`all_pass={gate.get('all_pass')}`；D1={gate.get('D1_not_collapsed', {}).get('pass')}，"
            f"D2={gate.get('D2_learned_reaches_mid_late', {}).get('pass')}，D3={gate.get('D3_state_dependent_stopping_value', {}).get('pass')}。\n\n"
            + ("本包为开发停止包：任一结构门未过，未启动正式训练。\n" if development_only else
               "正式包含三个全新60k模型与267000–267047七行逐开局结果；官方readout的coordination_gain是stopping相对同一策略learned行的增益，必须据此区分训练改善和实际交接贡献。\n")
            + "模型ZIP不打包，只提供路径与SHA-256，见MODEL_FILES_SHA256.json。日志、逐开局结果、运行manifest、冻结代码与哈希清单都在本包中。\n",
            encoding="utf-8")
        self.add_file(sources, "REPORT.md", report)
        model_file = stage / "MODEL_FILES_SHA256.json"
        write_json(model_file, model_hashes)
        self.add_file(sources, "MODEL_FILES_SHA256.json", model_file)
        for arcname, source in sources.items():
            destination = stage / arcname
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
                if sha256(source) != sha256(destination):
                    raise RuntimeError(f"交付复制哈希不符：{arcname}")
        listing = {path.relative_to(stage).as_posix(): sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        (stage / "ALL_FILES_SHA256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in listing.items()), encoding="utf-8")
        json_listing = {name: digest for name, digest in listing.items() if name.endswith(".json")}
        (stage / "JSON_SHA256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in json_listing.items()), encoding="utf-8")
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
        with zipfile.ZipFile(target) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC失败：{bad}")
            for name, digest in listing.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise RuntimeError(f"ZIP成员SHA不符：{name}")
        receipt = {"status": "verified", "created_at": now(), "verdict": verdict, "commit": COMMIT,
                   "zip": str(target), "zip_sha256": sha256(target), "zip_bytes": target.stat().st_size,
                   "crc_ok": True, "listed_members_sha256_ok": True, "development_only": development_only}
        write_json(REC / f"{tag}_RECEIPT.json", receipt)
        return receipt

    def promote(self, receipt: dict) -> None:
        latest = ROOT / "上层交付/最新"
        history = ROOT / "上层交付/历史" / "STAGE_C_20261002"
        if history.exists():
            raise RuntimeError("Stage C历史目标已存在，禁止自动覆盖最新交付")
        expected = {"README.md", "REPORT.md", "c2_report.json", "STAGE_C_20261002.zip", "STAGE_C_DELIVERY_RECEIPT.json"}
        existing = {path.name for path in latest.iterdir() if path.is_file()}
        if existing != expected:
            raise RuntimeError("最新交付入口内容已变化，保留待发布包，禁止自动移动")
        history.mkdir(parents=True)
        for source in latest.iterdir():
            if source.is_file():
                source.rename(history / source.name)
        tag = Path(receipt["zip"]).stem
        stage = STAGING / tag
        shutil.copy2(stage / "REPORT.md", latest / "REPORT.md")
        shutil.copy2(REPO / "eval/stopping/dev/devcheck.json", latest / "devcheck.json")
        if not receipt["development_only"]:
            shutil.copy2(REPO / "eval/stopping/formal/readout.json", latest / "readout.json")
        shutil.copy2(receipt["zip"], latest / Path(receipt["zip"]).name)
        write_json(latest / "STOPPING_DELIVERY_RECEIPT.json", receipt)
        verdict = receipt["verdict"]
        readout_note = "正式官方读数在readout.json，包含METHOD_HOLDS / METHOD_DOES_NOT_HOLD / FIDELITY_FAIL与coordination_gain。" if not receipt["development_only"] else "开发门未通过；未启动正式训练，等待上层讨论。"
        (latest / "README.md").write_text(
            f"# 当前交付：learned stopping option（{datetime.now().strftime('%Y%m%d')}）\n\n"
            f"官方状态：`{verdict}`。先读REPORT.md、devcheck.json及存在时readout.json；独立证据ZIP为{Path(receipt['zip']).name}，"
            f"SHA-256 `{receipt['zip_sha256']}`。{readout_note}\n\n"
            "本地ZIP已核验CRC及清单成员SHA。未自动发布到GitHub Release；那需要对完整原始证据包和目的地的单独授权。"
            "前一轮Stage C已归档至历史/STAGE_C_20261002。\n", encoding="utf-8")
        self.state["delivery"] = receipt
        self.save("delivery_completed")

    def run(self) -> None:
        self.prepare()
        self.wait_for_dev()
        devcheck = self.development_evaluation()
        if not devcheck.get("all_pass"):
            receipt = self.package(True, "DEV_GATE_STOP")
            self.promote(receipt)
            self.save("await_upper_discussion")
            return
        self.formal_training()
        result = self.formal_evaluation()
        receipt = self.package(False, result["verdict"])
        self.promote(receipt)
        if result["verdict"] == "METHOD_HOLDS":
            self.save("await_user_next_training")
        elif result["verdict"] == "METHOD_DOES_NOT_HOLD":
            self.save("await_upper_interface")
        else:
            self.save("stopped_fidelity")


def compact_status() -> int:
    if not STATE_PATH.exists():
        print("STATUS: NOT_STARTED")
        return 0
    state = read_json(STATE_PATH)
    print(f"STATUS: {state.get('phase')} updated_at={state.get('updated_at')}")
    if state.get("error"):
        print(f"ERROR: {state['error']}")
    progress = state.get("dev_progress")
    if progress:
        print("DEV: " + ", ".join(f"{key}={value}" for key, value in progress.items() if value is not None))
    dev_dir = REPO / "eval/stopping/dev/stopping"
    if dev_dir.exists():
        print(f"DEV_EVAL_FILES: {len(list(dev_dir.glob('seed_*.json')))}/48 locks={len(list(dev_dir.glob('*.lock')))} tmp={len(list(dev_dir.glob('*.tmp')))}")
    for seed in FORMAL_SEEDS:
        manifest = REPO / f"logs/stop_{seed}/manifest.json"
        if manifest.exists():
            run = read_json(manifest)
            print(f"FORMAL_TRAIN_{seed}: status={run.get('status')} outer_decisions={run.get('actual_outer_decisions')}")
    if state.get("formal_file_counts"):
        print("FORMAL_FILES: " + ", ".join(f"{key}={value}/48" for key, value in state["formal_file_counts"].items()))
    if state.get("devcheck"):
        check = state["devcheck"]
        print(f"DEV_GATE: all_pass={check.get('all_pass')} D1={check.get('D1_not_collapsed', {}).get('pass')} D2={check.get('D2_learned_reaches_mid_late', {}).get('pass')} D3={check.get('D3_state_dependent_stopping_value', {}).get('pass')}")
    if state.get("formal_readout"):
        print(f"VERDICT: {state['formal_readout'].get('verdict')}")
    if state.get("delivery"):
        print(f"DELIVERY: {state['delivery']['zip']} SHA256={state['delivery']['zip_sha256']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("start", "run", "formal-supervise", "status"))
    args = parser.parse_args()
    if args.mode == "status":
        return compact_status()
    if args.mode == "start":
        REC.mkdir(parents=True, exist_ok=True)
        launch = REC / "SUPERVISOR_LAUNCH.json"
        if LOCK_PATH.exists() or STATE_PATH.exists() or launch.exists():
            raise RuntimeError("已有监督锁、状态或启动回执；请运行status，禁止重复启动")
        out = REC / "supervisor.stdout.log"
        err = REC / "supervisor.stderr.log"
        if out.exists() or err.exists():
            raise RuntimeError("已有监督日志，禁止覆盖")
        command = [str(PYTHON), "-u", "-B", str(Path(__file__).resolve()), "run"]
        with out.open("wb") as stdout, err.open("wb") as stderr:
            process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=stdout, stderr=stderr,
                                       creationflags=0x08000000)
        write_json(launch, {"started_at": now(), "status": "started_not_yet_verified_completed",
                            "wrapper_pid": process.pid, "command": command, "state": str(STATE_PATH),
                            "stdout": str(out), "stderr": str(err)})
        print(f"STOPPING_SUPERVISOR_STARTED: PID={process.pid}")
        print(f"STATE: {STATE_PATH}")
        return 0
    REC.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError("监督锁已存在；不要重复启动。先用status核对原监督器和状态。")
    LOCK_PATH.write_text(json.dumps({"pid": os.getpid(), "started_at": now()}, ensure_ascii=False), encoding="utf-8")
    pipeline = Pipeline()
    try:
        if args.mode == "formal-supervise":
            pipeline.attach_manual_formal_training()
            result = pipeline.formal_evaluation()
            receipt = pipeline.package(False, result["verdict"])
            pipeline.promote(receipt)
            pipeline.save("await_user_next_training" if result["verdict"] == "METHOD_HOLDS"
                          else "await_upper_interface" if result["verdict"] == "METHOD_DOES_NOT_HOLD"
                          else "stopped_fidelity")
        else:
            pipeline.run()
    except Exception as exc:
        pipeline.fail(str(exc))
        return 1
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
