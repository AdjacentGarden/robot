import json
import math
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ControllerCommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def short_output(self, limit: int = 3000) -> str:
        text = "\n".join(part for part in (self.stdout.strip(), self.stderr.strip()) if part)
        if len(text) <= limit:
            return text
        return text[-limit:]


class ROS2NavigationController:
    """Defensive wrapper around controller_cli.py for mapping and navigation."""

    _POINT_NAME_RE = re.compile(r"^[^\s\x00-\x1f\x7f]{1,64}$")

    def __init__(
        self,
        controller_cli_path,
        *,
        command_timeout_sec: float = 45.0,
        status_cache_sec: float = 1.5,
        startup_grace_sec: float = 1.5,
        goal_wait_subscribers_sec: float = 25.0,
        goal_publish_times: int = 5,
        mapping_cycles: int = 2,
        disable_rviz: bool = True,
    ):
        self.controller_cli_path = os.path.abspath(controller_cli_path)
        self.controller_dir = os.path.dirname(self.controller_cli_path)
        self.points_db_path = os.path.join(self.controller_dir, "named_points.json")
        self.command_timeout_sec = float(os.getenv("ROS2_CONTROLLER_CMD_TIMEOUT", command_timeout_sec))
        self.status_cache_sec = float(os.getenv("ROS2_CONTROLLER_STATUS_CACHE_SEC", status_cache_sec))
        self.startup_grace_sec = float(os.getenv("ROS2_CONTROLLER_STARTUP_GRACE_SEC", startup_grace_sec))
        self.goal_wait_subscribers_sec = float(os.getenv("ROS2_GOAL_WAIT_SUBSCRIBERS_SEC", goal_wait_subscribers_sec))
        self.goal_publish_times = max(1, int(os.getenv("ROS2_GOAL_PUBLISH_TIMES", goal_publish_times)))
        self.mapping_cycles = max(1, int(os.getenv("ROS2_MAPPING_NUM_CYCLES", mapping_cycles)))
        self.disable_rviz = str(os.getenv("ROS2_DISABLE_RVIZ", str(disable_rviz))).lower() not in {"0", "false", "no"}

        self.navigation_started = False
        self.mapping_started = False
        self.last_error: Optional[str] = None
        self.last_status: Dict[str, Any] = {}
        self.last_command_result: Optional[ControllerCommandResult] = None
        self._last_status_at = 0.0
        self._lock = threading.RLock()

        if not os.path.exists(self.controller_cli_path):
            self.last_error = f"controller_cli.py 不存在: {self.controller_cli_path}"
            print(f"[ROS2] {self.last_error}")
        else:
            self.sync_status(force=True)

    def _base_command(self) -> List[str]:
        return ["python3", self.controller_cli_path, "--points-db", self.points_db_path]

    def _command_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("RMW_FASTRTPS_USE_QOS_FROM_XML", "0")
        return env

    def _format_command(self, command: List[str]) -> str:
        try:
            return shlex.join(command)
        except Exception:
            return " ".join(command)

    def _run_cli(
        self,
        args: List[str],
        *,
        timeout_sec: Optional[float] = None,
        retries: int = 1,
        retry_delay_sec: float = 1.0,
    ) -> ControllerCommandResult:
        if not os.path.exists(self.controller_cli_path):
            result = ControllerCommandResult(
                command=self._base_command() + args,
                returncode=127,
                stdout="",
                stderr=f"controller_cli.py not found: {self.controller_cli_path}",
                elapsed_sec=0.0,
            )
            self.last_command_result = result
            self.last_error = result.stderr
            return result

        command = self._base_command() + list(args)
        timeout = self.command_timeout_sec if timeout_sec is None else float(timeout_sec)
        attempts = max(1, int(retries))
        last_result: Optional[ControllerCommandResult] = None

        for attempt in range(1, attempts + 1):
            start = time.time()
            print(f"[ROS2] 执行命令({attempt}/{attempts}): {self._format_command(command)}")
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.controller_dir,
                    env=self._command_env(),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                result = ControllerCommandResult(
                    command=command,
                    returncode=int(completed.returncode),
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                    elapsed_sec=time.time() - start,
                )
            except subprocess.TimeoutExpired as exc:
                result = ControllerCommandResult(
                    command=command,
                    returncode=124,
                    stdout=exc.stdout or "",
                    stderr=(exc.stderr or "") + f"\ncommand timeout after {timeout:.1f}s",
                    elapsed_sec=time.time() - start,
                    timed_out=True,
                )
            except Exception as exc:
                result = ControllerCommandResult(
                    command=command,
                    returncode=1,
                    stdout="",
                    stderr=f"command exception: {exc}",
                    elapsed_sec=time.time() - start,
                )

            last_result = result
            self.last_command_result = result
            if result.ok:
                self.last_error = None
                output = result.short_output(limit=2000)
                if output:
                    print(f"[ROS2] 命令输出:\n{output}")
                return result

            self.last_error = result.short_output(limit=2000) or f"returncode={result.returncode}"
            print(f"[ROS2] 命令失败: returncode={result.returncode}, elapsed={result.elapsed_sec:.1f}s")
            if self.last_error:
                print(f"[ROS2] 失败输出:\n{self.last_error}")
            if attempt < attempts:
                time.sleep(max(0.0, retry_delay_sec))

        return last_result or ControllerCommandResult(command=command, returncode=1, stdout="", stderr="unknown", elapsed_sec=0.0)

    def _rviz_args(self) -> List[str]:
        return ["--no-rviz"] if self.disable_rviz else ["--rviz"]

    def _named_point_args(self, action: str, point_name: str, extra_options: Optional[List[str]] = None) -> List[str]:
        # "--" prevents point names that start with "-" from being parsed as options.
        return (extra_options or []) + [action, "--", point_name]

    def _load_named_points(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.points_db_path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            self.last_error = f"读取命名点失败: {exc}"
            print(f"[ROS2] {self.last_error}")
            return {}
        return raw if isinstance(raw, dict) else {}

    def _validate_point_name(self, point_name: str) -> Optional[str]:
        name = str(point_name or "").strip()
        if not name:
            self.last_error = "目标点名称为空"
            return None
        if not self._POINT_NAME_RE.fullmatch(name):
            self.last_error = f"目标点名称非法: {name!r}"
            return None
        return name

    def _point_exists(self, point_name: str) -> bool:
        if point_name == "home":
            return True
        points = self._load_named_points()
        exists = point_name in points
        if not exists:
            self.last_error = f"命名点不存在: {point_name}"
            print(f"[ROS2] {self.last_error}")
            if points:
                print(f"[ROS2] 当前命名点: {', '.join(sorted(points.keys()))}")
        return exists

    def _parse_status_output(self, output: str) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "mode": None,
            "mapping_running": None,
            "navigation_running": None,
            "mapping_pid": None,
            "navigation_pid": None,
            "navigation_goal_reached": None,
            "navigation_has_goal": None,
            "mapping_finished": None,
        }
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if line.startswith("mode:"):
                status["mode"] = line.split(":", 1)[1].strip()
                continue
            match = re.match(r"^(mapping|navigation): running=(True|False), pid=([^,]+),", line)
            if match:
                name = match.group(1)
                status[f"{name}_running"] = match.group(2) == "True"
                pid_raw = match.group(3).strip()
                status[f"{name}_pid"] = int(pid_raw) if pid_raw.isdigit() else None
                continue
            for key in ("navigation_goal_reached", "navigation_has_goal", "mapping_finished"):
                if key in line:
                    value = line.rsplit("=", 1)[-1].strip().strip(",")
                    if value in {"True", "False"}:
                        status[key] = value == "True"
        return status

    def sync_status(self, *, force: bool = False) -> bool:
        with self._lock:
            now = time.time()
            if not force and (now - self._last_status_at) < self.status_cache_sec:
                return bool(self.last_status)

            result = self._run_cli(["status"], timeout_sec=15.0, retries=1)
            if not result.ok:
                return False

            parsed = self._parse_status_output(result.stdout)
            self.last_status = parsed
            self._last_status_at = time.time()

            if parsed.get("mapping_running") is not None:
                self.mapping_started = bool(parsed["mapping_running"])
            if parsed.get("navigation_running") is not None:
                self.navigation_started = bool(parsed["navigation_running"])
            return True

    def is_mapping_active(self, *, refresh: bool = False) -> bool:
        if refresh:
            self.sync_status(force=True)
        return bool(self.mapping_started)

    def is_navigation_active(self, *, refresh: bool = False) -> bool:
        if refresh:
            self.sync_status(force=True)
        return bool(self.navigation_started)

    def run_controller_command(self, action, target_name=None, x=None, y=None, yaw=None, frame_id="map"):
        """Backward-compatible command entry point."""
        action = str(action or "").strip()
        if not action:
            self.last_error = "ROS2 action 为空"
            return False

        if action == "mapping":
            return self.start_mapping()
        if action == "navigation":
            return self.start_navigation(force_restart=False)
        if action == "stop":
            return self.stop_all()
        if action == "status":
            return self.get_status()
        if action == "points":
            return self.list_points()
        if action == "delete-point":
            return self.delete_point(target_name)
        if action in {"goal", "goal-name"}:
            if x is not None and y is not None:
                return self.navigate_to_pose(x=x, y=y, yaw=yaw, frame_id=frame_id)
            return self.navigate_to_point(target_name)

        if target_name:
            return self.navigate_to_point(target_name)
        if x is not None and y is not None:
            return self.navigate_to_pose(x=x, y=y, yaw=yaw, frame_id=frame_id)

        # Old callers passed the point name as the action itself.
        return self.navigate_to_point(action)

    def start_mapping(self):
        """启动建图。建图与导航互斥，启动前会刷新一次状态。"""
        with self._lock:
            self.sync_status(force=True)
            if self.mapping_started:
                print("[ROS2] 建图已经在运行")
                return True

            args = (
                self._rviz_args()
                + ["--mapping-cycles", str(self.mapping_cycles), "--mapping-random-goals", "0", "mapping"]
            )
            result = self._run_cli(args, timeout_sec=70.0, retries=2, retry_delay_sec=2.0)
            if not result.ok:
                self.mapping_started = False
                return False

            time.sleep(self.startup_grace_sec)
            if not self.sync_status(force=True):
                self.mapping_started = True
            print("✅ 建图已启动")
            return True

    def start_navigation(self, force_restart: bool = False):
        """启动导航定位。默认不重启已运行的导航，避免丢失已收敛定位。"""
        with self._lock:
            self.sync_status(force=True)
            if self.navigation_started and not force_restart:
                print("[ROS2] 导航已经在运行，复用当前定位状态")
                return True

            restart_arg = "--restart" if force_restart else "--no-restart"
            args = self._rviz_args() + [restart_arg, "navigation"]
            result = self._run_cli(args, timeout_sec=70.0, retries=2, retry_delay_sec=2.0)
            if not result.ok:
                self.navigation_started = False
                return False

            time.sleep(self.startup_grace_sec)
            if not self.sync_status(force=True):
                self.navigation_started = True
                self.mapping_started = False
            print("✅ 导航已启动")
            return True

    def navigate_to_point(self, point_name):
        """导航到命名点。发送目标前会确认命名点和导航进程状态。"""
        name = self._validate_point_name(point_name)
        if name is None:
            print(f"[ROS2] {self.last_error}")
            return False
        if not self._point_exists(name):
            return False

        with self._lock:
            self.sync_status(force=True)
            if self.mapping_started:
                self.last_error = "正在建图，不能同时发送导航目标"
                print(f"[ROS2] {self.last_error}")
                return False

            if not self.navigation_started:
                print("[ROS2] 导航未启动，先启动导航...")
                if not self.start_navigation(force_restart=False):
                    return False

            extra = [
                "--wait-subscribers",
                str(self.goal_wait_subscribers_sec),
                "--publish-times",
                str(self.goal_publish_times),
            ]
            timeout = max(45.0, self.goal_wait_subscribers_sec + 25.0)
            result = self._run_cli(
                self._named_point_args("goal-name", name, extra_options=extra),
                timeout_sec=timeout,
                retries=2,
                retry_delay_sec=2.0,
            )
            if result.ok:
                self.navigation_started = True
                print(f"已发送导航目标: {name}")
                return True
            return False

    def navigate_to_pose(self, x, y, yaw=0.0, frame_id="map"):
        try:
            x_value = float(x)
            y_value = float(y)
            yaw_value = 0.0 if yaw is None else float(yaw)
        except (TypeError, ValueError):
            self.last_error = f"非法导航坐标: x={x}, y={y}, yaw={yaw}"
            print(f"[ROS2] {self.last_error}")
            return False

        if not math.isfinite(x_value) or not math.isfinite(y_value) or not math.isfinite(yaw_value):
            self.last_error = f"导航坐标不是有限数: x={x}, y={y}, yaw={yaw}"
            print(f"[ROS2] {self.last_error}")
            return False

        with self._lock:
            self.sync_status(force=True)
            if self.mapping_started:
                self.last_error = "正在建图，不能同时发送导航目标"
                print(f"[ROS2] {self.last_error}")
                return False
            if not self.navigation_started and not self.start_navigation(force_restart=False):
                return False

            args = [
                "--wait-subscribers",
                str(self.goal_wait_subscribers_sec),
                "--publish-times",
                str(self.goal_publish_times),
                "goal",
                "--x",
                str(x_value),
                "--y",
                str(y_value),
                "--yaw",
                str(yaw_value),
                "--frame-id",
                str(frame_id or "map"),
            ]
            timeout = max(45.0, self.goal_wait_subscribers_sec + 25.0)
            result = self._run_cli(args, timeout_sec=timeout, retries=2, retry_delay_sec=2.0)
            if result.ok:
                self.navigation_started = True
                print(f"✅ 已发送导航坐标: x={x_value:.3f}, y={y_value:.3f}, yaw={yaw_value:.3f}")
                return True
            return False

    def list_points(self):
        result = self._run_cli(["points"], timeout_sec=15.0, retries=1)
        return result.ok

    def delete_point(self, point_name):
        name = self._validate_point_name(point_name)
        if name is None:
            print(f"[ROS2] {self.last_error}")
            return False
        result = self._run_cli(self._named_point_args("delete-point", name), timeout_sec=20.0, retries=1)
        return result.ok

    def wait_navigation(self):
        result = self._run_cli(["wait"], timeout_sec=15.0, retries=1)
        if result.ok:
            self.sync_status(force=True)
        return result.ok

    def stop_all(self):
        """停止所有 ROS2 建图/导航进程。"""
        with self._lock:
            result = self._run_cli(["stop"], timeout_sec=80.0, retries=2, retry_delay_sec=2.0)
            if result.ok:
                self.navigation_started = False
                self.mapping_started = False
                self.sync_status(force=True)
                print("✅ 所有功能已停止")
                return True
            self.sync_status(force=True)
            return False

    def get_status(self):
        ok = self.sync_status(force=True)
        if ok:
            print(f"[ROS2] 状态: {self.last_status}")
        return ok
