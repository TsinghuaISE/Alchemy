"""SSH + Singularity 远程执行器."""

from __future__ import annotations

import atexit
import os
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import ExecutorBase


# ── SSH 连接管理 (ControlMaster 复用) ──


class _SSHManager:
    """SSH 连接管理器 (单例), 通过 ControlMaster 复用连接."""

    _instance: Optional[_SSHManager] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._connections: Dict[str, subprocess.Popen] = {}
        self._conn_lock = threading.Lock()
        self._socket_dir = Path(tempfile.gettempdir()) / f"ssh_control_{os.getpid()}"
        self._socket_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close_all)

    def ssh_options(self, node: str) -> List[str]:
        socket = str(self._socket_dir / f"{node}.sock")
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={socket}",
            "-o", "ControlPersist=86400",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=no",
        ]

    def ensure(self, node: str, timeout: int = 30) -> bool:
        with self._conn_lock:
            socket = str(self._socket_dir / f"{node}.sock")
            if self._check(node, socket):
                return True
            if os.path.exists(socket):
                try:
                    os.remove(socket)
                except OSError:
                    pass

            proc = subprocess.Popen(
                ["ssh", "-o", "ControlMaster=yes", "-o", f"ControlPath={socket}",
                 "-o", "ControlPersist=yes", "-o", f"ConnectTimeout={timeout}",
                 "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
                 "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                 "-fN", node],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            start = time.time()
            while time.time() - start < timeout:
                if self._check(node, socket):
                    self._connections[node] = proc
                    return True
                if proc.poll() is not None:
                    return False
                time.sleep(0.2)
            proc.kill()
            return False

    def ensure_nodes(self, nodes: List[str], timeout: int = 30,
                     retry: int = 2) -> Dict[str, bool]:
        results = {}
        for node in nodes:
            ok = False
            for attempt in range(retry + 1):
                if self.ensure(node, timeout):
                    ok = True
                    break
                time.sleep(1)
            results[node] = ok
        return results

    def close_all(self):
        with self._conn_lock:
            nodes = list(self._connections.keys())
        for node in nodes:
            socket = str(self._socket_dir / f"{node}.sock")
            try:
                subprocess.run(["ssh", "-O", "exit", "-o", f"ControlPath={socket}", node],
                               capture_output=True, timeout=5)
            except Exception:
                pass
            if node in self._connections:
                try:
                    self._connections[node].kill()
                except Exception:
                    pass
        self._connections.clear()
        try:
            if self._socket_dir.exists():
                for f in self._socket_dir.iterdir():
                    f.unlink(missing_ok=True)
                self._socket_dir.rmdir()
        except OSError:
            pass

    @staticmethod
    def _check(node: str, socket: str) -> bool:
        if not os.path.exists(socket):
            return False
        try:
            return subprocess.run(
                ["ssh", "-O", "check", "-o", f"ControlPath={socket}", node],
                capture_output=True, timeout=5).returncode == 0
        except Exception:
            return False


def get_ssh_manager() -> _SSHManager:
    return _SSHManager()


# ── Singularity 执行器 ──


class SSHSingularityExecutor(ExecutorBase):
    """通过 SSH + Singularity 在远程节点执行容器化命令."""

    def __init__(self, sif_path: str, bind_paths: List[str], node: str, *,
                 gpu_id: Optional[int] = None,
                 singularity_exe: str = "singularity",
                 containall: bool = False,
                 conda_env: Optional[str] = None,
                 use_control_master: bool = True):
        self.sif_path = sif_path
        self.bind_paths = bind_paths
        self.node = node
        self.gpu_id = gpu_id
        self.singularity_exe = singularity_exe
        self.containall = containall
        self.conda_env = conda_env
        self.use_control_master = use_control_master

    # ── ExecutorBase 接口 ──

    def _prepare_command(self, command: List[str], work_dir: str,
                         env: Dict[str, str]) -> List[str]:
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_ALL", "C.UTF-8")
        if self.gpu_id is not None:
            env.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_id))

        sing_cmd = self._build_singularity(command, work_dir, env)

        if self.use_control_master:
            opts = get_ssh_manager().ssh_options(self.node)
            return ["ssh"] + opts + [self.node, sing_cmd]
        return ["ssh", self.node, sing_cmd]

    def _start_process(self, prepared_cmd):
        return subprocess.Popen(
            prepared_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)

    def _stream_tag(self):
        return "singularity"

    # ── Singularity 内部 ──

    def _build_singularity(self, command: List[str], work_dir: str,
                           env: Dict[str, str]) -> str:
        cmd = [self.singularity_exe, "exec"]
        if self.containall:
            cmd.append("--containall")
        cmd.append("--nv")
        if work_dir:
            cmd.extend(["--pwd", work_dir])
        for b in self.bind_paths:
            cmd.extend(["-B", b])
        cmd.append(self.sif_path)

        inner = " ".join(shlex.quote(p) for p in command)
        exports = []
        if self.conda_env:
            exports.append(f"export PATH=/opt/conda/envs/{self.conda_env}/bin:$PATH;")
        for k, v in env.items():
            if k != "PATH":
                exports.append(f"export {k}={shlex.quote(str(v))};")
        if exports:
            inner = f"{' '.join(exports)} {inner}"
        cmd.extend(["bash", "-lc", shlex.quote(inner)])
        return " ".join(cmd)
