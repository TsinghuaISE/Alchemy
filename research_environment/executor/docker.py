"""本地 Docker 执行器."""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

from research_environment.executor.base import ExecutorBase


class DockerExecutor(ExecutorBase):
    """通过 Docker 执行容器化命令.

    特性:
    - 自动构建镜像 (按镜像名去重, 多线程安全)
    - 执行后自动清理容器
    - GPU 透传
    """

    _build_locks: Dict[str, threading.Lock] = {}
    _build_guard = threading.Lock()
    _built: Dict[str, bool] = {}

    def __init__(self, image: str, bind_paths: List[str], *,
                 gpu_id: Optional[int] = None,
                 conda_env: Optional[str] = None,
                 extra_args: Optional[List[str]] = None,
                 local_root: Optional[str] = None,
                 remote_root: Optional[str] = None,
                 dockerfile_context: Optional[str] = None):
        self.image = image
        self.bind_paths = bind_paths
        self.gpu_id = gpu_id
        self.conda_env = conda_env
        self.extra_args = extra_args or []
        self.local_root = local_root
        self.remote_root = remote_root
        self.dockerfile_context = dockerfile_context
        self._container_name: Optional[str] = None

    # ── ExecutorBase 接口 ──

    def _pre_execute(self):
        self._ensure_image()
        self._container_name = f"resenv_{os.getpid()}_{threading.current_thread().ident}_{self.gpu_id or 0}"

    def _post_execute(self):
        if self._container_name:
            try:
                subprocess.run(["docker", "rm", "-f", self._container_name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except Exception:
                pass

    def _prepare_command(self, command: List[str], work_dir: str,
                         env: Dict[str, str]) -> str:
        if self.gpu_id is not None:
            env.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_id))

        binds = []
        for bind in self.bind_paths:
            parts = bind.split(":")
            host = self._remap_path(parts[0])
            rest = ":" + ":".join(parts[1:]) if len(parts) > 1 else ""
            binds.append(f"{host}{rest}")

        return self._build_cmd(command, work_dir, env, binds)

    def _start_process(self, prepared_cmd):
        return subprocess.Popen(
            prepared_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, shell=True, executable="/bin/bash")

    def _stream_tag(self):
        return "docker"

    # ── Docker 内部 ──

    def _remap_path(self, path: str) -> str:
        """将 remote_root 路径映射为 local_root 路径."""
        if self.local_root and self.remote_root and path.startswith(self.remote_root):
            rel = path[len(self.remote_root):].lstrip("/")
            return str(Path(self.local_root) / rel)
        return path

    def _ensure_image(self):
        if self.image in self._built:
            return
        if self._image_exists():
            self._built[self.image] = True
            return

        with self._build_guard:
            if self.image not in self._build_locks:
                self._build_locks[self.image] = threading.Lock()

        with self._build_locks[self.image]:
            if self.image in self._built or self._image_exists():
                self._built[self.image] = True
                return
            if not self.dockerfile_context:
                print(f"[docker] WARNING: 镜像 {self.image} 不存在且未配置 Dockerfile")
                return

            dockerfile = Path(self.dockerfile_context) / "Dockerfile"
            if not dockerfile.exists():
                print(f"[docker] WARNING: Dockerfile 不存在: {dockerfile}")
                return

            print(f"[docker] 构建镜像 {self.image} ...")
            proc = subprocess.run(
                ["docker", "build", "-t", self.image, "-f", str(dockerfile),
                 str(self.dockerfile_context)],
                text=True, timeout=3600)
            self._built[self.image] = proc.returncode == 0

    @staticmethod
    def _image_exists(image: str = None) -> bool:
        try:
            return subprocess.run(
                ["docker", "image", "inspect", image or ""],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except FileNotFoundError:
            return False

    def _image_exists(self) -> bool:
        try:
            return subprocess.run(
                ["docker", "image", "inspect", self.image],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except FileNotFoundError:
            return False

    def _build_cmd(self, command: List[str], work_dir: str,
                   env: Dict[str, str], binds: List[str]) -> str:
        cmd = ["docker", "run", "--rm", "--gpus", "all"]
        if self._container_name:
            cmd.extend(["--name", self._container_name])
        if work_dir:
            cmd.extend(["-w", work_dir])
        for b in binds:
            cmd.extend(["-v", b])
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.extend(self.extra_args)
        cmd.append(self.image)

        inner = " ".join(shlex.quote(p) for p in command)
        if self.conda_env:
            inner = f"export PATH=/opt/conda/envs/{self.conda_env}/bin:$PATH; {inner}"
        cmd.extend(["bash", "-c", shlex.quote(inner)])
        return " ".join(cmd)
