"""执行器基类 - 容器化命令执行的统一抽象."""

from __future__ import annotations

import selectors
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class CommandResult:
    return_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    cancelled: bool = False


class ExecutorBase(ABC):
    """容器执行器基类.

    子类实现 _prepare_command() 和 _start_process() 即可.
    """

    @abstractmethod
    def _prepare_command(self, command: List[str], work_dir: str,
                         env: Dict[str, str]) -> str | List[str]:
        """构建最终执行的命令."""

    @abstractmethod
    def _start_process(self, prepared_cmd: str | List[str]) -> subprocess.Popen:
        """启动子进程."""

    def _pre_execute(self) -> None:
        """执行前钩子 (如 Docker 镜像构建)."""

    def _post_execute(self) -> None:
        """执行后钩子 (如容器清理)."""

    def _stream_tag(self) -> str:
        return "executor"

    def execute(
        self,
        command: List[str],
        work_dir: str,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        streaming: bool = False,
        line_callback: Optional[Callable[[str, bool], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> CommandResult:
        self._pre_execute()

        start = time.time()
        env = dict(env or {})
        cancelled, timed_out = False, False

        process = self._start_process(self._prepare_command(command, work_dir, env))
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        sel = selectors.DefaultSelector()
        if process.stdout:
            sel.register(process.stdout, selectors.EVENT_READ)
        if process.stderr:
            sel.register(process.stderr, selectors.EVENT_READ)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    cancelled = True
                    process.kill()
                    break
                if timeout and (time.time() - start) > timeout:
                    timed_out = True
                    process.kill()
                    break
                if process.poll() is not None and not sel.get_map():
                    break

                for key, _ in sel.select(timeout=0.2):
                    line = key.fileobj.readline()
                    if not line:
                        sel.unregister(key.fileobj)
                        continue
                    line = line.rstrip("\n")
                    is_err = key.fileobj is process.stderr
                    (stderr_lines if is_err else stdout_lines).append(line)
                    if line_callback:
                        line_callback(line, is_err)
                    if streaming:
                        tag = self._stream_tag()
                        print(f"[{tag}]{'[stderr]' if is_err else ''} {line}")

                if process.poll() is not None and not sel.get_map():
                    break
        finally:
            sel.close()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            self._post_execute()

        rc = process.wait() if process.returncode is None else process.returncode
        return CommandResult(
            return_code=rc,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            duration=time.time() - start,
            timed_out=timed_out,
            cancelled=cancelled,
        )
