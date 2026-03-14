"""Jinja 模板渲染."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_SYSTEM_MARKER = "---SYSTEM_PROMPT---"
_USER_MARKER = "---USER_PROMPT---"

_jinja_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "prompts")),
    trim_blocks=True, lstrip_blocks=True,
)


def render(template_name: str, **kwargs) -> tuple[str, str]:
    rendered = _jinja_env.get_template(f"{template_name}.jinja").render(**kwargs)
    sys_part, user_part = rendered.split(_USER_MARKER, 1)
    return sys_part.replace(_SYSTEM_MARKER, "").strip(), user_part.strip()
