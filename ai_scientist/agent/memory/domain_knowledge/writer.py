from __future__ import annotations

from pathlib import Path


START_MARKER = "<!-- LEARNED_KNOWLEDGE:START -->"
END_MARKER = "<!-- LEARNED_KNOWLEDGE:END -->"


def render_base_document(domain: str, task: str) -> str:
    return f"# Domain Knowledge\n\n- Domain: {domain}\n- Task: {task}\n"


def replace_learned_block(existing: str, learned_markdown: str) -> str:
    learned_block = (
        f"{START_MARKER}\n"
        f"{learned_markdown.strip()}\n"
        f"{END_MARKER}\n"
    )

    if START_MARKER in existing and END_MARKER in existing:
        prefix, remainder = existing.split(START_MARKER, 1)
        _, suffix = remainder.split(END_MARKER, 1)
        prefix = prefix.rstrip()
        suffix = suffix.lstrip("\n")
        rebuilt = prefix + "\n\n" + learned_block
        if suffix:
            rebuilt += "\n" + suffix
        return rebuilt.rstrip() + "\n"

    base = existing.rstrip()
    if base:
        base += "\n\n"
    return base + learned_block


def write_knowledge_file(path: Path, domain: str, task: str, learned_markdown: str) -> str:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = render_base_document(domain, task)

    updated = replace_learned_block(existing, learned_markdown)
    path.write_text(updated, encoding="utf-8")
    return updated
