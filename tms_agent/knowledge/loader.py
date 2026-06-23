"""知识库加载与导入接口。

统一 chunk 结构:{id, title, text, tags}。
支持两种来源(用户后续放入即被索引,无需改代码):
- *.json:chunk 列表
- *.md :整文件作为一个 chunk(标题取首个标题行或文件名)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from .. import config


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)

    def searchable(self) -> str:
        """供检索的全文(标题+标签权重通过重复体现)。"""
        return f"{self.title} {self.title} {' '.join(self.tags)} {self.text}"


def load_chunks(docs_dir: Path | None = None) -> list[KnowledgeChunk]:
    docs_dir = Path(docs_dir) if docs_dir else config.KNOWLEDGE_DIR
    if not docs_dir.exists():
        return []

    chunks: list[KnowledgeChunk] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix == ".json":
            for item in json.loads(path.read_text(encoding="utf-8")):
                chunks.append(
                    KnowledgeChunk(
                        id=item.get("id", f"{path.stem}_{len(chunks)}"),
                        title=item.get("title", ""),
                        text=item.get("text", ""),
                        tags=list(item.get("tags", [])),
                    )
                )
        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8").strip()
            first = text.splitlines()[0] if text else path.stem
            title = first.lstrip("# ").strip() or path.stem
            chunks.append(KnowledgeChunk(id=path.stem, title=title, text=text))
    return chunks
