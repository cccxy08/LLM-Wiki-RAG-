"""将 Wiki 页面批量导入 ChromaDB 向量库"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rag_engine import RAGEngine
from core.config import settings

WIKI_DIR = Path(settings.wiki_pages_dir)
EXCLUDE = {"index.md", "log.md"}

def main():
    rag = RAGEngine()
    md_files = sorted(WIKI_DIR.glob("*.md"))

    total_chunks = 0
    total_files = 0

    for fp in md_files:
        if fp.name in EXCLUDE:
            print(f"  ⏭️  跳过: {fp.name}")
            continue
        try:
            chunks = rag.index_document(fp)
            total_chunks += chunks
            total_files += 1
            print(f"  ✅ {fp.name} → {chunks} 个切片")
        except Exception as e:
            print(f"  ❌ {fp.name} 导入失败: {e}")

    count = rag.collection_count()
    print(f"\n📊 导入完成: {total_files} 个文件, {total_chunks} 个切片")
    print(f"📊 向量库总量: {count} 条")

if __name__ == "__main__":
    main()
