"""
app/agent/vector_store.py

Chroma 轻量级本地向量持久化索引封装：
  - 数据持久化在 data/chroma_db（测试可通过参数注入临时目录）；
  - 禁用 Chroma 默认 ONNX 模型下载，强制由外部在线 EmbeddingProvider 显式提供向量；
  - 通过 index_manifest.json 记录文档内容 SHA-256 哈希与模型标识，
    仅在索引为空、模型变更或文档内容变化时才调用 Embedding API 做增量/全量建索引。
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.llm.embedding import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "smart_cs_knowledge"
_MANIFEST_FILENAME = "index_manifest.json"


class _DisabledDefaultEmbeddingFunction:
    """
    占位 EmbeddingFunction（兼容 Chroma 1.x 协议）：
    阻止 Chroma 隐式加载或下载默认的 ONNXMiniLM_L6_V2 本地模型。
    本模块所有写入与查询均显式传入 embeddings / query_embeddings，正常路径永远不会触发 __call__。
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        raise RuntimeError(
            "禁止触发 Chroma 本地默认 ONNX 模型，请通过在线 EmbeddingProvider 显式传入向量。"
        )

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "manual_external_embedding"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_DisabledDefaultEmbeddingFunction":
        return _DisabledDefaultEmbeddingFunction()

    def validate_config_update(self, old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
        return

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        return

    def is_legacy(self) -> bool:
        return False


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_embed_text(chunk: dict[str, Any]) -> str:
    """将章节标题与正文拼接作为向量化输入，确保语义向量同时覆盖标题与细节条款。"""
    return f"{chunk['section']}\n{chunk['content']}"


class ChromaVectorStore:
    """面向知识库切片的 Chroma 持久化向量索引。"""

    def __init__(self, persist_dir: Path, collection_name: str = _COLLECTION_NAME) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._manifest_path = persist_dir / _MANIFEST_FILENAME
        self._client: Any = None
        self._collection: Any = None

    def _ensure_collection(self) -> Any:
        """懒加载 Chroma PersistentClient 与余弦距离集合（关闭匿名遥测）。"""
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.config import Settings

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_DisabledDefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _recreate_collection(self) -> Any:
        """重建集合（用于模型切换、维度变化或索引损坏时的全量刷新）。"""
        self._ensure_collection()
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            embedding_function=_DisabledDefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("vector_store: 读取索引清单失败，将重建索引: %s", e)
            return {}

    def _save_manifest(self, model_id: str, corpus_hash: str, chunk_hashes: dict[str, str]) -> None:
        payload = {
            "model": model_id,
            "corpus_hash": corpus_hash,
            "chunks": chunk_hashes,
        }
        self._manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def sync_chunks(
        self,
        chunks: list[dict[str, Any]],
        provider: BaseEmbeddingProvider,
    ) -> bool:
        """
        将当前知识库切片同步到 Chroma 持久化索引。
          - 内容哈希与模型标识一致且条数匹配：直接复用本地索引，不调用 Embedding API；
          - 模型未变且已有索引：仅对新增/修改的切片调用 Embedding API 增量 upsert，删除已移除切片；
          - 索引为空或模型变更：全量向量化并重建集合。
        返回 True 表示向量索引可用；发生任何异常时捕获、打 WARNING 并返回 False（触发纯 BM25 降级）。
        """
        if not chunks:
            return False

        try:
            collection = self._ensure_collection()
            model_id = f"{provider.name}:{provider.model}"

            chunk_hashes: dict[str, str] = {
                c["chunk_id"]: _sha256_text(f"{model_id}\n{_build_embed_text(c)}")
                for c in chunks
            }
            corpus_hash = _sha256_text(
                json.dumps(sorted(chunk_hashes.items()), ensure_ascii=False)
            )

            manifest = self._load_manifest()
            existing_count = int(collection.count())

            # 快速路径：模型与全文哈希完全一致且持久化条数完整，零 API 调用直接复用
            if (
                manifest.get("model") == model_id
                and manifest.get("corpus_hash") == corpus_hash
                and existing_count == len(chunks)
            ):
                logger.info(
                    "vector_store: 知识库内容未变更，直接复用本地 Chroma 持久化索引（%d 块，0 次 Embedding 调用）",
                    existing_count,
                )
                return True

            old_chunks: dict[str, str] = manifest.get("chunks") if isinstance(manifest.get("chunks"), dict) else {}
            same_model = manifest.get("model") == model_id and existing_count > 0 and len(old_chunks) == existing_count

            if same_model:
                # 增量同步路径：仅向量化新增或内容变更的切片，节省在线 Embedding 费用
                deleted_ids = [cid for cid in old_chunks if cid not in chunk_hashes]
                changed_chunks = [c for c in chunks if old_chunks.get(c["chunk_id"]) != chunk_hashes[c["chunk_id"]]]

                if deleted_ids:
                    collection.delete(ids=deleted_ids)

                if changed_chunks:
                    texts = [_build_embed_text(c) for c in changed_chunks]
                    vectors = provider.embed_texts(texts)
                    collection.upsert(
                        ids=[c["chunk_id"] for c in changed_chunks],
                        embeddings=vectors,
                        documents=texts,
                        metadatas=[{"source": c["source"], "section": c["section"]} for c in changed_chunks],
                    )
                logger.info(
                    "vector_store: Chroma 增量更新完成（新增/变更 %d 块，删除 %d 块，总计 %d 块）",
                    len(changed_chunks),
                    len(deleted_ids),
                    len(chunks),
                )
            else:
                # 全量构建路径：先算完向量再重建集合，避免 API 失败导致旧索引被清空
                texts = [_build_embed_text(c) for c in chunks]
                vectors = provider.embed_texts(texts)
                collection = self._recreate_collection()
                collection.upsert(
                    ids=[c["chunk_id"] for c in chunks],
                    embeddings=vectors,
                    documents=texts,
                    metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
                )
                logger.info(
                    "vector_store: Chroma 全量索引构建完成（%d 块，model=%s，目录=%s）",
                    len(chunks),
                    model_id,
                    self._persist_dir,
                )

            self._save_manifest(model_id, corpus_hash, chunk_hashes)
            return True
        except Exception as e:
            logger.warning("vector_store: Chroma 同步/建索引失败，自动降级为纯 BM25: %s", e)
            return False

    def query_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 6,
        min_similarity: float = 0.58,
        max_drop: float | None = 0.10,
    ) -> list[tuple[str, float]]:
        """
        使用已计算好的查询向量直接执行 Chroma 相似度检索（便于上层独立缓存向量并精确统计 chroma_ms）。
        余弦距离 distance = 1 - cosine_similarity，Pre-RRF 双重门控：
          1. 绝对阈值门控：过滤 similarity < min_similarity 的弱相关项；
          2. 相对衰减门控：当 max_drop 不为 None 时，过滤与 Top-1 相似度落差超过 max_drop 的边缘蹭分项。
        """
        collection = self._ensure_collection()
        total = int(collection.count())
        if total == 0 or not query_vector:
            return []

        n_results = min(max(top_k, 1), total)
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["distances"],
        )
        ids = (raw.get("ids") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        scored = [
            (str(chunk_id), 1.0 - float(dist))
            for chunk_id, dist in zip(ids, distances)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            return []

        top1_sim = scored[0][1]
        hits: list[tuple[str, float]] = []
        for chunk_id, similarity in scored:
            if similarity < min_similarity:
                continue
            if max_drop is not None and (top1_sim - similarity) > max_drop:
                continue
            hits.append((chunk_id, similarity))
        return hits

    def query(
        self,
        query_text: str,
        provider: BaseEmbeddingProvider,
        top_k: int = 6,
        min_similarity: float = 0.58,
        max_drop: float | None = 0.10,
    ) -> list[tuple[str, float]]:
        """执行单路向量语义检索（在线向量化 + Chroma 查询一体化便捷封装）。"""
        vectors = provider.embed_texts([query_text])
        if not vectors:
            return []
        return self.query_by_vector(
            query_vector=vectors[0],
            top_k=top_k,
            min_similarity=min_similarity,
            max_drop=max_drop,
        )
