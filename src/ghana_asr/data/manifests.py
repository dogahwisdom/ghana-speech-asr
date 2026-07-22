"""Manifest preparation + materialized training cache for efficient I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ghana_asr.config import DataConfig
from ghana_asr.utils.logging import get_logger

logger = get_logger(__name__)


def _stable_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _assign_split(source_file: str, language: str, seed: int, ratios: dict[str, float]) -> str:
    bucket = _stable_bucket(f"{language}|{source_file}", seed)
    train_end = ratios["train"]
    val_end = train_end + ratios["validation"]
    if bucket < train_end:
        return "train"
    if bucket < val_end:
        return "validation"
    return "test"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _index_subset(root: Path, subset: str) -> pl.DataFrame:
    shard_dir = root / subset
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Missing subset directory: {shard_dir}")

    parts: list[pl.DataFrame] = []
    for parquet_path in sorted(shard_dir.glob("*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["id", "language", "text", "duration", "source_file"],
        )
        df = pl.from_arrow(table).with_columns(
            [
                pl.lit(subset).alias("subset"),
                pl.lit(str(parquet_path)).alias("parquet_path"),
                pl.arange(0, table.num_rows).alias("row_index"),
            ]
        )
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No parquet shards in {shard_dir}")
    return pl.concat(parts, how="vertical_relaxed")


def _materialize_split(split_df: pl.DataFrame, out_path: Path) -> None:
    """Write a dense cache containing only selected rows (incl. audio bytes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    # Group by shard so each source file is opened once.
    for (parquet_path,), group in tqdm(
        list(split_df.group_by(["parquet_path"])),
        desc=f"cache:{out_path.stem}",
    ):
        wanted = set(group["id"].to_list())
        table = pq.read_table(parquet_path)
        # Filter to wanted ids
        id_col = table.column("id").to_pylist()
        keep = [i for i, sid in enumerate(id_col) if sid in wanted]
        if not keep:
            continue
        filtered = table.take(keep)
        # Attach subset column if missing
        if "subset" not in filtered.column_names:
            subset_val = group["subset"][0]
            filtered = filtered.append_column("subset", pa.array([subset_val] * filtered.num_rows))
        if writer is None:
            writer = pq.ParquetWriter(
                out_path,
                filtered.schema,
                compression="zstd",
            )
        writer.write_table(filtered)
    if writer is None:
        # Empty split — write empty schema placeholder from metadata-only frame
        empty = split_df.select(
            ["id", "language", "text", "duration", "source_file", "subset"]
        ).to_arrow()
        pq.write_table(empty, out_path)
    else:
        writer.close()


def build_manifests(cfg: DataConfig, seed: int, materialize: bool = True) -> dict[str, Path]:
    root = Path(cfg.root)
    out_dir = Path(cfg.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pl.DataFrame] = []
    for subset in cfg.subsets:
        logger.info("Indexing subset %s", subset)
        df = _index_subset(root, subset)
        df = df.filter(
            (pl.col("duration") >= cfg.min_duration_s)
            & (pl.col("duration") <= cfg.max_duration_s)
            & pl.col("text").is_not_null()
            & (pl.col("text").str.len_chars() > 0)
        )
        df = df.with_columns(
            pl.col("text").map_elements(_normalize_text, return_dtype=pl.String).alias("text")
        )
        frames.append(df)
        logger.info("  kept %s clips after filters", f"{len(df):,}")

    all_df = pl.concat(frames, how="vertical_relaxed")
    ratios = {
        "train": cfg.split_ratios.train,
        "validation": cfg.split_ratios.validation,
        "test": cfg.split_ratios.test,
    }
    splits = [
        _assign_split(sf, lang, seed, ratios)
        for sf, lang in zip(all_df["source_file"].to_list(), all_df["language"].to_list())
    ]
    all_df = all_df.with_columns(pl.Series("split", splits))

    capped_parts: list[pl.DataFrame] = []
    for (subset,), group in all_df.group_by(["subset"]):
        for split_name, max_hours in (
            ("train", cfg.max_train_hours_per_language),
            ("validation", cfg.max_eval_hours_per_language),
            ("test", cfg.max_eval_hours_per_language),
        ):
            part = group.filter(pl.col("split") == split_name).sort("id")
            if len(part) == 0:
                continue
            if max_hours > 0:
                part = part.with_columns(pl.col("duration").cum_sum().alias("_cum"))
                part = part.filter(pl.col("_cum") <= max_hours * 3600.0).drop("_cum")
            capped_parts.append(part)

    final_df = pl.concat(capped_parts, how="vertical_relaxed")

    paths: dict[str, Path] = {}
    summary: dict[str, dict[str, float | int]] = {}
    for split_name in ("train", "validation", "test"):
        split_df = final_df.filter(pl.col("split") == split_name)
        meta_path = out_dir / f"{split_name}.parquet"
        split_df.write_parquet(meta_path)
        paths[split_name] = meta_path
        hours = float(split_df["duration"].sum()) / 3600.0 if len(split_df) else 0.0
        summary[split_name] = {"clips": len(split_df), "hours": round(hours, 3)}
        logger.info(
            "Wrote %s meta (%s clips, %.2f h) -> %s",
            split_name,
            f"{len(split_df):,}",
            hours,
            meta_path,
        )
        if materialize:
            cache_path = out_dir / f"{split_name}_cache.parquet"
            logger.info("Materializing dense audio cache -> %s", cache_path)
            _materialize_split(split_df, cache_path)
            paths[f"{split_name}_cache"] = cache_path

    meta = {
        "author": "Wisdom Dogah",
        "seed": seed,
        "subsets": cfg.subsets,
        "filters": {
            "min_duration_s": cfg.min_duration_s,
            "max_duration_s": cfg.max_duration_s,
            "max_train_hours_per_language": cfg.max_train_hours_per_language,
            "max_eval_hours_per_language": cfg.max_eval_hours_per_language,
        },
        "summary": summary,
        "by_language": {},
        "materialized": materialize,
    }
    for (subset, split_name), g in final_df.group_by(["subset", "split"]):
        key = f"{subset}/{split_name}"
        meta["by_language"][key] = {
            "clips": len(g),
            "hours": round(float(g["duration"].sum()) / 3600.0, 3),
        }

    meta_path = out_dir / "manifest_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    logger.info("Manifest metadata -> %s", meta_path)
    return paths
