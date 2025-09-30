"""Service for handling user-uploaded tabular datasets."""

import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from werkzeug.datastructures import FileStorage

from ..config import Config


class UploadedDatasetService:
    """Manage ingestion and querying of uploaded CSV datasets."""

    def __init__(self) -> None:
        self.base_dir = Config.UPLOAD_BASE_DIR
        self.chunk_size = max(1, Config.UPLOAD_CHUNK_SIZE)
        self.sample_row_budget = max(1, Config.SAMPLE_ROWS_PER_DATASET)
        os.makedirs(self.base_dir, exist_ok=True)

    def ingest_files(self, files: Iterable[FileStorage]) -> Dict[str, Any]:
        """Persist uploaded CSV files in manageable chunks and build metadata."""
        file_list = [f for f in files if f and getattr(f, "filename", "")]  # type: ignore[arg-type]
        if not file_list:
            raise ValueError("No CSV files were provided")

        if len(file_list) > Config.MAX_UPLOAD_FILES:
            raise ValueError(f"Too many files: limit is {Config.MAX_UPLOAD_FILES}")

        dataset_id = str(uuid.uuid4())
        dataset_dir = os.path.join(self.base_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        numeric_summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "min": None,
            "max": None,
            "sum": 0.0,
            "count": 0,
            "sample_values": []
        })
        categorical_summary: Dict[str, Counter] = defaultdict(Counter)

        column_types: Dict[str, str] = {}
        numeric_columns: List[str] = []
        categorical_columns: List[str] = []
        datetime_columns: List[str] = []
        sample_rows: List[Dict[str, Any]] = []
        chunk_manifest: List[Dict[str, Any]] = []
        original_files: List[Dict[str, Any]] = []
        total_rows = 0
        chunk_index = 0

        try:
            for file_index, storage in enumerate(file_list):
                original_name = os.path.basename(storage.filename or f"dataset_{file_index}.csv")
                if not original_name.lower().endswith(".csv"):
                    raise ValueError(f"Unsupported file type for '{original_name}'. Only CSV files are allowed.")
                stored_name = f"{file_index:03d}_{original_name}"
                stored_path = os.path.join(dataset_dir, stored_name)
                storage.save(stored_path)

                file_row_count = 0

                for chunk in self._read_csv_in_chunks(stored_path):
                    if chunk.empty:
                        continue

                    chunk = chunk.convert_dtypes()
                    inferred_datetime = self._detect_datetime_columns(chunk, set(datetime_columns))
                    datetime_columns = self._merge_unique(datetime_columns, inferred_datetime)

                    numeric_cols = chunk.select_dtypes(include=["number", "Float64", "Int64", "Float32", "Int32"]).columns.tolist()
                    numeric_columns = self._merge_unique(numeric_columns, numeric_cols)

                    object_like_cols = [c for c in chunk.columns if c not in numeric_cols]
                    categorical_candidates = [c for c in object_like_cols if c not in inferred_datetime]
                    categorical_columns = self._merge_unique(categorical_columns, categorical_candidates)

                    for column, dtype in chunk.dtypes.items():
                        dtype_str = str(dtype)
                        column_types[column] = column_types.get(column, dtype_str)

                    self._update_numeric_summary(chunk, numeric_cols, numeric_summary)
                    self._update_categorical_summary(chunk, categorical_candidates, categorical_summary)

                    rows_needed = self.sample_row_budget - len(sample_rows)
                    if rows_needed > 0:
                        sample_rows.extend(chunk.head(rows_needed).to_dict("records"))

                    chunk_rows = len(chunk)
                    total_rows += chunk_rows
                    file_row_count += chunk_rows

                    chunk_name = f"chunk_{chunk_index:05d}.parquet"
                    chunk_path = os.path.join(dataset_dir, chunk_name)
                    chunk.to_parquet(chunk_path, index=False)

                    chunk_manifest.append({
                        "chunk_id": chunk_index,
                        "file_name": chunk_name,
                        "row_count": chunk_rows,
                        "source_file": stored_name
                    })
                    chunk_index += 1

                original_files.append({
                    "file_name": original_name,
                    "stored_as": stored_name,
                    "row_count": file_row_count
                })

            metadata = self._build_metadata(
                dataset_id=dataset_id,
                original_files=original_files,
                chunk_manifest=chunk_manifest,
                total_rows=total_rows,
                column_types=column_types,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                datetime_columns=datetime_columns,
                numeric_summary=numeric_summary,
                categorical_summary=categorical_summary,
                sample_rows=sample_rows
            )

            self._write_metadata(dataset_dir, metadata)
            return self._public_ingest_response(metadata)

        except Exception:
            self._safe_cleanup(dataset_dir)
            raise

    def answer_question(self, dataset_id: str, question: str, llm_service) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        metadata = self.load_metadata(dataset_id)
        selected = self._select_relevant_columns(question, metadata)
        reasoning_sources = self._build_reasoning_sources(selected, metadata)
        charts = self._recommend_charts(question, selected, metadata)

        zero_shot_prompt = self._build_zero_shot_prompt(question, metadata)
        cot_prompt = self._build_cot_prompt(question, metadata, reasoning_sources, charts)

        if llm_service and getattr(llm_service, "is_available", lambda: False)():
            answer = llm_service.generate_dataset_response(zero_shot_prompt, cot_prompt)
        else:
            answer = self._baseline_answer(question, reasoning_sources, charts)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "question": question,
            "answer": answer,
            "reasoning_sources": reasoning_sources,
            "chart_suggestions": charts,
            "zero_shot_prompt": zero_shot_prompt,
            "cot_prompt": cot_prompt,
            "selected_columns": selected
        }

    def load_metadata(self, dataset_id: str) -> Dict[str, Any]:
        metadata_path = os.path.join(self.base_dir, dataset_id, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_csv_in_chunks(self, path: str) -> Iterable[pd.DataFrame]:
        try:
            return pd.read_csv(path, chunksize=self.chunk_size)
        except pd.errors.EmptyDataError:
            return []

    def _detect_datetime_columns(self, frame: pd.DataFrame, known_columns: Iterable[str]) -> List[str]:
        detected: List[str] = list(known_columns)
        remaining = [col for col in frame.columns if col not in detected]

        for column in remaining:
            series = frame[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                detected.append(column)
                continue

            if not pd.api.types.is_object_dtype(series):
                continue

            sample = series.dropna().astype(str).head(50)
            if sample.empty:
                continue

            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.6:
                detected.append(column)

        return detected

    def _merge_unique(self, current: List[str], candidates: Iterable[str]) -> List[str]:
        existing = set(current)
        for candidate in candidates:
            if candidate not in existing:
                current.append(candidate)
                existing.add(candidate)
        return current

    def _update_numeric_summary(self, frame: pd.DataFrame, columns: Iterable[str], summary: Dict[str, Dict[str, Any]]) -> None:
        for column in columns:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            numeric_series = numeric_series.dropna()
            if numeric_series.empty:
                continue

            stats = summary[column]
            col_min = float(numeric_series.min())
            col_max = float(numeric_series.max())

            stats["min"] = col_min if stats["min"] is None else float(min(stats["min"], col_min))
            stats["max"] = col_max if stats["max"] is None else float(max(stats["max"], col_max))
            stats["sum"] += float(numeric_series.sum())
            stats["count"] += int(numeric_series.count())

            if len(stats["sample_values"]) < 100:
                sample_count = min(5, len(numeric_series))
                stats["sample_values"].extend(numeric_series.sample(sample_count, random_state=42).tolist())

    def _update_categorical_summary(self, frame: pd.DataFrame, columns: Iterable[str], summary: Dict[str, Counter]) -> None:
        for column in columns:
            series = frame[column].dropna().astype(str)
            if series.empty:
                continue
            counts = series.value_counts().head(20)
            summary[column].update(counts.to_dict())

    def _build_metadata(
        self,
        *,
        dataset_id: str,
        original_files: List[Dict[str, Any]],
        chunk_manifest: List[Dict[str, Any]],
        total_rows: int,
        column_types: Dict[str, str],
        numeric_columns: List[str],
        categorical_columns: List[str],
        datetime_columns: List[str],
        numeric_summary: Dict[str, Dict[str, Any]],
        categorical_summary: Dict[str, Counter],
        sample_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        numeric_profile: Dict[str, Dict[str, Any]] = {}
        for column, stats in numeric_summary.items():
            count = stats["count"]
            if count == 0:
                continue
            mean_value = stats["sum"] / count if count else None
            median_value = float(np.median(stats["sample_values"])) if stats["sample_values"] else None
            numeric_profile[column] = {
                "min": float(stats["min"]) if stats["min"] is not None else None,
                "max": float(stats["max"]) if stats["max"] is not None else None,
                "mean": float(mean_value) if mean_value is not None else None,
                "median": median_value,
                "count": int(stats["count"]),
                "sample_values": stats["sample_values"][:10]
            }

        categorical_profile: Dict[str, List[Dict[str, Any]]] = {}
        for column, counter in categorical_summary.items():
            top_values = counter.most_common(10)
            categorical_profile[column] = [
                {"value": value, "count": int(count)} for value, count in top_values
            ]

        return {
            "dataset_id": dataset_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "original_files": original_files,
            "chunk_manifest": chunk_manifest,
            "total_rows": total_rows,
            "column_types": column_types,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "numeric_summary": numeric_profile,
            "categorical_summary": categorical_profile,
            "sample_rows": sample_rows
        }

    def _write_metadata(self, dataset_dir: str, metadata: Dict[str, Any]) -> None:
        path = os.path.join(dataset_dir, "metadata.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _public_ingest_response(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset_id": metadata["dataset_id"],
            "total_rows": metadata["total_rows"],
            "column_types": metadata["column_types"],
            "numeric_columns": metadata["numeric_columns"],
            "categorical_columns": metadata["categorical_columns"],
            "datetime_columns": metadata["datetime_columns"],
            "chunk_count": len(metadata.get("chunk_manifest", [])),
            "original_files": [
                {"file_name": item["file_name"], "row_count": item.get("row_count", 0)}
                for item in metadata.get("original_files", [])
            ],
            "numeric_summary": metadata.get("numeric_summary", {}),
            "categorical_summary": metadata.get("categorical_summary", {}),
            "sample_rows": metadata.get("sample_rows", [])
        }

    def _safe_cleanup(self, dataset_dir: str) -> None:
        if not os.path.isdir(dataset_dir):
            return
        for root, _, files in os.walk(dataset_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except OSError:
                    pass
            try:
                os.rmdir(root)
            except OSError:
                pass

    def _select_relevant_columns(self, question: str, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        question_lower = question.lower()
        selected_numeric: List[str] = []
        selected_categorical: List[str] = []
        selected_datetime: List[str] = []

        def pick_columns(candidates: List[str], max_count: int, accumulator: List[str]) -> None:
            for candidate in candidates:
                if candidate.lower() in question_lower and candidate not in accumulator:
                    accumulator.append(candidate)
                if len(accumulator) >= max_count:
                    break

        pick_columns(metadata.get("numeric_columns", []), 3, selected_numeric)
        pick_columns(metadata.get("categorical_columns", []), 3, selected_categorical)
        pick_columns(metadata.get("datetime_columns", []), 2, selected_datetime)

        trend_keywords = ["trend", "over time", "time series", "season", "forecast", "timeline"]
        if not selected_datetime and any(k in question_lower for k in trend_keywords):
            selected_datetime = metadata.get("datetime_columns", [])[:1]

        comparison_keywords = ["compare", "versus", "vs", "top", "most", "least", "rank"]
        if not selected_categorical and any(k in question_lower for k in comparison_keywords):
            selected_categorical = metadata.get("categorical_columns", [])[:2]

        if not selected_numeric and metadata.get("numeric_columns"):
            selected_numeric = metadata["numeric_columns"][:2]

        return {
            "numeric": selected_numeric,
            "categorical": selected_categorical,
            "datetime": selected_datetime
        }

    def _build_reasoning_sources(self, selected: Dict[str, List[str]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []

        for column in selected.get("numeric", []):
            stats = metadata.get("numeric_summary", {}).get(column)
            if not stats:
                continue
            sources.append({
                "type": "numeric_summary",
                "column": column,
                "stats": stats
            })

        for column in selected.get("categorical", []):
            values = metadata.get("categorical_summary", {}).get(column)
            if not values:
                continue
            sources.append({
                "type": "categorical_distribution",
                "column": column,
                "top_values": values
            })

        sample_rows = metadata.get("sample_rows", [])
        if sample_rows:
            sources.append({
                "type": "sample_rows",
                "rows": sample_rows[: min(5, len(sample_rows))]
            })

        return sources

    def _recommend_charts(
        self,
        question: str,
        selected: Dict[str, List[str]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        question_lower = question.lower()

        numeric = selected.get("numeric", [])
        categorical = selected.get("categorical", [])
        datetime_cols = selected.get("datetime", [])

        if datetime_cols and numeric:
            suggestions.append({
                "chart_type": "line",
                "columns": [datetime_cols[0], numeric[0]],
                "reason": "Shows how the primary metric changes over time"
            })

        if categorical and numeric:
            suggestions.append({
                "chart_type": "bar",
                "columns": [categorical[0], numeric[0]],
                "reason": "Compares the metric across categories"
            })

        if len(numeric) >= 2:
            suggestions.append({
                "chart_type": "scatter",
                "columns": numeric[:2],
                "reason": "Highlights correlation between key numeric fields"
            })

        if any(keyword in question_lower for keyword in ["distribution", "proportion", "share", "composition"]):
            target_col = categorical[0] if categorical else (metadata.get("categorical_columns", [])[:1] or [None])[0]
            if target_col:
                suggestions.append({
                    "chart_type": "pie",
                    "columns": [target_col],
                    "reason": "Communicates proportional breakdown of categorical values"
                })

        return suggestions[:4]

    def _build_zero_shot_prompt(self, question: str, metadata: Dict[str, Any]) -> str:
        columns_summary = ", ".join([
            f"{name} ({dtype})" for name, dtype in metadata.get("column_types", {}).items()
        ])

        numeric_lines = []
        for column, stats in metadata.get("numeric_summary", {}).items():
            line = (
                f"- {column}: min={stats.get('min')}, max={stats.get('max')}, "
                f"mean={stats.get('mean')}, median={stats.get('median')}, observations={stats.get('count')}"
            )
            numeric_lines.append(line)

        categorical_lines = []
        for column, entries in metadata.get("categorical_summary", {}).items():
            values = ", ".join([f"{item['value']} ({item['count']})" for item in entries[:5]])
            categorical_lines.append(f"- {column}: {values}")

        sample_rows = metadata.get("sample_rows", [])
        sample_preview = json.dumps(sample_rows[:3], indent=2) if sample_rows else "[]"

        return (
            "You are a data analyst working with a user-uploaded dataset.\n"
            f"Dataset ID: {metadata.get('dataset_id')} (rows: {metadata.get('total_rows')})\n"
            f"Columns: {columns_summary}\n\n"
            "Numeric profiles:\n" + ("\n".join(numeric_lines) or "- None") + "\n\n"
            "Categorical profiles:\n" + ("\n".join(categorical_lines) or "- None") + "\n\n"
            f"Sample rows:\n{sample_preview}\n\n"
            f"User question: \"{question}\"\n"
            "Respond in English using only this context. Cite the relevant columns in your explanation."
        )

    def _build_cot_prompt(
        self,
        question: str,
        metadata: Dict[str, Any],
        sources: List[Dict[str, Any]],
        charts: List[Dict[str, Any]]
    ) -> str:
        source_summary = []
        for source in sources:
            if source["type"] == "numeric_summary":
                stats = source["stats"]
                source_summary.append(
                    f"Numeric column '{source['column']}' spans {stats.get('min')} to {stats.get('max')} "
                    f"with mean {stats.get('mean')}"
                )
            elif source["type"] == "categorical_distribution":
                values = ", ".join([f"{item['value']} ({item['count']})" for item in source["top_values"][:3]])
                source_summary.append(f"Categorical column '{source['column']}' top values: {values}")
            elif source["type"] == "sample_rows":
                source_summary.append("Sample rows available for qualitative checks")

        chart_summary = [
            f"{item['chart_type']} chart for {', '.join(item['columns'])}: {item['reason']}"
            for item in charts
        ]

        return (
            "Think through the question step by step before composing the final answer.\n"
            f"Question: {question}\n"
            "Key evidence:\n- " + "\n- ".join(source_summary or ["Limited metadata only"]) + "\n"
            "Proposed visualisations:\n- " + "\n- ".join(chart_summary or ["No chart suggestion available"]) + "\n\n"
            "Use the evidence to justify the answer. Provide only the final response and chart recommendation in natural language."
        )

    def _baseline_answer(
        self,
        question: str,
        sources: List[Dict[str, Any]],
        charts: List[Dict[str, Any]]
    ) -> str:
        numeric_parts = []
        categorical_parts = []

        for source in sources:
            if source["type"] == "numeric_summary":
                stats = source["stats"]
                numeric_parts.append(
                    f"Column {source['column']} ranges from {stats.get('min')} to {stats.get('max')} "
                    f"with an average of {stats.get('mean')}"
                )
            elif source["type"] == "categorical_distribution":
                top = source["top_values"][:3]
                formatted = ", ".join([f"{item['value']} ({item['count']})" for item in top])
                categorical_parts.append(f"Column {source['column']} typical values: {formatted}")

        summary_lines = [
            "Baseline analysis (LLM unavailable):",
            f"Question: {question}",
        ]

        if numeric_parts:
            summary_lines.append("Numeric insights: " + "; ".join(numeric_parts))
        if categorical_parts:
            summary_lines.append("Categorical insights: " + "; ".join(categorical_parts))
        if charts:
            chart_text = "; ".join([f"{item['chart_type']} chart using {', '.join(item['columns'])}" for item in charts])
            summary_lines.append("Suggested charts: " + chart_text)

        return "\n".join(summary_lines)

