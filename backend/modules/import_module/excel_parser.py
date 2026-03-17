import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import openpyxl
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ExcelParser:
    """Parses Excel files and extracts metadata for column mapping."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}

    # Keywords that strongly suggest a temporal/ordering column
    TEMPORAL_KEYWORDS = [
        "date", "data", "time", "tempo", "hora", "dt", "timestamp",
        "periodo", "period", "ano", "year", "mes", "month", "semana",
        "week", "dia", "day", "ordem", "order", "seq", "sequencia",
        "sequence", "draw", "sorteio", "concurso", "contest", "numero",
        "num", "id", "index", "indice", "rodada", "round",
    ]

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Main parse method.

        Returns a dict with:
            sheets          : list of sheet names
            default_sheet   : sheet chosen automatically (first with numeric data)
            columns         : {col_name: column_type} mapping
            temporal_column : guessed temporal/order column or None
            numeric_columns : list of numeric column names
            preview         : first 20 rows as list of dicts
            row_count       : total rows in the default sheet
            column_count    : total columns in the default sheet
            file_info       : basic metadata about the file
        """
        path = Path(file_path)
        self._validate_path(path)

        sheets = self.detect_sheets(file_path)
        if not sheets:
            raise ValueError("No sheets found in the workbook.")

        # Pick the best sheet (first with numeric data)
        default_sheet, df = self._select_best_sheet(file_path, sheets)

        columns = self.detect_columns(df)
        temporal_col = self.detect_temporal_column(df)
        numeric_cols = [c for c, t in columns.items() if t == "numeric"]
        preview = self.get_preview(df)

        return {
            "sheets": sheets,
            "default_sheet": default_sheet,
            "columns": columns,
            "temporal_column": temporal_col,
            "numeric_columns": numeric_cols,
            "preview": preview,
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_info": {
                "path": str(path.resolve()),
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            },
        }

    # ------------------------------------------------------------------
    # Sheet detection
    # ------------------------------------------------------------------

    def detect_sheets(self, file_path: str) -> List[str]:
        """Return list of sheet names present in the workbook."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in {".xlsx", ".xlsm"}:
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            names = list(wb.sheetnames)
            wb.close()
            return names
        elif ext == ".xls":
            import xlrd  # type: ignore
            wb = xlrd.open_workbook(file_path)
            return wb.sheet_names()
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

    # ------------------------------------------------------------------
    # Column type detection
    # ------------------------------------------------------------------

    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Classify every column in *df* as one of:
            'numeric'  – predominantly integer or float values
            'date'     – datetime or parseable date strings
            'text'     – everything else
        """
        result: Dict[str, str] = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                result[col] = "text"
                continue

            dtype = df[col].dtype

            # Already a datetime dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                result[col] = "date"
                continue

            # Numeric dtypes
            if pd.api.types.is_numeric_dtype(dtype):
                # Heuristic: 8-digit integers that look like YYYYMMDD
                if self._looks_like_date_int(series):
                    result[col] = "date"
                else:
                    result[col] = "numeric"
                continue

            # Object / string columns: probe a sample
            if dtype == object:
                result[col] = self._sniff_object_column(series)
                continue

            result[col] = "text"

        return result

    def _sniff_object_column(self, series: pd.Series) -> str:
        """Determine whether an object column is date, numeric, or text."""
        sample = series.head(100)

        # Attempt numeric conversion first
        numeric_mask = pd.to_numeric(sample, errors="coerce").notna()
        if numeric_mask.mean() >= 0.8:
            return "numeric"

        # Attempt datetime conversion
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() >= 0.7:
                return "date"
        except Exception:
            pass

        # Manual format checks
        date_hits = sum(self._is_date_string(str(v).strip()) for v in sample)
        if date_hits / len(sample) >= 0.6:
            return "date"

        return "text"

    @staticmethod
    def _is_date_string(s: str) -> bool:
        """Try a set of common date formats."""
        DATE_FORMATS = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%d-%m-%Y", "%Y%m%d", "%d.%m.%Y",
            "%Y/%m/%d", "%d/%m/%y", "%m/%d/%y",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in DATE_FORMATS:
            try:
                datetime.strptime(s, fmt)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _looks_like_date_int(series: pd.Series) -> bool:
        """Return True if a numeric series looks like YYYYMMDD integers."""
        try:
            ints = series.astype(np.int64)
            return bool(ints.between(19000101, 20991231).mean() > 0.8)
        except (ValueError, TypeError, OverflowError):
            return False

    # ------------------------------------------------------------------
    # Temporal column detection
    # ------------------------------------------------------------------

    def detect_temporal_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Heuristically find the most likely date/order column.

        Priority:
            1. Datetime dtype columns.
            2. Column names containing a temporal keyword (most matches wins).
            3. Columns classified as 'date'.
            4. Monotonically increasing integer column (likely a draw counter).
        """
        # 1. Datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        col_lower = {col: str(col).lower() for col in df.columns}

        # 2. Name-based keyword score
        scored: List[Tuple[int, str]] = []
        for col, name in col_lower.items():
            score = sum(1 for kw in self.TEMPORAL_KEYWORDS if kw in name)
            if score > 0:
                scored.append((score, col))
        if scored:
            scored.sort(key=lambda x: -x[0])
            return scored[0][1]

        # 3. Date-classified columns
        classifications = self.detect_columns(df)
        date_cols = [c for c, t in classifications.items() if t == "date"]
        if date_cols:
            return date_cols[0]

        # 4. Monotonically increasing integer column
        numeric_cols = [c for c, t in classifications.items() if t == "numeric"]
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 2:
                continue
            try:
                arr = pd.to_numeric(s, errors="coerce").dropna().values
                if len(arr) > 1 and np.all(np.diff(arr) >= 0):
                    return col
            except (ValueError, TypeError):
                pass

        return None

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def get_preview(self, df: pd.DataFrame, rows: int = 20) -> List[Dict]:
        """
        Return the first *rows* rows as a list of plain Python dicts.

        All numpy / pandas non-serialisable types are converted to
        native Python equivalents or None.
        """
        subset = df.head(rows)
        records: List[Dict[str, Any]] = []
        for _, row in subset.iterrows():
            record: Dict[str, Any] = {}
            for col, val in row.items():
                record[col] = self._safe_value(val)
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

    def _load_sheet(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        """Load a single sheet into a DataFrame."""
        ext = Path(file_path).suffix.lower()
        engine = "xlrd" if ext == ".xls" else "openpyxl"

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            engine=engine,
            header=0,
        )

        # Drop completely empty rows / columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _select_best_sheet(
        self, file_path: str, sheets: List[str]
    ) -> Tuple[str, pd.DataFrame]:
        """
        Pick the sheet with the most numeric columns.
        Falls back to the first sheet if none contain numerics.
        """
        best_sheet = sheets[0]
        best_df = self._load_sheet(file_path, sheets[0])
        best_score = self._numeric_column_count(best_df)

        for sheet in sheets[1:]:
            try:
                df = self._load_sheet(file_path, sheet)
                score = self._numeric_column_count(df)
                if score > best_score:
                    best_score = score
                    best_sheet = sheet
                    best_df = df
            except Exception:
                continue

        return best_sheet, best_df

    def _numeric_column_count(self, df: pd.DataFrame) -> int:
        """Count columns that are predominantly numeric."""
        count = 0
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                count += 1
            elif df[col].dtype == object:
                converted = pd.to_numeric(series.head(30), errors="coerce")
                if converted.notna().mean() >= 0.8:
                    count += 1
        return count

    @staticmethod
    def _safe_value(val: Any) -> Any:
        """Convert a cell value to a JSON-safe Python type."""
        if val is None:
            return None
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            if np.isnan(val):
                return None
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, float) and np.isnan(val):
            return None
        # Generic pandas NA check (avoids calling on arrays/dicts)
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val
