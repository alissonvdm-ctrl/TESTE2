import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Container for the results of a full validation pass."""

    # Cleaned DataFrame with only valid rows (sequence columns cast to int)
    clean_df: pd.DataFrame

    # DataFrame of rows that failed validation (original row index preserved)
    invalid_df: pd.DataFrame

    # Summary statistics
    stats: Dict[str, Any] = field(default_factory=dict)

    # Per-row error messages keyed by original DataFrame index
    errors: Dict[int, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.invalid_df) == 0

    @property
    def total_rows(self) -> int:
        return len(self.clean_df) + len(self.invalid_df)

    @property
    def valid_count(self) -> int:
        return len(self.clean_df)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_df)


class DataValidator:
    """Validates that each row has exactly k valid numbers in [1, N]."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        n_max: int,
        k_count: int,
        sequence_columns: List[str],
        allow_repeats: bool = False,
        temporal_column: Optional[str] = None,
    ) -> ValidationResult:
        """
        Full validation returning clean df, invalid rows, and stats.

        Parameters
        ----------
        df               : Input DataFrame.
        n_max            : Maximum allowed value (numbers must be in [1, n_max]).
        k_count          : Required number of valid numbers per row.
        sequence_columns : Column names that contain the sequence numbers.
        allow_repeats    : Whether duplicate numbers within a single row are OK.
        temporal_column  : If provided, the cleaned df is sorted by this column.

        Returns
        -------
        ValidationResult
        """
        if not sequence_columns:
            raise ValueError("sequence_columns must not be empty.")
        if n_max < 1:
            raise ValueError("n_max must be >= 1.")
        if k_count < 1:
            raise ValueError("k_count must be >= 1.")
        missing = [c for c in sequence_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        valid_indices: List[int] = []
        invalid_indices: List[int] = []
        errors: Dict[int, str] = {}

        for idx, row in df.iterrows():
            numbers = self._extract_numbers(row, sequence_columns)
            ok, msg = self.validate_row(numbers, n_max, k_count, allow_repeats)
            if ok:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)
                errors[idx] = msg

        clean_df = df.loc[valid_indices].copy()
        invalid_df = df.loc[invalid_indices].copy()

        # Cast sequence columns to int in clean_df
        for col in sequence_columns:
            clean_df[col] = clean_df[col].astype(int)

        # Sort by temporal column if provided
        if temporal_column and temporal_column in clean_df.columns:
            try:
                clean_df = clean_df.sort_values(temporal_column).reset_index(drop=True)
            except Exception:
                pass  # non-fatal; keep original order

        # Find duplicate rows within the valid set
        duplicate_df = self.find_duplicates(clean_df, sequence_columns)

        stats = self._compute_stats(
            df, clean_df, invalid_df, duplicate_df, n_max, k_count, sequence_columns
        )

        return ValidationResult(
            clean_df=clean_df,
            invalid_df=invalid_df,
            stats=stats,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Row-level validation
    # ------------------------------------------------------------------

    def validate_row(
        self,
        numbers: List[int],
        n_max: int,
        k_count: int,
        allow_repeats: bool,
    ) -> Tuple[bool, str]:
        """
        Validate a single row of numbers.

        Returns (True, "") if valid, or (False, reason_string) otherwise.
        """
        # Check for missing / non-integer values
        if len(numbers) == 0:
            return False, "Row contains no numeric values."

        # Check count
        if len(numbers) != k_count:
            return (
                False,
                f"Expected {k_count} numbers, found {len(numbers)}.",
            )

        # Range check
        out_of_range = [n for n in numbers if not (1 <= n <= n_max)]
        if out_of_range:
            return (
                False,
                f"Values out of range [1, {n_max}]: {out_of_range}.",
            )

        # Repeat check
        if not allow_repeats and len(numbers) != len(set(numbers)):
            seen = set()
            dupes = sorted({n for n in numbers if n in seen or seen.add(n)})
            return False, f"Duplicate numbers within row: {dupes}."

        return True, ""

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def find_duplicates(
        self, df: pd.DataFrame, sequence_columns: List[str]
    ) -> pd.DataFrame:
        """
        Find rows whose sorted sequence of numbers is identical to another row.

        The comparison is order-insensitive: [1,2,3] == [3,1,2].

        Returns a DataFrame of the duplicate rows (all occurrences except the
        first), with an additional column ``_duplicate_of`` indicating the
        original row index.
        """
        if df.empty:
            return df.iloc[0:0].copy()

        cols = [c for c in sequence_columns if c in df.columns]
        if not cols:
            return df.iloc[0:0].copy()

        # Build a sorted-tuple key for each row
        def row_key(row: pd.Series) -> Tuple:
            vals = []
            for c in cols:
                try:
                    vals.append(int(row[c]))
                except (ValueError, TypeError):
                    vals.append(None)
            return tuple(sorted(vals))

        keys = df.apply(row_key, axis=1)
        duplicated_mask = keys.duplicated(keep="first")

        dup_df = df[duplicated_mask].copy()

        # Annotate each duplicate with the index of its first occurrence
        first_occurrence = {}
        for idx, key in keys.items():
            if key not in first_occurrence:
                first_occurrence[key] = idx

        dup_df["_duplicate_of"] = dup_df.index.map(
            lambda i: first_occurrence.get(keys[i])
        )
        return dup_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numbers(row: pd.Series, sequence_columns: List[str]) -> List[int]:
        """
        Extract integer values from the specified columns of a row.

        Non-numeric or NaN values are silently skipped (they will cause the
        count check to fail downstream).
        """
        numbers: List[int] = []
        for col in sequence_columns:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            try:
                numbers.append(int(float(val)))
            except (ValueError, TypeError):
                pass
        return numbers

    @staticmethod
    def _compute_stats(
        original_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        invalid_df: pd.DataFrame,
        duplicate_df: pd.DataFrame,
        n_max: int,
        k_count: int,
        sequence_columns: List[str],
    ) -> Dict[str, Any]:
        """Compute summary statistics for the validation pass."""
        total = len(original_df)
        valid = len(clean_df)
        invalid = len(invalid_df)
        duplicates = len(duplicate_df)

        stats: Dict[str, Any] = {
            "total_rows": total,
            "valid_rows": valid,
            "invalid_rows": invalid,
            "duplicate_rows": duplicates,
            "validity_rate": round(valid / total, 4) if total else 0.0,
            "n_max": n_max,
            "k_count": k_count,
            "sequence_columns": sequence_columns,
        }

        # Frequency distribution of each number across the clean set
        if valid > 0:
            all_numbers: List[int] = []
            for col in sequence_columns:
                if col in clean_df.columns:
                    all_numbers.extend(clean_df[col].dropna().astype(int).tolist())

            freq = np.zeros(n_max + 1, dtype=int)
            for n in all_numbers:
                if 1 <= n <= n_max:
                    freq[n] += 1

            freq_dict = {str(i): int(freq[i]) for i in range(1, n_max + 1)}
            stats["number_frequency"] = freq_dict
            stats["most_frequent"] = int(np.argmax(freq[1:]) + 1)
            stats["least_frequent"] = int(
                np.argmin(np.where(freq[1:] == 0, np.iinfo(int).max, freq[1:])) + 1
            )
        else:
            stats["number_frequency"] = {}
            stats["most_frequent"] = None
            stats["least_frequent"] = None

        return stats
