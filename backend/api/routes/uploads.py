"""
Uploads router.

Handles Excel file ingestion, column detection, data preview, and
N/k parameter validation before an experiment is created.
"""

from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_async_session as get_db

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed MIME types / extensions for uploaded data files
_ALLOWED_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
    "application/octet-stream",  # generic binary — browsers sometimes send this
}
_ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
_PREVIEW_ROWS = 10


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ColumnInfo(BaseModel):
    """Metadata about a single detected column."""

    name: str = Field(..., description="Column header as read from the file")
    dtype: str = Field(..., description="Inferred pandas dtype (e.g. 'float64', 'int64', 'object')")
    is_numeric: bool = Field(..., description="True if the column contains numeric data")
    null_count: int = Field(..., ge=0, description="Number of null/NaN values in the column")
    sample_values: list[Any] = Field(default_factory=list, description="Up to 5 representative non-null values")


class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    upload_id: str = Field(..., description="Unique identifier for this upload session")
    filename: str = Field(..., description="Original filename as provided by the client")
    row_count: int = Field(..., ge=0, description="Total number of data rows (excluding header)")
    column_count: int = Field(..., ge=0, description="Total number of columns")
    columns: list[ColumnInfo] = Field(..., description="Per-column metadata")
    preview: list[dict[str, Any]] = Field(..., description=f"First {_PREVIEW_ROWS} rows as list-of-dicts")
    uploaded_at: datetime = Field(..., description="Server-side timestamp of the upload")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")


class ValidationRequest(BaseModel):
    """Parameters used to validate suitability of the uploaded dataset."""

    sequence_column: str = Field(..., description="Column containing the numeric sequence to analyse")
    n_value: int = Field(..., ge=1, description="Analysis window length N")
    k_value: int = Field(..., ge=1, description="Lag/step parameter k")


class ValidationIssue(BaseModel):
    """A single validation finding (warning or error)."""

    level: str = Field(..., description="'error' or 'warning'")
    code: str = Field(..., description="Machine-readable issue code")
    message: str = Field(..., description="Human-readable description")


class ValidationResponse(BaseModel):
    """Result of N/k parameter validation against the uploaded dataset."""

    upload_id: str
    valid: bool = Field(..., description="True only when there are no error-level issues")
    row_count: int
    usable_rows: int = Field(..., description="Rows available after accounting for N and k")
    issues: list[ValidationIssue] = Field(default_factory=list)


class ColumnsResponse(BaseModel):
    """List of detected columns for a given upload."""

    upload_id: str
    columns: list[ColumnInfo]


class PreviewResponse(BaseModel):
    """First N rows of the uploaded dataset."""

    upload_id: str
    row_count: int
    preview_rows: int
    columns: list[str]
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upload_path(upload_id: str) -> Path:
    """Return the filesystem path where an uploaded file is stored."""
    return Path(settings.upload_dir) / upload_id


def _load_dataframe(upload_id: str) -> pd.DataFrame:
    """
    Load a previously uploaded Excel file from disk into a DataFrame.

    Raises HTTP 404 if the file cannot be found, and HTTP 422 if it cannot
    be parsed.
    """
    directory = _upload_path(upload_id)
    if not directory.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload '{upload_id}' not found. The session may have expired.",
        )

    # Find the Excel file inside the upload directory
    excel_files = list(directory.glob("*.xls*"))
    if not excel_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No Excel file found for upload '{upload_id}'.",
        )

    try:
        df = pd.read_excel(excel_files[0], engine="openpyxl" if excel_files[0].suffix == ".xlsx" else "xlrd")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse Excel file for upload '%s'", upload_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse the uploaded file: {exc}",
        ) from exc

    return df


def _build_column_infos(df: pd.DataFrame) -> list[ColumnInfo]:
    """Extract per-column metadata from a DataFrame."""
    infos: list[ColumnInfo] = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        non_null = series.dropna()
        sample = non_null.head(5).tolist()
        infos.append(
            ColumnInfo(
                name=str(col),
                dtype=str(series.dtype),
                is_numeric=is_numeric,
                null_count=int(series.isna().sum()),
                sample_values=sample,
            )
        )
    return infos


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/excel",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an Excel file and get a data preview",
)
async def upload_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx or .xls)"),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """
    Accept an Excel file upload, persist it to the upload store, and return:

    * a unique ``upload_id`` for subsequent API calls,
    * detected column metadata (names, dtypes, null counts, sample values),
    * a preview of the first 10 data rows.

    The file is **not** processed until ``POST /uploads/{upload_id}/validate``
    is called with the chosen ``sequence_column``, ``N``, and ``k``.
    """
    # --- Extension / MIME validation ---
    original_filename = file.filename or "upload.xlsx"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Only .xlsx and .xls files are accepted.",
        )

    # --- Size guard ---
    raw_bytes = await file.read()
    file_size = len(raw_bytes)
    if file_size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size / 1_048_576:.1f} MB exceeds the "
                f"{settings.max_upload_size_mb} MB limit."
            ),
        )

    # --- Parse ---
    try:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse uploaded Excel file '%s'", original_filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse Excel file: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file contains no data rows.",
        )

    # --- Persist to disk ---
    upload_id = str(uuid.uuid4())
    upload_dir = _upload_path(upload_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / original_filename
    dest_path.write_bytes(raw_bytes)
    logger.info(
        "Saved upload '%s' (%d bytes, %d rows) -> %s",
        upload_id,
        file_size,
        len(df),
        dest_path,
    )

    # TODO: persist upload metadata to DB

    columns = _build_column_infos(df)
    preview_df = df.head(_PREVIEW_ROWS)
    preview = preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")

    return UploadResponse(
        upload_id=upload_id,
        filename=original_filename,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        preview=preview,
        uploaded_at=datetime.utcnow(),
        file_size_bytes=file_size,
    )


@router.post(
    "/{upload_id}/validate",
    response_model=ValidationResponse,
    summary="Validate dataset with N and k parameters",
)
async def validate_upload(
    upload_id: str,
    payload: ValidationRequest,
    db: AsyncSession = Depends(get_db),
) -> ValidationResponse:
    """
    Validate whether the uploaded dataset is suitable for analysis given:

    * the chosen ``sequence_column``,
    * the analysis window ``n_value``,
    * the lag parameter ``k_value``.

    Returns a list of issues (errors and warnings).  The caller should
    proceed only when ``valid`` is ``True``.
    """
    df = _load_dataframe(upload_id)
    issues: list[ValidationIssue] = []

    # --- Column existence ---
    if payload.sequence_column not in df.columns:
        issues.append(
            ValidationIssue(
                level="error",
                code="COLUMN_NOT_FOUND",
                message=(
                    f"Column '{payload.sequence_column}' does not exist in the dataset. "
                    f"Available columns: {', '.join(str(c) for c in df.columns)}."
                ),
            )
        )
        return ValidationResponse(
            upload_id=upload_id,
            valid=False,
            row_count=len(df),
            usable_rows=0,
            issues=issues,
        )

    series = df[payload.sequence_column]

    # --- Numeric dtype ---
    if not pd.api.types.is_numeric_dtype(series):
        issues.append(
            ValidationIssue(
                level="error",
                code="COLUMN_NOT_NUMERIC",
                message=(
                    f"Column '{payload.sequence_column}' has dtype '{series.dtype}', "
                    "which is not numeric. Select a numeric column for analysis."
                ),
            )
        )

    # --- Missing values ---
    null_count = int(series.isna().sum())
    if null_count > 0:
        null_pct = 100.0 * null_count / len(series)
        level = "error" if null_pct > 50 else "warning"
        issues.append(
            ValidationIssue(
                level=level,
                code="MISSING_VALUES",
                message=(
                    f"Column '{payload.sequence_column}' contains {null_count} missing "
                    f"values ({null_pct:.1f}% of rows). "
                    + ("Too many missing values for reliable analysis." if level == "error" else
                       "Missing values will be imputed before analysis.")
                ),
            )
        )

    # --- Usable rows after N and k ---
    row_count = len(df)
    # Minimum rows needed: N + k (embedding requires at least N observations, k steps for lag)
    min_rows_needed = payload.n_value + payload.k_value
    usable_rows = max(0, row_count - min_rows_needed)

    if row_count < min_rows_needed:
        issues.append(
            ValidationIssue(
                level="error",
                code="INSUFFICIENT_ROWS",
                message=(
                    f"Dataset has {row_count} rows but N+k = "
                    f"{payload.n_value}+{payload.k_value} = {min_rows_needed} rows are required. "
                    "Reduce N or k, or supply a longer sequence."
                ),
            )
        )
    elif usable_rows < 20:
        issues.append(
            ValidationIssue(
                level="warning",
                code="FEW_USABLE_ROWS",
                message=(
                    f"Only {usable_rows} usable rows remain after accounting for N and k. "
                    "Model accuracy may be limited; consider reducing N or k."
                ),
            )
        )

    # --- k must be less than N ---
    if payload.k_value >= payload.n_value:
        issues.append(
            ValidationIssue(
                level="warning",
                code="K_GE_N",
                message=(
                    f"k ({payload.k_value}) is greater than or equal to N ({payload.n_value}). "
                    "Typically k should be much smaller than N for meaningful lag embedding."
                ),
            )
        )

    has_errors = any(i.level == "error" for i in issues)

    return ValidationResponse(
        upload_id=upload_id,
        valid=not has_errors,
        row_count=row_count,
        usable_rows=usable_rows,
        issues=issues,
    )


@router.get(
    "/{upload_id}/columns",
    response_model=ColumnsResponse,
    summary="Get detected columns for an upload",
)
async def get_columns(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
) -> ColumnsResponse:
    """
    Return the column metadata detected from a previously uploaded file.

    Useful for populating column-selector UI components without re-uploading.
    """
    df = _load_dataframe(upload_id)
    columns = _build_column_infos(df)
    return ColumnsResponse(upload_id=upload_id, columns=columns)


@router.get(
    "/{upload_id}/preview",
    response_model=PreviewResponse,
    summary="Get data preview (first 10 rows)",
)
async def get_preview(
    upload_id: str,
    rows: int = Query(default=_PREVIEW_ROWS, ge=1, le=100, description="Number of rows to return"),
    db: AsyncSession = Depends(get_db),
) -> PreviewResponse:
    """
    Return the first N rows (default 10) of the uploaded dataset as a
    list-of-dicts, suitable for rendering a preview table in the UI.
    """
    df = _load_dataframe(upload_id)
    preview_df = df.head(rows)
    data = preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")

    return PreviewResponse(
        upload_id=upload_id,
        row_count=len(df),
        preview_rows=len(data),
        columns=[str(c) for c in df.columns],
        data=data,
    )
