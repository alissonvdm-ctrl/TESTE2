"""
Reports router.

Handles on-demand generation, listing, downloading, and CSV export of
analysis reports (PDF / Excel) and derived data artefacts (feature importance
and predictions CSVs) for a given experiment.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_async_session as get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ReportFormat(str, Enum):
    """Supported output formats for generated reports."""

    PDF = "pdf"
    EXCEL = "excel"


class ReportStatus(str, Enum):
    """Lifecycle status of a report generation job."""

    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ReportGenerateRequest(BaseModel):
    """Options for generating a new report."""

    format: ReportFormat = Field(
        default=ReportFormat.PDF,
        description="Output format: 'pdf' or 'excel'",
    )
    include_sections: Optional[list[str]] = Field(
        default=None,
        description=(
            "Subset of sections to include. Omit for all sections. "
            "Supported: 'summary', 'diagnostic', 'changepoints', 'models', "
            "'ensemble', 'features', 'predictions'."
        ),
    )
    title: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Custom title for the report cover page",
    )
    author: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Author name to embed in the report metadata",
    )


class ReportGenerateResponse(BaseModel):
    """Immediate response after enqueueing report generation."""

    report_id: str
    experiment_id: str
    format: ReportFormat
    status: ReportStatus
    message: str
    queued_at: datetime


class ReportItem(BaseModel):
    """Metadata for a single generated report."""

    report_id: str
    experiment_id: str
    format: ReportFormat
    status: ReportStatus
    title: Optional[str] = None
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = Field(
        default=None,
        description="Relative URL for downloading the report",
    )
    created_at: datetime
    completed_at: Optional[datetime] = None


class ReportListResponse(BaseModel):
    """List of reports available for an experiment."""

    experiment_id: str
    total: int
    reports: list[ReportItem]


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


async def _require_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> str:
    """
    Verify the experiment exists, returning its ID.

    Raises HTTP 404 if not found.
    """
    # TODO: replace with actual ORM lookup once models are defined
    # from backend.db.models import Experiment
    # result = await db.get(Experiment, experiment_id)
    # if result is None:
    #     raise HTTPException(status_code=404, detail="Experiment not found")
    # return experiment_id

    # Stub: always raise until ORM is wired up
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Experiment '{experiment_id}' not found.",
    )


def _report_dir(experiment_id: str) -> Path:
    """Return the directory where reports for an experiment are stored."""
    return Path(settings.upload_dir) / "reports" / experiment_id


def _report_file_path(experiment_id: str, report_id: str, fmt: ReportFormat) -> Path:
    """Build the full file path for a report artefact."""
    ext = "pdf" if fmt == ReportFormat.PDF else "xlsx"
    return _report_dir(experiment_id) / f"{report_id}.{ext}"


async def _get_report_or_404(
    experiment_id: str,
    report_id: str,
    db: AsyncSession,
) -> dict[str, Any]:
    """
    Fetch report metadata from the database, raising HTTP 404 if missing.
    """
    # TODO: implement DB lookup
    # from backend.db.models import Report
    # result = await db.get(Report, report_id)
    # if result is None or result.experiment_id != experiment_id:
    #     raise HTTPException(status_code=404, detail="Report not found")
    # return result

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Report '{report_id}' not found for experiment '{experiment_id}'.",
    )


# ---------------------------------------------------------------------------
# Background task: report generation
# ---------------------------------------------------------------------------


async def _generate_report_task(
    experiment_id: str,
    report_id: str,
    fmt: ReportFormat,
    include_sections: Optional[list[str]],
    title: Optional[str],
    author: Optional[str],
) -> None:
    """
    Background coroutine that runs the report generation pipeline.

    Responsibilities:
    1. Mark the report record as GENERATING in the DB.
    2. Gather results from the analysis result store.
    3. Render the report (PDF via WeasyPrint / ReportLab, or Excel via openpyxl).
    4. Write the file to ``_report_file_path()``.
    5. Update the DB record to READY (or FAILED on error).
    """
    report_path = _report_file_path(experiment_id, report_id, fmt)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting report generation: experiment=%s, report=%s, format=%s",
        experiment_id,
        report_id,
        fmt,
    )

    try:
        # TODO: implement actual rendering logic, e.g.:
        # from backend.modules.reporting import PDFReportRenderer, ExcelReportRenderer
        # results = await load_experiment_results(experiment_id)
        # renderer = PDFReportRenderer(...) if fmt == ReportFormat.PDF else ExcelReportRenderer(...)
        # renderer.render(results, output_path=report_path, title=title, author=author)

        # Stub: write a placeholder file
        report_path.write_text(
            f"[Placeholder {fmt.value.upper()} report for experiment {experiment_id}]\n"
            f"Report ID : {report_id}\n"
            f"Generated : {datetime.utcnow().isoformat()}\n"
        )
        logger.info("Report '%s' written to %s", report_id, report_path)

        # TODO: update DB record status to READY

    except Exception:  # noqa: BLE001
        logger.exception(
            "Report generation failed: experiment=%s, report=%s", experiment_id, report_id
        )
        # TODO: update DB record status to FAILED


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/{experiment_id}/generate",
    response_model=ReportGenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate a PDF or Excel report",
)
async def generate_report(
    experiment_id: str,
    payload: ReportGenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReportGenerateResponse:
    """
    Enqueue report generation for the given experiment.

    The report is built asynchronously.  Poll ``GET
    /reports/{experiment_id}/list`` to check when the status reaches
    ``ready``, then call ``GET /reports/{experiment_id}/download/{report_id}``
    to retrieve the file.
    """
    await _require_experiment(experiment_id, db)

    report_id = str(uuid.uuid4())
    queued_at = datetime.utcnow()

    logger.info(
        "Enqueueing %s report generation for experiment '%s' (report_id=%s)",
        payload.format.value,
        experiment_id,
        report_id,
    )

    # TODO: persist initial PENDING report record to DB

    background_tasks.add_task(
        _generate_report_task,
        experiment_id=experiment_id,
        report_id=report_id,
        fmt=payload.format,
        include_sections=payload.include_sections,
        title=payload.title,
        author=payload.author,
    )

    return ReportGenerateResponse(
        report_id=report_id,
        experiment_id=experiment_id,
        format=payload.format,
        status=ReportStatus.PENDING,
        message=f"{payload.format.value.upper()} report generation has been queued.",
        queued_at=queued_at,
    )


@router.get(
    "/{experiment_id}/list",
    response_model=ReportListResponse,
    summary="List available reports for an experiment",
)
async def list_reports(
    experiment_id: str,
    fmt: Optional[ReportFormat] = Query(default=None, alias="format", description="Filter by format"),
    report_status: Optional[ReportStatus] = Query(
        default=None, alias="status", description="Filter by status"
    ),
    db: AsyncSession = Depends(get_db),
) -> ReportListResponse:
    """
    Return all reports generated (or pending) for the experiment, optionally
    filtered by format and/or status.
    """
    await _require_experiment(experiment_id, db)
    logger.info(
        "Listing reports for experiment '%s' (format=%s, status=%s)",
        experiment_id,
        fmt,
        report_status,
    )

    # TODO: query DB for report records with optional filters
    return ReportListResponse(
        experiment_id=experiment_id,
        total=0,
        reports=[],
    )


@router.get(
    "/{experiment_id}/download/{report_id}",
    summary="Download a generated report",
    responses={
        200: {
            "description": "Report file (PDF or Excel)",
            "content": {
                "application/pdf": {},
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {},
            },
        }
    },
)
async def download_report(
    experiment_id: str,
    report_id: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """
    Download a previously generated report file.

    Returns the raw PDF or Excel binary with appropriate ``Content-Type`` and
    ``Content-Disposition`` headers so browsers trigger a file download.
    """
    await _require_experiment(experiment_id, db)
    report = await _get_report_or_404(experiment_id, report_id, db)

    # TODO: derive format from the DB record; placeholder uses PDF
    fmt = ReportFormat(report.get("format", ReportFormat.PDF))  # type: ignore[union-attr]

    if report.get("status") != ReportStatus.READY:  # type: ignore[union-attr]
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Report is not ready yet (status: {report.get('status')}).",  # type: ignore[union-attr]
        )

    file_path = _report_file_path(experiment_id, report_id, fmt)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found on server. It may have been deleted.",
        )

    media_type = (
        "application/pdf"
        if fmt == ReportFormat.PDF
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    filename = f"report_{experiment_id[:8]}_{report_id[:8]}.{fmt.value if fmt == ReportFormat.PDF else 'xlsx'}"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )


@router.get(
    "/{experiment_id}/export/features",
    summary="Export feature importance as CSV",
    responses={
        200: {
            "description": "CSV file containing feature importance scores",
            "content": {"text/csv": {}},
        }
    },
)
async def export_features_csv(
    experiment_id: str,
    model_name: str = Query(
        default="ensemble",
        description="Model whose feature importances to export",
    ),
    top_n: int = Query(default=0, ge=0, description="Limit to top N features (0 = all)"),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Stream the feature importance table for the experiment as a CSV download.

    Columns: ``rank``, ``feature_name``, ``category``, ``importance_score``,
    ``description``.
    """
    await _require_experiment(experiment_id, db)
    logger.info(
        "Exporting features CSV for experiment '%s' (model=%s, top_n=%d)",
        experiment_id,
        model_name,
        top_n,
    )

    # TODO: load feature importance records from DB / result store
    # features = await load_feature_importance(experiment_id, model_name, top_n or None)

    # Build CSV content
    rows: list[str] = ["rank,feature_name,category,importance_score,description"]
    # TODO: append real rows; placeholder is empty

    csv_body = "\n".join(rows) + "\n"

    filename = f"features_{experiment_id[:8]}_{model_name}.csv"
    return StreamingResponse(
        content=iter([csv_body]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{experiment_id}/export/predictions",
    summary="Export predictions as CSV",
    responses={
        200: {
            "description": "CSV file containing model predictions",
            "content": {"text/csv": {}},
        }
    },
)
async def export_predictions_csv(
    experiment_id: str,
    model_name: str = Query(
        default="ensemble",
        description="Model whose predictions to export",
    ),
    horizon: int = Query(default=10, ge=1, le=500, description="Forecast horizon"),
    include_confidence_intervals: bool = Query(
        default=True, description="Include lower/upper confidence-interval columns"
    ),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Stream the model predictions for the experiment as a CSV download.

    Columns: ``index``, ``actual``, ``predicted``
    (plus ``lower_ci``, ``upper_ci`` when ``include_confidence_intervals=true``).
    """
    await _require_experiment(experiment_id, db)
    logger.info(
        "Exporting predictions CSV for experiment '%s' (model=%s, horizon=%d)",
        experiment_id,
        model_name,
        horizon,
    )

    # TODO: load predictions from DB / result store
    # predictions = await load_predictions(experiment_id, model_name, horizon)

    # Build header
    headers = ["index", "actual", "predicted"]
    if include_confidence_intervals:
        headers += ["lower_ci", "upper_ci"]

    rows: list[str] = [",".join(headers)]
    # TODO: append real rows; placeholder is empty

    csv_body = "\n".join(rows) + "\n"

    filename = f"predictions_{experiment_id[:8]}_{model_name}.csv"
    return StreamingResponse(
        content=iter([csv_body]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
