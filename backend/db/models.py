"""
SQLAlchemy ORM models for the NumericSequenceAnalyzer application.

Models
------
- Experiment       – top-level container for an analysis run
- Dataset          – uploaded file associated with an experiment
- Sample           – individual numeric sequence row from a dataset
- AnalysisResult   – output of a single analytics module
- ModelResult      – trained ML model metrics and predictions
- Prediction       – on-demand inference result
- Report           – generated PDF/HTML report artefact
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------


class ExperimentStatus(str, enum.Enum):
    """Lifecycle states for an Experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisStatus(str, enum.Enum):
    """Execution states for an AnalysisResult."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReportType(str, enum.Enum):
    """Supported report output formats."""

    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    EXCEL = "excel"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class Experiment(Base):
    """
    Top-level container for a numeric sequence analysis run.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    name:
        Human-readable label, unique per deployment.
    description:
        Optional free-text description.
    n_max:
        Maximum value that can appear in a sequence number (domain size).
    k_count:
        Number of numbers drawn per sequence (draw size).
    order_matters:
        Whether the order of numbers within a sequence is significant.
    status:
        Current lifecycle state (see ``ExperimentStatus``).
    created_at / updated_at:
        Timestamps maintained automatically by the database.
    config_json:
        Arbitrary JSON blob for storing module-level configuration overrides.
    """

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    n_max: Mapped[int] = mapped_column(Integer, nullable=False)
    k_count: Mapped[int] = mapped_column(Integer, nullable=False)
    order_matters: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, name="experiment_status"),
        nullable=False,
        default=ExperimentStatus.PENDING,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    config_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)

    # Relationships
    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset", back_populates="experiment", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[list["AnalysisResult"]] = relationship(
        "AnalysisResult", back_populates="experiment", cascade="all, delete-orphan"
    )
    model_results: Mapped[list["ModelResult"]] = relationship(
        "ModelResult", back_populates="experiment", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="experiment", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(
        "Report", back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} name={self.name!r} status={self.status}>"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset(Base):
    """
    An uploaded data file associated with one Experiment.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    experiment_id:
        Foreign key to ``experiments.id``.
    file_path:
        Absolute path on the server where the file is stored.
    original_filename:
        The filename as uploaded by the user.
    row_count:
        Total number of rows parsed from the file.
    valid_rows:
        Number of rows that passed validation.
    invalid_rows:
        Number of rows that failed validation.
    column_mapping_json:
        JSON object describing how file columns map to domain fields.
    created_at:
        Upload timestamp.
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    valid_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_mapping_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="datasets")
    samples: Mapped[list["Sample"]] = relationship(
        "Sample", back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Dataset id={self.id} experiment_id={self.experiment_id}"
            f" filename={self.original_filename!r}>"
        )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


class Sample(Base):
    """
    One individual numeric sequence row sourced from a Dataset.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    dataset_id:
        Foreign key to ``datasets.id``.
    sequence_order:
        Zero-based index indicating the row's position within the dataset.
        Used to reconstruct temporal ordering.
    numbers_json:
        JSON array storing the actual numbers of the draw
        (e.g. ``[5, 14, 22, 33, 41, 48]``).
    timestamp:
        Optional draw date/time parsed from the source file.
    is_valid:
        Whether this row passed all validation checks.
    """

    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_order: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    numbers_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="samples")

    def __repr__(self) -> str:
        return (
            f"<Sample id={self.id} dataset_id={self.dataset_id}"
            f" order={self.sequence_order} valid={self.is_valid}>"
        )


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


class AnalysisResult(Base):
    """
    Output produced by a single analytics module for an Experiment.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    experiment_id:
        Foreign key to ``experiments.id``.
    module_name:
        Dotted Python path or short slug identifying the analytics module.
    result_json:
        Full JSON output of the module (metrics, charts data, tables, etc.).
    created_at:
        Timestamp when the result was persisted.
    duration_seconds:
        Wall-clock time the module took to complete.
    status:
        Execution outcome (see ``AnalysisStatus``).
    """

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    module_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    result_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus, name="analysis_status"),
        nullable=False,
        default=AnalysisStatus.PENDING,
        index=True,
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="analysis_results"
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult id={self.id} experiment_id={self.experiment_id}"
            f" module={self.module_name!r} status={self.status}>"
        )


# ---------------------------------------------------------------------------
# ModelResult
# ---------------------------------------------------------------------------


class ModelResult(Base):
    """
    Artefacts produced by training a single ML model for an Experiment.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    experiment_id:
        Foreign key to ``experiments.id``.
    model_name:
        Human-readable model label (e.g. ``"RandomForest"``, ``"LSTM"``).
    module:
        Dotted Python path to the training module that produced this result.
    metrics_json:
        JSON object with evaluation metrics (accuracy, F1, RMSE, etc.).
    predictions_json:
        JSON array of model predictions on the validation / test set.
    feature_importance_json:
        JSON array of ``{feature, importance}`` objects (if applicable).
    trained_at:
        Timestamp when training completed.
    val_score:
        Primary validation metric score (for quick comparisons).
    test_score:
        Primary test metric score.
    """

    __tablename__ = "model_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    module: Mapped[str] = mapped_column(String(512), nullable=False)
    metrics_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    predictions_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    feature_importance_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    trained_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    val_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    test_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="model_results"
    )

    def __repr__(self) -> str:
        return (
            f"<ModelResult id={self.id} experiment_id={self.experiment_id}"
            f" model={self.model_name!r} val_score={self.val_score}>"
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class Prediction(Base):
    """
    Result of an on-demand inference request against the trained ensemble.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    experiment_id:
        Foreign key to ``experiments.id``.
    numbers_json:
        JSON array containing the predicted numbers.
    confidence:
        Overall confidence score in [0, 1].
    explanation:
        Human-readable explanation of why these numbers were predicted.
    model_ensemble_json:
        JSON object describing individual model contributions to the ensemble.
    created_at:
        Timestamp of inference.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    numbers_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_ensemble_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="predictions"
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} experiment_id={self.experiment_id}"
            f" confidence={self.confidence}>"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Report(Base):
    """
    A generated report artefact (PDF, HTML, JSON, or Excel) for an Experiment.

    Attributes
    ----------
    id:
        Auto-incrementing surrogate primary key.
    experiment_id:
        Foreign key to ``experiments.id``.
    report_type:
        Output format (see ``ReportType``).
    file_path:
        Absolute path on the server where the report file is stored.
    created_at:
        Timestamp when the report was generated and persisted.
    """

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    report_type: Mapped[ReportType] = mapped_column(
        Enum(ReportType, name="report_type"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="reports"
    )

    def __repr__(self) -> str:
        return (
            f"<Report id={self.id} experiment_id={self.experiment_id}"
            f" type={self.report_type} path={self.file_path!r}>"
        )
