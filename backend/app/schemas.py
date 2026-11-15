"""Pydantic schemas for the API."""
from typing import Optional, List
from pydantic import BaseModel, Field


class BodyMeasurements(BaseModel):
    height_cm: Optional[float] = Field(None, gt=80, le=250)
    chest_cm: Optional[float] = Field(None, gt=50, le=200)
    waist_cm: Optional[float] = Field(None, gt=40, le=200)
    shoulder_cm: Optional[float] = Field(None, gt=20, le=80)
    hip_cm: Optional[float] = Field(None, gt=50, le=200)
    preferred_fit: Optional[str] = Field(None, pattern="^(slim|regular|oversized)$")


class GarmentMeasurements(BaseModel):
    chest_width_cm: Optional[float] = Field(None, gt=15, le=80)   # flat half-width
    shoulder_cm: Optional[float] = Field(None, gt=20, le=80)
    length_cm: Optional[float] = Field(None, gt=20, le=130)
    sleeve_cm: Optional[float] = Field(None, gt=5, le=80)
    waist_width_cm: Optional[float] = Field(None, gt=15, le=80)   # flat half-width (lower body)
    hip_width_cm: Optional[float] = Field(None, gt=15, le=80)     # flat half-width (lower body)


class ProfileOut(BodyMeasurements):
    preferred_fit: str = "regular"


class FitVerdict(BaseModel):
    area: str
    garment: Optional[float] = None
    body: Optional[float] = None
    ease_cm: Optional[float] = None
    status: str            # tight | ideal | relaxed | loose | unknown
    note: str


class FitAnalysis(BaseModel):
    fit_preference: str
    category: str
    verdicts: List[FitVerdict]
    overall: str
    recommended_fit: str
    rules_used: dict
    disclaimer: str


class PreservationItem(BaseModel):
    aspect: str
    status: str            # preserved | potentially_altered | not_verified
    measured: bool
    detail: str
    metric: Optional[dict] = None


class PreservationReport(BaseModel):
    items: List[PreservationItem]
    summary: str
    method_notes: List[str]


class ReliabilityItem(BaseModel):
    check: str
    status: str            # pass | warning | unverified
    measured: bool
    detail: str


class ReliabilityReport(BaseModel):
    items: List[ReliabilityItem]
    notes: List[str]


class Recommendation(BaseModel):
    title: str
    suggestions: List[str]
    colors: List[str]
    note: str


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str            # queued | running | done | error
    stage: str
    message: str
    progress: float        # 0..1
    result: Optional[dict] = None
    error: Optional[str] = None
