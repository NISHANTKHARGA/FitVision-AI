"""ORM models: We never store raw person/garment photographs by default,
only the generated try-on image is kept on disk, and it is persisted to the
database-backed history only when the user explicitly asks to save it."""
import datetime as dt

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text

from .db import Base


class BodyProfile(Base):
    __tablename__ = "body_profile"

    id = Column(Integer, primary_key=True, default=1)
    height_cm = Column(Float, nullable=True)
    chest_cm = Column(Float, nullable=True)
    waist_cm = Column(Float, nullable=True)
    shoulder_cm = Column(Float, nullable=True)
    hip_cm = Column(Float, nullable=True)
    preferred_fit = Column(String(16), default="regular")
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


class ResultRecord(Base):
    __tablename__ = "result_records"

    id = Column(String(36), primary_key=True)          # job id
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    category = Column(String(16), nullable=False)       # tops | bottoms
    garment_type = Column(String(32), default="")       # free text, e.g. "t-shirt"
    fit = Column(String(16), nullable=False)            # slim | regular | oversized
    steps = Column(Integer, default=20)
    seed = Column(Integer, default=42)
    saved = Column(Boolean, default=False)              # user chose to keep it
    has_measurements = Column(Boolean, default=False)
    summary_json = Column(Text, default="{}")           # small report summary for history cards
    image_path = Column(String(512), nullable=True)     # relative to RESULTS_DIR
    person_image_path = Column(String(512), nullable=True)
