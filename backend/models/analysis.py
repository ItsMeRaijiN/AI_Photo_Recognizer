from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.core.database import Base


class Analysis(Base):
    """Image analysis result model."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    file_hash = Column(String(64), index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)

    is_ai = Column(Boolean, nullable=False)
    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    threshold_used = Column(Float, nullable=False, default=0.5)
    inference_time_ms = Column(Float, nullable=False)

    model_type = Column(String(50), nullable=False)
    backbone_name = Column(String(50), nullable=False)
    model_version = Column(String(100), nullable=True)

    heatmap_path = Column(Text, nullable=True)

    custom_metrics = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    owner = relationship("User", back_populates="analyses")

    __table_args__ = (
        Index("ix_analyses_cache_lookup", "file_hash", "owner_id", "backbone_name"),
        Index("ix_analyses_owner_created", "owner_id", "created_at"),
        Index("ix_analyses_is_ai", "is_ai"),
    )

    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, filename='{self.filename}', is_ai={self.is_ai})>"

    @property
    def model_full_name(self) -> str:
        return f"{self.model_type}_{self.backbone_name}"

    @property
    def has_heatmap(self) -> bool:
        return bool(self.heatmap_path)
