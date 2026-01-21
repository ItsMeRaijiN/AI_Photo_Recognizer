from sqlalchemy.orm import Session

from backend.models.analysis import Analysis
from backend.models.user import User


class TestAnalysisModel:
    def test_create_analysis_with_metrics(self, test_db: Session, test_user: User):
        metrics_data = {
            "blur_detection": {"score": 100, "label": "sharp"},
            "noise_analysis": {"level": 0.5},
        }

        analysis = Analysis(
            file_hash="unique_hash_999",
            filename="photo.jpg",
            file_path="/uploads/photo.jpg",
            is_ai=True,
            score=0.95,
            confidence=0.90,
            inference_time_ms=150.5,
            model_type="torch",
            backbone_name="convnext_tiny",
            custom_metrics=metrics_data,
            owner_id=test_user.id,
        )
        test_db.add(analysis)
        test_db.commit()

        assert analysis.id is not None
        assert analysis.custom_metrics["blur_detection"]["label"] == "sharp"
        assert analysis.owner_id == test_user.id
        assert analysis.owner.username == test_user.username

    def test_guest_analysis_allowed(self, test_db: Session):
        analysis = Analysis(
            file_hash="guest_hash",
            filename="guest.jpg",
            file_path="path",
            is_ai=False,
            score=0.1,
            confidence=0.9,
            inference_time_ms=5,
            model_type="t",
            backbone_name="b",
            owner_id=None,
        )
        test_db.add(analysis)
        test_db.commit()

        assert analysis.id is not None
        assert analysis.owner_id is None

class TestAnalysisCascade:
    def test_deleting_user_deletes_analyses(self, test_db: Session, test_user: User):
        analysis = Analysis(
            file_hash="hash_to_delete",
            filename="test.jpg",
            file_path="path",
            is_ai=True,
            score=0.5,
            confidence=0.5,
            inference_time_ms=10,
            model_type="t",
            backbone_name="b",
            owner_id=test_user.id,
        )
        test_db.add(analysis)
        test_db.commit()

        test_db.delete(test_user)
        test_db.commit()
        test_db.expire_all()

        deleted = test_db.query(Analysis).filter_by(file_hash="hash_to_delete").first()
        assert deleted is None