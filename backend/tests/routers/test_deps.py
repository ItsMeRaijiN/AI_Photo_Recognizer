import pytest
from fastapi import HTTPException

from backend.routers.deps import get_current_user, get_current_superuser


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_raises_401_when_no_user(self):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(user=None)
        assert exc.value.status_code == 401


class TestGetCurrentSuperuser:
    @pytest.mark.asyncio
    async def test_raises_403_for_regular_user(self, test_user):
        with pytest.raises(HTTPException) as exc:
            await get_current_superuser(user=test_user)
        assert exc.value.status_code == 403