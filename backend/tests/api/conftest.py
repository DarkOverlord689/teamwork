"""Conftest for API tests.

Mocks out the database engine at import time so that tests can run without
asyncpg/PostgreSQL installed in the development environment.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True, scope="session")
def _mock_db_engine():
    """Prevent SQLAlchemy from trying to connect to PostgreSQL on import."""
    mock_engine = MagicMock()
    mock_engine.begin = MagicMock(return_value=AsyncMock().__aenter__())

    with patch("app.database.engine", mock_engine), \
         patch("app.database.Base", MagicMock()), \
         patch("app.database.get_db", return_value=AsyncMock()):
        yield
