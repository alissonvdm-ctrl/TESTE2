"""Integration tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def client():
    """Create a test client with mocked DB and Redis."""
    with (
        patch("backend.db.session.init_db", new_callable=AsyncMock),
        patch("backend.db.session.close_db", new_callable=AsyncMock),
        patch("backend.core.database.init_db", new_callable=AsyncMock),
        patch("backend.core.database.close_db", new_callable=AsyncMock),
        patch("redis.asyncio.from_url") as mock_redis,
    ):
        mock_r = AsyncMock()
        mock_r.ping = AsyncMock(return_value=True)
        mock_r.aclose = AsyncMock()
        mock_redis.return_value = mock_r

        from backend.api.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_root_redirects(self, client):
        resp = client.get("/", follow_redirects=False)
        # Root returns a JSON message
        assert resp.status_code in (200, 307, 302)

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "openapi" in data
        assert "paths" in data


class TestUploadEndpoints:
    def test_upload_invalid_extension(self, client):
        resp = client.post(
            "/uploads/excel",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 415

    def test_upload_empty_body_fails(self, client):
        # Sending nothing should fail
        resp = client.post("/uploads/excel")
        assert resp.status_code == 422


class TestExperimentEndpoints:
    def test_list_experiments(self, client):
        resp = client.get("/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["items"] == []

    def test_get_nonexistent_experiment(self, client):
        resp = client.get("/experiments/nonexistent-id")
        assert resp.status_code == 404
