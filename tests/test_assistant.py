from __future__ import annotations

from fastapi.testclient import TestClient

from agent_eval.web import create_app


def test_assistant_can_create_session_and_answer_with_citations(settings) -> None:
    client = TestClient(create_app(settings))

    home = client.get("/assistant")
    assert home.status_code == 200

    ask = client.post(
        "/assistant/ask",
        json={"prompt": "In FastAPI, how can I hide internal fields from a response?"},
    )
    assert ask.status_code == 200

    payload = ask.json()
    assert payload["session"]["session_id"].startswith("session_")
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["citations"]
    assert payload["messages"][1]["run_payload"]["steps"]

    session_id = payload["session"]["session_id"]
    detail = client.get(f"/assistant/{session_id}?format=json")
    assert detail.status_code == 200
    assert detail.json()["session"]["session_id"] == session_id


def test_assistant_can_use_existing_session(settings) -> None:
    client = TestClient(create_app(settings))

    first = client.post("/assistant/ask", json={"prompt": "What does SQLite FTS5 do?"}).json()
    session_id = first["session"]["session_id"]

    second = client.post(
        "/assistant/ask",
        json={
            "session_id": session_id,
            "prompt": "And what is WAL mode used for?",
            "config_id": "verifier_heuristic",
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["session"]["session_id"] == session_id
    assert len(payload["messages"]) == 4
