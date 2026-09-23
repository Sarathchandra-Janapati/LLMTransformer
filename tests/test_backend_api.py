from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_attention_example_endpoint_returns_visualizer_payload() -> None:
    response = client.get("/api/attention/example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "I love transformers"
    assert payload["tokens"] == ["i", "love", "transformers"]
    assert payload["token_ids"] == [1, 2, 3]
    assert payload["attention_width"] == 2
    assert len(payload["key_weights"]) == 3
    assert len(payload["weights"]) == len(payload["tokens"])


def test_attention_run_endpoint_returns_tensors_for_posted_text() -> None:
    response = client.post(
        "/api/attention/run",
        json={"text": "tokens learn attention", "task_mode": "next_token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tokens"] == ["tokens", "learn", "attention"]
    assert payload["token_ids"] == [5, 6, 4]
    assert payload["causal_mask"] is True
    assert payload["task_mode"] == "next_token"
    assert payload["mask"][0] == [False, True, True]
    assert len(payload["position_encodings"]) == 3
    assert len(payload["attention_inputs"]) == 3
    assert len(payload["context"]) == 3


def test_attention_run_endpoint_rejects_too_many_tokens() -> None:
    response = client.post(
        "/api/attention/run",
        json={"text": "one two three four five six seven"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "text supports up to 6 tokens"


def test_attention_compare_endpoint_returns_unmasked_and_masked_payloads() -> None:
    response = client.post(
        "/api/attention/compare",
        json={"text": "i love attention"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "i love attention"
    assert payload["unmasked"]["causal_mask"] is False
    assert payload["masked"]["causal_mask"] is True
    assert payload["unmasked"]["tokens"] == payload["masked"]["tokens"]
    assert payload["unmasked"]["weights"][0][1] > 0
    assert payload["masked"]["weights"][0][1] == 0


def test_attention_run_endpoint_supports_custom_local_window_task() -> None:
    response = client.post(
        "/api/attention/run",
        json={"text": "i love attention tokens", "task_mode": "local_window"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_label"] == "Local window"
    assert payload["mask"][0] == [False, False, True, True]


def test_task_modes_endpoint_lists_learning_tasks() -> None:
    response = client.get("/api/attention/task-modes")

    assert response.status_code == 200
    modes = response.json()["task_modes"]
    assert "next_token" in modes
    assert "fill_blank" in modes
    assert "classification" in modes
    assert "local_window" in modes


def test_training_example_endpoint_returns_default_configuration() -> None:
    response = client.get("/api/training/example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_request"]["epochs"] == 40
    assert payload["default_request"]["train_embeddings"] is True
    assert payload["default_request"]["train_transformer"] is False
    assert payload["vocabulary"]["love"] == 2
    assert payload["task"] == "next-token prediction"


def test_training_run_endpoint_returns_metrics_and_predictions() -> None:
    response = client.post(
        "/api/training/run",
        json={
            "epochs": 2,
            "learning_rate": 0.08,
            "embedding_learning_rate": 0.01,
            "context_size": 1,
            "train_embeddings": True,
            "train_transformer": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    summary = payload["summary"]
    assert payload["run_path"].endswith("tiny_gpt_backend_latest.json")
    assert summary["epochs"] == 2
    assert summary["train_embeddings"] is True
    assert summary["train_transformer"] is False
    assert summary["initial_loss"] > summary["final_loss"]
    assert summary["target_token_ids"] == [2]
    assert len(summary["metrics"]) == summary["example_count"] * 2 * 2
    assert len(summary["token_embeddings"]) == len(summary["vocabulary"])


def test_training_latest_endpoint_reads_last_backend_run() -> None:
    client.post(
        "/api/training/run",
        json={"epochs": 1, "context_size": 1, "train_embeddings": False},
    )

    response = client.get("/api/training/latest")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["epochs"] == 1
    assert summary["train_embeddings"] is False
    assert "vocabulary_weights" in summary


def test_training_run_endpoint_rejects_invalid_epochs() -> None:
    response = client.post("/api/training/run", json={"epochs": 0})

    assert response.status_code == 422
