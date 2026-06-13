import json

import pytest
from cryptography.fernet import Fernet

from app.application.security.app_config_encryption import AppConfigEncryption
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    NotebookConfig,
    NotebookEmbeddingConfig,
)
from app.infrastructure.repositories.db_app_config_repository import (
    DBAppConfigRepository,
)


def test_app_config_encryption_encrypts_and_decrypts_sensitive_fields():
    encryption = AppConfigEncryption(Fernet.generate_key().decode())
    app_config = AppConfig(
        llm_config=LLMConfig(api_key="llm-secret", model_name="model-a"),
        agent_config=AgentConfig(max_iterations=12),
        mcp_config=MCPConfig(
            mcpServers={
                "demo": MCPServerConfig(
                    url="https://mcp.example.com",
                    headers={"Authorization": "Bearer mcp-secret"},
                    env={"TOKEN": "env-secret"},
                )
            }
        ),
        a2a_config=A2AConfig(),
        notebook_config=NotebookConfig(
            embedding_config=NotebookEmbeddingConfig(api_key="embedding-secret")
        ),
    )

    encrypted_data = encryption.encrypt_app_config(app_config)
    encrypted_json = json.dumps(encrypted_data)

    assert "llm-secret" not in encrypted_json
    assert "mcp-secret" not in encrypted_json
    assert "env-secret" not in encrypted_json
    assert "embedding-secret" not in encrypted_json
    assert encrypted_data["llm_config"]["api_key"].startswith("enc:v1:")
    assert encrypted_data["notebook_config"]["embedding_config"][
        "api_key"
    ].startswith("enc:v1:")
    server_data = encrypted_data["mcp_config"]["mcpServers"]["demo"]
    assert server_data["headers"].startswith("enc:v1:")
    assert server_data["env"].startswith("enc:v1:")

    decrypted_data = encryption.decrypt_app_config_data(encrypted_data)
    decrypted_config = AppConfig.model_validate(decrypted_data)
    assert decrypted_config.llm_config.api_key == "llm-secret"
    assert decrypted_config.mcp_config.mcpServers["demo"].headers == {
        "Authorization": "Bearer mcp-secret"
    }
    assert decrypted_config.mcp_config.mcpServers["demo"].env == {
        "TOKEN": "env-secret"
    }
    assert (
        decrypted_config.notebook_config.embedding_config.api_key
        == "embedding-secret"
    )


def test_app_config_encryption_accepts_plain_secret_key():
    encryption = AppConfigEncryption("plain-development-secret")
    app_config = AppConfig(
        llm_config=LLMConfig(api_key="llm-secret"),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
    )

    encrypted_data = encryption.encrypt_app_config(app_config)
    decrypted_data = encryption.decrypt_app_config_data(encrypted_data)

    assert encrypted_data["llm_config"]["api_key"].startswith("enc:v1:")
    assert decrypted_data["llm_config"]["api_key"] == "llm-secret"


def test_app_config_encryption_is_compatible_with_plaintext_and_empty_key():
    plaintext_config = AppConfig(
        llm_config=LLMConfig(api_key="plain-secret"),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
    )
    plaintext_data = plaintext_config.model_dump(mode="json")

    disabled_encryption = AppConfigEncryption("")
    assert disabled_encryption.encrypt_app_config(plaintext_config) == plaintext_data
    assert disabled_encryption.decrypt_app_config_data(plaintext_data) == plaintext_data

    encryption = AppConfigEncryption(Fernet.generate_key().decode())
    assert encryption.decrypt_app_config_data(plaintext_data) == plaintext_data


def test_app_config_encryption_does_not_encrypt_sensitive_values_twice():
    encryption = AppConfigEncryption(Fernet.generate_key().decode())
    app_config = AppConfig(
        llm_config=LLMConfig(api_key="llm-secret"),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
    )
    encrypted_once = encryption.encrypt_app_config(app_config)
    encrypted_twice = encryption.encrypt_app_config_data(encrypted_once)

    assert encrypted_twice["llm_config"]["api_key"] == encrypted_once["llm_config"][
        "api_key"
    ]


@pytest.mark.anyio
async def test_db_app_config_repository_stores_encrypted_and_returns_decrypted():
    db_session = FakeDBSession()
    repository = DBAppConfigRepository(
        db_session=db_session,
        encryption=AppConfigEncryption(Fernet.generate_key().decode()),
    )
    app_config = AppConfig(
        llm_config=LLMConfig(api_key="llm-secret"),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(
            mcpServers={
                "demo": MCPServerConfig(
                    url="https://mcp.example.com",
                    env={"TOKEN": "env-secret"},
                )
            }
        ),
        a2a_config=A2AConfig(),
    )

    await repository.save("user-a", app_config)

    stored_record = db_session.record
    assert stored_record.llm_config["api_key"].startswith("enc:v1:")
    assert stored_record.mcp_config["mcpServers"]["demo"]["env"].startswith("enc:v1:")
    assert "llm-secret" not in json.dumps(stored_record.llm_config)
    assert "env-secret" not in json.dumps(stored_record.mcp_config)

    loaded_config = await repository.get_by_user_id("user-a")
    assert loaded_config.llm_config.api_key == "llm-secret"
    assert loaded_config.mcp_config.mcpServers["demo"].env == {"TOKEN": "env-secret"}


class FakeDBSession:
    def __init__(self):
        self.record = None

    async def execute(self, stmt):
        return FakeResult(self.record)

    def add(self, record):
        self.record = record


class FakeResult:
    def __init__(self, record):
        self.record = record

    def scalar_one_or_none(self):
        return self.record
