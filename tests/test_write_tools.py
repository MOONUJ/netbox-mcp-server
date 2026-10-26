"""Exercise writes through the MCP protocol and a real HTTPX mock transport."""

import asyncio
import sys

import httpx
import pytest
from fastmcp import Client, FastMCP
from pydantic import ValidationError

from netbox_mcp_server.config import Settings
from netbox_mcp_server.netbox_client import NetBoxRestClient
from netbox_mcp_server.server import parse_cli_args
from netbox_mcp_server.write_tools import register_write_tools

TYPES = {"dcim.site": {"endpoint": "dcim/sites"}}
TOOL_NAMES = {"netbox_create_object", "netbox_update_object", "netbox_delete_object"}


@pytest.fixture
def setup(monkeypatch):
    # Isolate configuration from developer environment and .env.
    for key in list(__import__("os").environ):
        monkeypatch.delenv(key)
    requests = []

    def build(handler=None, **overrides):
        values = {
            "netbox_url": "https://netbox.example.com",
            "netbox_token": "secret-test-token",
            "netbox_enable_write": True,
            "netbox_enable_delete": True,
            "netbox_write_allowed_object_types": ["dcim.site"],
        }
        values.update(overrides)
        settings = Settings(_env_file=None, **values)
        api = NetBoxRestClient(str(settings.netbox_url), settings.netbox_token.get_secret_value())
        api.session.close()

        def respond(request):
            requests.append(request)
            if handler:
                return handler(request)
            if request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(
                201 if request.method == "POST" else 200,
                json={"id": 7, "name": "Lab"},
            )

        api.session = httpx.Client(transport=httpx.MockTransport(respond))
        mcp = FastMCP("write-test")
        register_write_tools(mcp, api, settings, TYPES)
        return mcp, settings, requests, api

    return build


def call(mcp, name, args):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, args, raise_on_error=False)

    return asyncio.run(run())


def names(mcp):
    async def run():
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"netbox_enable_write": False, "netbox_enable_delete": False}, set()),
        ({"netbox_write_allowed_object_types": []}, set()),
        ({"netbox_enable_delete": False}, TOOL_NAMES - {"netbox_delete_object"}),
        ({}, TOOL_NAMES),
    ],
)
def test_tool_exposure(setup, overrides, expected):
    mcp, _, _, _ = setup(**overrides)
    assert names(mcp) == expected


@pytest.mark.parametrize(
    ("action", "args", "method", "path", "body"),
    [
        (
            "create",
            {"data": {"name": "Lab", "slug": "lab"}},
            "POST",
            "/api/dcim/sites/",
            b'{"name":"Lab","slug":"lab"}',
        ),
        (
            "update",
            {"object_id": 7, "data": {"description": None}},
            "PATCH",
            "/api/dcim/sites/7/",
            b'{"description":null}',
        ),
        ("delete", {"object_id": 7}, "DELETE", "/api/dcim/sites/7/", b""),
    ],
)
def test_request_and_response(setup, action, args, method, path, body):
    mcp, _, requests, _ = setup()
    result = call(mcp, f"netbox_{action}_object", {"object_type": "dcim.site", **args})
    assert not result.is_error
    assert len(requests) == 1
    assert (requests[0].method, requests[0].url.path, requests[0].content) == (method, path, body)
    assert result.data == (
        {"object_type": "dcim.site", "object_id": 7, "deleted": True}
        if action == "delete"
        else {"id": 7, "name": "Lab"}
    )


@pytest.mark.parametrize(
    ("action", "args"),
    [
        ("create", {"data": {}}),
        ("create", {"data": []}),
        ("create", {"data": {"id": 1}}),
        ("update", {"object_id": 0, "data": {"name": "Lab"}}),
        ("update", {"object_id": True, "data": {"name": "Lab"}}),
        ("update", {"object_id": 7, "data": {}}),
        ("delete", {"object_id": -1}),
    ],
)
def test_invalid_input_never_calls_netbox(setup, action, args):
    mcp, _, requests, _ = setup()
    result = call(mcp, f"netbox_{action}_object", {"object_type": "dcim.site", **args})
    assert result.is_error
    assert not requests


@pytest.mark.parametrize("object_type", ["users.token", "../users/tokens", "plugin.unknown"])
def test_unallowed_type(setup, object_type):
    mcp, _, requests, _ = setup()
    result = call(
        mcp, "netbox_create_object", {"object_type": object_type, "data": {"name": "Lab"}}
    )
    assert result.is_error
    assert not requests


@pytest.mark.parametrize("policy", ["write", "delete", "allowlist"])
def test_policy_rechecked_at_execution(setup, policy):
    mcp, settings, requests, _ = setup()
    if policy == "write":
        settings.netbox_enable_write = False
    elif policy == "delete":
        settings.netbox_enable_delete = False
    else:
        settings.netbox_write_allowed_object_types = []
    result = call(mcp, "netbox_delete_object", {"object_type": "dcim.site", "object_id": 7})
    assert result.is_error
    assert not requests


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 500])
def test_api_errors(setup, status):
    mcp, _, requests, _ = setup(
        lambda request: httpx.Response(status, json={"name": ["Invalid secret-test-token"]})
    )
    result = call(
        mcp, "netbox_create_object", {"object_type": "dcim.site", "data": {"name": "Lab"}}
    )
    assert result.is_error
    message = result.content[0].text
    assert f"HTTP {status}" in message
    assert "secret-test-token" not in message
    if status in (400, 409):
        assert "name" in message
    assert len(requests) == 1


def test_timeout_not_retried(setup):
    def timeout(request):
        raise httpx.ReadTimeout("private request details", request=request)

    mcp, _, requests, _ = setup(timeout)
    result = call(
        mcp, "netbox_create_object", {"object_type": "dcim.site", "data": {"name": "Lab"}}
    )
    assert result.is_error
    assert "outcome may be unknown" in result.content[0].text
    assert "private request details" not in result.content[0].text
    assert len(requests) == 1


def test_delete_requires_confirmation_from_api(setup):
    mcp, _, _, _ = setup(lambda request: httpx.Response(200, json={}))
    result = call(mcp, "netbox_delete_object", {"object_type": "dcim.site", "object_id": 7})
    assert result.is_error
    assert "did not confirm deletion" in result.content[0].text


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"netbox_enable_write": False}, "requires NETBOX_ENABLE_WRITE"),
        ({"transport": "http"}, "HTTP writes require MCP_AUTH_TOKEN"),
        ({"transport": "http", "mcp_auth_token": "  "}, "HTTP writes require MCP_AUTH_TOKEN"),
    ],
)
def test_invalid_config(setup, overrides, error):
    with pytest.raises(ValidationError, match=error):
        setup(**overrides)


def test_http_writes_with_auth(setup):
    mcp, _, _, _ = setup(transport="http", mcp_auth_token="mcp-secret")
    assert names(mcp) == TOOL_NAMES


def test_cli_overrides_environment(setup, monkeypatch):
    monkeypatch.setenv("NETBOX_ENABLE_WRITE", "true")
    monkeypatch.setenv("NETBOX_WRITE_ALLOWED_OBJECT_TYPES", '["dcim.device"]')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--no-netbox-enable-write",
            "--no-netbox-enable-delete",
            "--netbox-write-allowed-object-types",
        ],
    )
    _, settings, _, _ = setup(**parse_cli_args())
    assert not settings.netbox_enable_write
    assert not settings.netbox_enable_delete
    assert settings.netbox_write_allowed_object_types == []


def test_default_settings_are_read_only(setup):
    settings = Settings(
        netbox_url="https://netbox.example.com", netbox_token="test", _env_file=None
    )
    assert not settings.netbox_enable_write
    assert not settings.netbox_enable_delete
    assert settings.netbox_write_allowed_object_types == []


def test_environment_allowlist(setup, monkeypatch):
    monkeypatch.setenv("NETBOX_ENABLE_WRITE", "true")
    monkeypatch.setenv("NETBOX_WRITE_ALLOWED_OBJECT_TYPES", '["dcim.site","dcim.device"]')
    settings = Settings(
        netbox_url="https://netbox.example.com", netbox_token="test", _env_file=None
    )
    assert settings.netbox_enable_write
    assert settings.netbox_write_allowed_object_types == ["dcim.site", "dcim.device"]


def test_plugin_must_be_registered_and_allowlisted(setup):
    mcp, _, requests, api = setup(
        netbox_write_allowed_object_types=["example.widget"],
    )
    args = {"object_type": "example.widget", "data": {"name": "Widget"}}
    assert call(mcp, "netbox_create_object", args).is_error
    assert not requests
    settings = Settings(
        netbox_url="https://netbox.example.com",
        netbox_token="test",
        _env_file=None,
        netbox_enable_write=True,
        netbox_write_allowed_object_types=["example.widget"],
    )
    plugin_mcp = FastMCP("plugin-write-test")
    register_write_tools(
        plugin_mcp,
        api,
        settings,
        {"example.widget": {"endpoint": "plugins/example/widgets"}},
    )
    assert not call(plugin_mcp, "netbox_create_object", args).is_error
    assert requests[0].url.path == "/api/plugins/example/widgets/"


def test_main_registers_tools_before_serving(setup, monkeypatch):
    from netbox_mcp_server import server

    _, settings, requests, api = setup()
    app = FastMCP("startup-test")
    monkeypatch.setattr(server, "mcp", app)
    monkeypatch.setattr(server, "netbox", None)
    monkeypatch.setattr(server, "parse_cli_args", lambda: {})
    monkeypatch.setattr(server, "Settings", lambda **kwargs: settings)
    monkeypatch.setattr(server, "NetBoxRestClient", lambda **kwargs: api)
    observed = []

    def run(**kwargs):
        observed.append(names(app))
        result = call(
            app,
            "netbox_create_object",
            {
                "object_type": "dcim.site",
                "data": {"name": "Lab", "slug": "lab"},
            },
        )
        assert not result.is_error

    monkeypatch.setattr(app, "run", run)
    server.main()
    assert observed == [TOOL_NAMES]
    assert len(requests) == 1
