"""Optional single-object write tools with explicit server-side policy."""

import json
from typing import Annotated, Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from netbox_mcp_server.config import Settings
from netbox_mcp_server.netbox_client import NetBoxClientBase

ObjectID = Annotated[int, Field(gt=0, strict=True)]


def register_write_tools(
    mcp: FastMCP,
    client: NetBoxClientBase,
    settings: Settings,
    object_types: dict[str, dict[str, str]],
) -> None:
    """Register allowed write operations; retain policy checks at invocation time."""
    if not settings.netbox_enable_write or not settings.netbox_write_allowed_object_types:
        return

    def execute(
        action: str,
        object_type: str,
        object_id: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not settings.netbox_enable_write:
            raise ToolError("Writes are disabled")
        if action == "delete" and not settings.netbox_enable_delete:
            raise ToolError("Deletion is disabled")
        if object_type not in settings.netbox_write_allowed_object_types:
            raise ToolError(f"Writes are not allowed for object_type '{object_type}'")
        if object_type not in object_types:
            raise ToolError(f"Unknown object_type '{object_type}'")
        if action != "create" and (type(object_id) is not int or object_id <= 0):
            raise ToolError("object_id must be a positive integer")
        if action != "delete" and not data:
            raise ToolError("data must be a non-empty object")
        if data and "id" in data:
            raise ToolError("Do not include id in data; use object_id for updates")

        endpoint = object_types[object_type]["endpoint"]
        try:
            if action == "create":
                return client.create(endpoint, data)
            if action == "update":
                return client.update(endpoint, object_id, data)
            if not client.delete(endpoint, object_id):
                raise ToolError(
                    "NetBox did not confirm deletion; inspect the object before retrying"
                )
            return {"object_type": object_type, "object_id": object_id, "deleted": True}
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            messages = {
                400: "Validation failed or operation blocked by object constraints",
                401: "NetBox authentication failed",
                403: "NetBox permission denied",
                404: "NetBox object or endpoint not found",
                405: "Operation is not supported by this endpoint",
                409: "Operation conflicts with existing data or object constraints",
            }
            message = messages.get(status, "NetBox write request failed")
            # Only return structured validation details, never HTML or request headers.
            details = ""
            if status in (400, 409):
                try:
                    body = exc.response.json()
                except ValueError:
                    body = None
                if isinstance(body, (dict, list)):
                    details = f": {json.dumps(body, ensure_ascii=False)}"
                    details = details.replace(
                        settings.netbox_token.get_secret_value(), "[REDACTED]"
                    )
            if status >= 500:
                message += "; outcome may be unknown, inspect NetBox before retrying"
            raise ToolError(f"{message} (HTTP {status}){details}") from None
        except httpx.RequestError:
            raise ToolError(
                "NetBox communication failed; write outcome may be unknown. "
                "Inspect NetBox before retrying; this request was not automatically retried."
            ) from None

    @mcp.tool(
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False}
    )
    def netbox_create_object(object_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create one NetBox object and return it.

        Args:
            object_type: Exact registered type, e.g. dcim.site; must be write-allowed.
            data: Non-empty writable fields, e.g. {"name": "Lab", "slug": "lab"}.
                Resolve related objects using read tools and supply their numeric IDs.
                Required fields and choices depend on the NetBox version and model.
        """
        return execute("create", object_type, data=data)

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False})
    def netbox_update_object(
        object_type: str,
        object_id: ObjectID,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Patch one object by ID and return the updated object.

        Args:
            object_type: Exact registered, write-allowed type, e.g. dcim.device.
            object_id: Positive numeric ID obtained using read tools.
            data: Only fields to change, e.g. {"description": "Updated"}.
                Omitted fields stay unchanged; null is sent explicitly to NetBox.
                List fields may replace existing lists. No automatic retry.
        """
        return execute("update", object_type, object_id, data)

    if settings.netbox_enable_delete:

        @mcp.tool(
            annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False}
        )
        def netbox_delete_object(object_type: str, object_id: ObjectID) -> dict[str, Any]:
            """Permanently delete one object by ID and return deletion confirmation.

            Args:
                object_type: Exact registered, write-allowed type.
                object_id: Positive numeric ID. Inspect the object before deleting.
                    NetBox may cascade deletion to related objects or reject it due
                    to constraints. There is no undo or automatic retry.
            """
            return execute("delete", object_type, object_id)
