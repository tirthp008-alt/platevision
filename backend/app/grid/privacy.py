"""Privacy helpers: plate masking and role-based access control.

The system must not assume unrestricted access to surveillance data, so plate
text is masked for roles that lack the ``view_full_plate`` permission and every
privacy-sensitive action is written to the audit log.
"""

import os
from typing import Optional

from fastapi import Header, HTTPException, Request, status

# Roles -> permissions
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer": {"read"},
    "operator": {"read", "write_camera", "control_stream"},
    "admin": {
        "read",
        "write_camera",
        "control_stream",
        "view_full_plate",
        "view_audit",
        "manage_fusion",
    },
}


def _key_map() -> dict[str, str]:
    """Resolve role from API key env vars so no secret is committed to the repo."""
    mapping: dict[str, str] = {}
    for role, var in (
        ("admin", "DRISHTI_ADMIN_KEY"),
        ("operator", "DRISHTI_OPERATOR_KEY"),
        ("viewer", "DRISHTI_VIEWER_KEY"),
    ):
        value = os.getenv(var)
        if value:
            mapping[value] = role
    return mapping


def resolve_role(api_key: Optional[str]) -> str:
    if not api_key:
        return "viewer"
    return _key_map().get(api_key, "viewer")


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())


def mask_plate(plate: str, keep_prefix: int = 4, keep_suffix: int = 2) -> str:
    """Mask the middle of a plate: 'GJ01AB1234' -> 'GJ01****34'."""
    if not plate:
        return ""
    if len(plate) <= keep_prefix + keep_suffix:
        return plate[: max(0, len(plate) - 2)] + "**"
    return plate[:keep_prefix] + "*" * (len(plate) - keep_prefix - keep_suffix) + plate[-keep_suffix:]


async def get_access_context(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    """FastAPI dependency returning role/actor/permissions for the caller."""
    role = resolve_role(x_api_key)
    actor = role if x_api_key else "anonymous"
    client_ip = request.client.host if request.client else None
    return {
        "role": role,
        "actor": actor,
        "can_view_full_plate": has_permission(role, "view_full_plate"),
        "client_ip": client_ip,
    }


def enforce(ctx: dict, permission: str) -> None:
    if not has_permission(ctx.get("role", "viewer"), permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": f"Role '{ctx.get('role')}' lacks permission '{permission}'.",
                "details": None,
            },
        )
