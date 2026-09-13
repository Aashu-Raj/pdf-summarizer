from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import auth_db, require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


def serialize_user(row: dict) -> dict:
    created = row.get("created_at")
    if isinstance(created, datetime):
        created = created.isoformat()
    return {
        "id": row.get("id"),
        "username": row["username"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "status": row["status"],
        "created_at": created,
    }


class StatusUpdate(BaseModel):
    status: Literal["approved", "pending", "rejected"]


class RoleUpdate(BaseModel):
    role: Literal["admin", "user"]


@router.get("/users")
def list_users(_admin: Annotated[dict, Depends(require_admin)]):
    users = [serialize_user(u) for u in auth_db.get_all_users()]
    pending = [serialize_user(u) for u in auth_db.get_pending_users()]
    return {
        "users": users,
        "pending": pending,
        "stats": {
            "approved": sum(1 for u in users if u["status"] == "approved"),
            "pending": sum(1 for u in users if u["status"] == "pending"),
            "rejected": sum(1 for u in users if u["status"] == "rejected"),
        },
    }


@router.patch("/users/{username}/status")
def update_status(
    username: str,
    body: StatusUpdate,
    _admin: Annotated[dict, Depends(require_admin)],
):
    if not auth_db.update_user_status(username, body.status):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"Updated {username} status to {body.status}"}


@router.patch("/users/{username}/role")
def update_role(
    username: str,
    body: RoleUpdate,
    admin: Annotated[dict, Depends(require_admin)],
):
    if username == admin["username"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    if not auth_db.update_user_role(username, body.role):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"Changed {username} to {body.role}"}


@router.delete("/users/{username}")
def delete_user(
    username: str,
    admin: Annotated[dict, Depends(require_admin)],
):
    if username == admin["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if not auth_db.delete_user(username):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"Deleted {username}"}
