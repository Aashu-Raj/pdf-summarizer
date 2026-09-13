from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.deps import auth_db, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    username: str = Field(min_length=3)
    password: str = Field(min_length=6)
    confirm_password: str = Field(min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    if not body.username or not body.password:
        raise HTTPException(status_code=400, detail="Please fill in all fields.")

    success, result = auth_db.authenticate_user(body.username, body.password)
    if not success:
        raise HTTPException(status_code=401, detail=result)

    token = create_access_token(result)
    return TokenResponse(
        access_token=token,
        user={
            "username": result["username"],
            "role": result["role"],
            "full_name": result["full_name"],
            "email": result.get("email"),
            "status": result.get("status"),
        },
    )


@router.post("/register")
def register(body: RegisterRequest):
    if body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Match Streamlit registration: pending status (approval system)
    success, message = auth_db.create_user(
        username=body.username,
        email=body.email,
        full_name=body.full_name,
        password=body.password,
        role="user",
        status="pending",
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"message": message}


@router.get("/me")
def me(user: Annotated[dict, Depends(get_current_user)]):
    return user


@router.post("/logout")
def logout(_user: Annotated[dict, Depends(get_current_user)]):
    # JWT is cleared client-side; endpoint exists for session symmetry.
    return {"message": "Logged out"}


def serialize_user_row(row: dict) -> dict:
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
