# Agentic RAG - authentication endpoints (register / login / logout / me / password).

from fastapi import APIRouter, HTTPException, Request

import app.database.mongo as mongo
from app.api.dependencies import (auth_rate_clear, auth_rate_limit, bearer_token,
                                  require_user)
from app.schemas.users import ChangePasswordRequest, LoginRequest, RegisterRequest

router = APIRouter()


@router.post("/api/register")
def register(req: RegisterRequest, request: Request):
    auth_rate_limit(request)
    try:
        res = mongo.register_user(req.username, req.display_name, req.password)
        auth_rate_clear(request)
        return res
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/api/login")
def login(req: LoginRequest, request: Request):
    auth_rate_limit(request)
    res = mongo.login_user(req.username, req.password)
    if not res:
        raise HTTPException(status_code=401, detail="invalid username or password")
    auth_rate_clear(request)
    return res


@router.post("/api/logout")
def logout(request: Request):
    mongo.logout(bearer_token(request))
    return {"ok": True}


@router.get("/api/me")
def me(request: Request):
    return require_user(request)


@router.post("/api/password")
def change_password(req: ChangePasswordRequest, request: Request):
    """Change the signed-in user's password (verifies the current one first)."""
    user = require_user(request)  # 401 if not authenticated
    res = mongo.change_password(user["id"], req.current_password, req.new_password,
                                keep_token=bearer_token(request))
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "could not change password"))
    return {"ok": True}
