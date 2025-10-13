from typing import Optional

import bcrypt
from fastapi import Cookie, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import decode_token
from app.core.config import settings
from app.db.session import get_session
from app.models.user import User


ACCESS_TOKEN_COOKIE_NAME = settings.access_token_cookie_name


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


async def authenticate_user(email: str, password: str, session: AsyncSession) -> Optional[User]:
    statement = select(User).where(User.email == email)
    result = await session.execute(statement)
    user = result.scalars().first()

    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def set_access_token_cookie(response: Response, token: str) -> None:
    max_age = settings.access_token_expire_minutes * 60
    response.set_cookie(
        ACCESS_TOKEN_COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        secure=settings.access_token_cookie_secure,
        samesite=settings.access_token_cookie_samesite,
    )


def clear_access_token_cookie(response: Response) -> None:
    response.delete_cookie(
        ACCESS_TOKEN_COOKIE_NAME,
        httponly=True,
        secure=settings.access_token_cookie_secure,
        samesite=settings.access_token_cookie_samesite,
    )


async def get_current_user(
    access_token: str | None = Cookie(default=None, alias=ACCESS_TOKEN_COOKIE_NAME),
    session: AsyncSession = Depends(get_session),
) -> User:
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    try:
        payload = decode_token(access_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        ) from exc

    try:
        user_id = int(payload.sub)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        ) from exc

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user
