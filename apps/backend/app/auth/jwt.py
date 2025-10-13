from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings
from app.schemas.auth import TokenPayload


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except jwt.PyJWTError as exc:
        raise ValueError("Invalid token") from exc

    exp = payload.get("exp")
    if isinstance(exp, (int, float)):
        payload["exp"] = datetime.fromtimestamp(exp, tz=timezone.utc)

    return TokenPayload(**payload)
