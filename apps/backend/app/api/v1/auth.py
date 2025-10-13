from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import create_access_token
from app.auth.security import (
    authenticate_user,
    clear_access_token_cookie,
    get_current_user,
    hash_password,
    set_access_token_cookie,
)
from app.db.session import get_session
from app.models.user import User
from app.schemas.auth import UserCreate, UserLogin, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def signup(
    user_in: UserCreate,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    result = await session.execute(select(User).where(User.email == user_in.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(email=user_in.email, hashed_password=hash_password(user_in.password))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return UserRead.model_validate(user)


@router.post("/login", response_model=UserRead)
async def login(
    credentials: UserLogin,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    user = await authenticate_user(credentials.email, credentials.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email or password"
        )

    access_token = create_access_token(subject=str(user.id))
    set_access_token_cookie(response, access_token)
    return UserRead.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    clear_access_token_cookie(response)


@router.get("/me", response_model=UserRead)
async def read_current_user(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)
