from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, EmailStr, field_serializer


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserLogin(UserBase):
    password: str


class UserRead(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> datetime:
        sydney_tz = ZoneInfo("Australia/Sydney")
        if value.tzinfo:
            return value.astimezone(sydney_tz)
        return value.replace(tzinfo=ZoneInfo("UTC")).astimezone(sydney_tz)


class TokenPayload(BaseModel):
    sub: str
    exp: datetime
