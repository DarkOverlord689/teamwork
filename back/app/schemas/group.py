from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class GroupBase(BaseModel):
    course_id: UUID | None = None
    name: str


class GroupCreate(GroupBase):
    pass


class GroupResponse(GroupBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True