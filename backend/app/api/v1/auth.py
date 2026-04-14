from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta

from app.database import get_db
from app.utils.config import settings

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    if form_data.username == "demo" and form_data.password == "demo":
        return {"access_token": "demo-token", "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas")


@router.post("/register")
async def register(db: AsyncSession = Depends(get_db)):
    return {"message": "Registro no implementado"}


@router.get("/me")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    return {"user_id": "demo-user", "username": "demo"}