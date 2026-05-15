"""database.py - Configuración de la base de datos

Configura SQLAlchemy asíncrono con asyncpg para PostgreSQL:
- engine:      motor de base de datos que maneja la conexión
- Base:        clase base para los modelos declarativos
- AsyncSessionLocal: fábrica de sesiones asíncronas
- get_db:      dependencia de FastAPI para inyectar sesiones

La URL de conexión se lee desde la configuración (Settings).
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.utils.config import settings


class Base(DeclarativeBase):
    """Clase base para todos los modelos SQLAlchemy."""

    pass


engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """Dependencia de FastAPI que provee una sesión de base de datos por request."""
    async with AsyncSessionLocal() as session:
        yield session
