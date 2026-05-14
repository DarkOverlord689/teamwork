# SMATC-UPAO - AGENTS.md

This project is a multimodal team collaboration analysis system with a Python FastAPI backend and React/TypeScript frontend.

## Quick Commands

### Backend (Python/FastAPI)

```bash
# Install dependencies
cd backend && pip install -r requirements.txt

# Run development server
cd backend && uvicorn app.main:app --reload --port 8000

# Run a single test
cd backend && python -m pytest tests/ -v -k "test_name_here"

# Run all tests with coverage
cd backend && python -m pytest tests/ --cov=app --cov-report=term-missing

# Lint (if ruff installed)
cd backend && ruff check .

# Type check (if mypy installed)
cd backend && mypy app/
```

### Frontend (React/TypeScript)

```bash
# Install dependencies
cd frontend && npm install

# Run development server
cd frontend && npm run dev

# Run a single test (Jest pattern)
cd frontend && npm test -- --testPathPattern="filename.test"

# Build for production
cd frontend && npm run build

# Lint
cd frontend && npm run lint

# Type check
cd frontend && npx tsc --noEmit
```

### Infrastructure

```bash
# Start all services
cd infrastructure && docker-compose up -d

# View logs
cd infrastructure && docker-compose logs -f backend

# Stop all services
cd infrastructure && docker-compose down
```

## Code Style Guidelines

### Python (Backend)

**Imports**
- Use absolute imports: `from app.models import Group` (not `from ..models import Group`)
- Group imports in order: standard library, third-party, local application
- Example:
```python
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Group
from app.schemas import GroupResponse
```

**Formatting**
- Maximum line length: 100 characters
- Use Black for formatting (if configured): `black app/`
- Use 4 spaces for indentation (no tabs)
- Add trailing commas in multi-line imports

**Types**
- Use type hints for all function parameters and return values
- Use Python 3.11+ union syntax: `str | None` instead of `Optional[str]`
- Use `from typing import Literal` for string literal types
- Example:
```python
async def get_group(group_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> GroupResponse:
    ...
```

**Naming Conventions**
- Classes: `PascalCase` (e.g., `AnalysisSession`)
- Functions/variables: `snake_case` (e.g., `get_groups`, `group_id`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_WORKERS`)
- Private methods: prefix with underscore (e.g., `_internal_method`)
- Async functions: prefix with `async_` when ambiguous (e.g., `async_get_data`)

**Error Handling**
- Use FastAPI HTTPException for API errors
- Log errors with appropriate levels (loguru)
- Return structured error responses
- Example:
```python
from fastapi import HTTPException, status

async def get_group(group_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.get(Group, group_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Group {group_id} not found"
        )
    return result
```

**Database**
- Use async SQLAlchemy with asyncpg driver
- Use dependency injection for database sessions
- Never commit in the middle of functions; let the router handle it

**Configuration**
- Use Pydantic Settings: `from app.utils.config import settings`
- Never hardcode configuration values

### TypeScript/React (Frontend)

**Imports**
- Use absolute imports with path aliases (configure in tsconfig.json)
- Order: React/React-DOM, external libraries, internal components/hooks, utilities
- Example:
```typescript
import { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useParams } from "react-router-dom";
import { Box, Typography } from "@mui/material";

import { groupService } from "../services/api";
import type { RootState } from "../store";
```

**Formatting**
- Maximum line length: 100 characters
- Use ESLint + Prettier (automatic on save)
- 2 spaces for indentation

**Types**
- Define interfaces/types for all data structures
- Use TypeScript strict mode
- Example:
```typescript
interface StudentMetrics {
  id: string;
  name: string;
  speaking_time: number;
  turn_count: number;
  collaboration_score: number;
}
```

**Naming Conventions**
- Components: `PascalCase` (e.g., `Dashboard.tsx`)
- Functions/variables: `camelCase` (e.g., `getGroupData`)
- Constants: `UPPER_SNAKE_CASE`
- Files: `kebab-case` for non-component files (e.g., `api-config.ts`)

**Components**
- Use functional components with hooks
- Destructure props for readability
- Extract complex logic into custom hooks
- Example:
```typescript
export default function Dashboard() {
  const dispatch = useDispatch();
  const { groups, loading } = useSelector((state: RootState) => state.analysis);
  // ... component logic
}
```

**State Management (Redux)**
- Use Redux Toolkit with createSlice
- Keep slices focused (analysis, user, ui)
- Use typed hooks (`useAppDispatch`, `useAppSelector`)

**Error Handling**
- Use try/catch with async/await
- Show user-friendly error messages via Snackbar
- Log errors to console for debugging
- Example:
```typescript
try {
  const data = await groupService.list();
  dispatch(setGroups(data));
} catch (error) {
  console.error("Error loading groups:", error);
  dispatch(showSnackbar({ message: "Failed to load groups", severity: "error" }));
}
```

**UI Components**
- Use Material UI components
- Follow MUI naming patterns: `Card`, `CardContent`, `Box`
- Use responsive Grid: `<Grid item xs={12} md={6}>`

## Project Structure

```
cogno/
├── backend/
│   ├── app/
│   │   ├── api/v1/     # API endpoints
│   │   ├── core/        # Processing modules
│   │   ├── models/      # SQLAlchemy models
│   │   ├── schemas/     # Pydantic schemas
│   │   ├── services/   # Business logic
│   │   ├── tasks/      # Celery tasks
│   │   └── utils/      # Utilities
│   ├── tests/          # Test files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── hooks/      # Custom hooks
│   │   ├── pages/      # Page components
│   │   ├── services/   # API services
│   │   ├── store/      # Redux store
│   │   └── types/      # TypeScript types
│   └── package.json
├── infrastructure/     # Docker configs
└── docs/              # Documentation
```

## Testing Strategy

- Backend: pytest with async support
- Frontend: Vitest (built into Vite)
- Run full test suite before submitting PRs
- Aim for >80% code coverage

## Git Conventions

- Commit message format: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Example: `feat(analysis): add group detail endpoint`
- Never commit secrets or API keys
- Use `.env.example` for environment variables