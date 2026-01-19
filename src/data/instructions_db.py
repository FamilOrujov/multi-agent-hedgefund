
import asyncpg
from datetime import datetime
from typing import List, Dict, Optional
from src.config.settings import get_settings


class InstructionsDB:
    """Database manager for HITL instructions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                database=self.settings.postgres_db,
                min_size=1,
                max_size=10,
            )
    
    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    async def init_tables(self):
        """Initialize database tables."""
        await self.connect()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hitl_instructions (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10),
                    instruction TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    modified_at TIMESTAMP DEFAULT NOW(),
                    review_id VARCHAR(100),
                    decision VARCHAR(20),
                    confidence INTEGER,
                    tags TEXT[]
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hitl_ticker ON hitl_instructions(ticker)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hitl_created ON hitl_instructions(created_at DESC)
            """)
    
    async def add_instruction(
        self,
        instruction: str,
        ticker: Optional[str] = None,
        review_id: Optional[str] = None,
        decision: Optional[str] = None,
        confidence: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Add a new instruction."""
        await self.connect()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO hitl_instructions 
                (instruction, ticker, review_id, decision, confidence, tags)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, instruction, ticker, review_id, decision, confidence, tags or [])
            return row['id']
    
    async def get_all_instructions(self) -> List[Dict]:
        """Get all instructions ordered by date."""
        await self.connect()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, ticker, instruction, created_at, review_id, decision, confidence, tags
                FROM hitl_instructions
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in rows]
    
    async def get_instructions_by_ticker(self, ticker: str) -> List[Dict]:
        """Get instructions for a specific ticker."""
        await self.connect()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, ticker, instruction, created_at, review_id, decision, confidence, tags
                FROM hitl_instructions
                WHERE ticker = $1
                ORDER BY created_at DESC
            """, ticker)
            return [dict(row) for row in rows]
    
    async def delete_instruction(self, instruction_id: int) -> bool:
        """Delete an instruction by ID."""
        await self.connect()
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM hitl_instructions WHERE id = $1
            """, instruction_id)
            return result == "DELETE 1"
    
    async def clear_all_instructions(self) -> int:
        """Clear all instructions."""
        await self.connect()
        async with self.pool.acquire() as conn:
            result = await conn.fetch("""
                DELETE FROM hitl_instructions RETURNING id
            """)
            return len(result)
    
    async def search_instructions(self, query: str) -> List[Dict]:
        """Search instructions by keyword."""
        await self.connect()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, ticker, instruction, created_at, review_id, decision, confidence, tags
                FROM hitl_instructions
                WHERE instruction ILIKE $1 OR ticker ILIKE $1
                ORDER BY created_at DESC
            """, f"%{query}%")
            return [dict(row) for row in rows]
