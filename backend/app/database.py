import sqlite3
import hashlib
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Database:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized with users, sessions, and user_activity tables.")
    
    def create_user(self, email: str, name: str, password: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO users (email, name, password_hash)
                VALUES (?, ?, ?)
            ''', (email, name, password_hash))
            user_id = cursor.lastrowid
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (?, ?, ?)
            ''', (user_id, 'registration', f'User registered: {email}'))
            conn.commit()
            logger.info(f"User created: {email}")
            return {
                "id": str(user_id),
                "email": email,
                "name": name,
                "created_at": datetime.now().isoformat()
            }
        except sqlite3.IntegrityError:
            logger.warning(f"Email already exists: {email}")
            raise ValueError("Email already exists")
        finally:
            conn.close()
    
    def verify_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute('''
                SELECT id, email, name, created_at
                FROM users
                WHERE email = ? AND password_hash = ?
            ''', (email, password_hash))
            user = cursor.fetchone()
            if user:
                user_id, email, name, created_at = user
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user_id,))
                cursor.execute('''
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (?, 'login', 'User logged in')
                ''', (user_id,))
                conn.commit()
                logger.info(f"User verified and logged in: {email}")
                return {
                    "id": str(user_id),
                    "email": email,
                    "name": name,
                    "created_at": created_at
                }
            else:
                logger.warning(f"Login failed: {email}")
                return None
        finally:
            conn.close()
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT id, email, name, created_at, last_login
                FROM users WHERE email = ?
            ''', (email,))
            user = cursor.fetchone()
            if user:
                logger.info(f"User fetched by email: {email}")
                user_id, email, name, created_at, last_login = user
                return {
                    "id": str(user_id),
                    "email": email,
                    "name": name,
                    "created_at": created_at,
                    "last_login": last_login
                }
            return None
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT id, email, name, created_at, last_login
                FROM users WHERE id = ?
            ''', (user_id,))
            user = cursor.fetchone()
            if user:
                logger.info(f"User fetched by ID: {user_id}")
                user_id, email, name, created_at, last_login = user
                return {
                    "id": str(user_id),
                    "email": email,
                    "name": name,
                    "created_at": created_at,
                    "last_login": last_login
                }
            return None
        finally:
            conn.close()
    
    def store_session(self, user_id: int, token_hash: str, expires_at: datetime):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO sessions (user_id, token_hash, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, token_hash, expires_at))
            conn.commit()
            logger.info(f"Session stored for user ID: {user_id}")
        finally:
            conn.close()
    
    def validate_session(self, token_hash: str) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT user_id FROM sessions
                WHERE token_hash = ? AND expires_at > CURRENT_TIMESTAMP
            ''', (token_hash,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Valid session found for token hash.")
            return result[0] if result else None
        finally:
            conn.close()
    
    def delete_expired_sessions(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP
            ''')
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"{deleted} expired sessions deleted.")
        finally:
            conn.close()
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT COUNT(*) FROM user_activity
                WHERE user_id = ? AND activity_type = 'login'
            ''', (user_id,))
            total_logins = cursor.fetchone()[0]
            cursor.execute('''
                SELECT last_login FROM users WHERE id = ?
            ''', (user_id,))
            last_login = cursor.fetchone()[0]
            cursor.execute('''
                SELECT created_at FROM users WHERE id = ?
            ''', (user_id,))
            created_at = cursor.fetchone()[0]
            logger.info(f"Stats fetched for user ID: {user_id}")
            return {
                "total_logins": total_logins,
                "last_login": last_login,
                "created_at": created_at
            }
        finally:
            conn.close()
    
    def update_user_profile(self, user_id: int, name: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE users SET name = ? WHERE id = ?
            ''', (name, user_id))
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                logger.info(f"User profile updated for ID: {user_id}")
            return success
        finally:
            conn.close()

# Global instance
db = Database()
