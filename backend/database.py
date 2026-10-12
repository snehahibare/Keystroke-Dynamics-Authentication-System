"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
SQLite Database Layer

Stores users, extracted biometric feature samples, and references to trained models.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Default DB location in backend directory
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keystroke_dna.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a sqlite3 database connection with dictionary row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the database schema if tables do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Feature Samples Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                feature_vector TEXT NOT NULL,  -- JSON serialized list of floats
                sample_type TEXT NOT NULL CHECK(sample_type IN ('enrollment', 'verification')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        # 3. Models Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_type TEXT NOT NULL CHECK(model_type IN ('ocsvm', 'isolation_forest', 'iforest')),
                model_path TEXT NOT NULL,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_samples_used INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        # 4. Evaluation Labeled Test Samples Table (Completely separate from training samples)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_samples (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                typist_name TEXT NOT NULL,
                target_username TEXT NOT NULL,
                is_genuine BOOLEAN NOT NULL,
                feature_vector TEXT NOT NULL,  -- JSON serialized list of 8 floats
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()


def create_user(username: str, db_path: str = DB_PATH) -> int:
    """
    Creates a new user or returns existing user_id if username already exists.
    """
    cleaned_name = username.strip().lower()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (cleaned_name,))
        row = cursor.fetchone()
        if row:
            return row["user_id"]

        cursor.execute("INSERT INTO users (username) VALUES (?)", (cleaned_name,))
        conn.commit()
        return cursor.lastrowid


def get_user_by_username(username: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves user record by username."""
    cleaned_name = username.strip().lower()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, created_at FROM users WHERE username = ?", (cleaned_name,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves user record by user_id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, created_at FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def save_feature_sample(user_id: int, feature_vector: List[float], sample_type: str = "enrollment", db_path: str = DB_PATH) -> int:
    """
    Saves an extracted feature vector for a user.
    """
    json_vector = json.dumps(feature_vector)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feature_samples (user_id, feature_vector, sample_type)
            VALUES (?, ?, ?)
        """, (user_id, json_vector, sample_type))
        conn.commit()
        return cursor.lastrowid


def get_user_samples(user_id: int, sample_type: str = "enrollment", db_path: str = DB_PATH) -> List[List[float]]:
    """
    Retrieves all feature vectors stored for a user under a specific sample_type.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT feature_vector FROM feature_samples 
            WHERE user_id = ? AND sample_type = ?
            ORDER BY created_at ASC
        """, (user_id, sample_type))
        rows = cursor.fetchall()
        return [json.loads(row["feature_vector"]) for row in rows]


def save_model_reference(user_id: int, model_type: str, model_path: str, num_samples: int, db_path: str = DB_PATH) -> int:
    """
    Records a trained model artifact reference in the database.
    """
    # Normalize model_type alias
    m_type = "isolation_forest" if model_type.lower() in ("isolation_forest", "iforest") else "ocsvm"
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO models (user_id, model_type, model_path, num_samples_used)
            VALUES (?, ?, ?, ?)
        """, (user_id, m_type, model_path, num_samples))
        conn.commit()
        return cursor.lastrowid


def get_latest_model(user_id: int, model_type: str, db_path: str = DB_PATH) -> Optional[str]:
    """
    Retrieves the file path of the most recently trained model for a user.
    """
    m_type = "isolation_forest" if model_type.lower() in ("isolation_forest", "iforest") else "ocsvm"
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_path FROM models 
            WHERE user_id = ? AND model_type = ?
            ORDER BY trained_at DESC, model_id DESC
            LIMIT 1
        """, (user_id, m_type))
        row = cursor.fetchone()
        return row["model_path"] if row else None


def get_all_enrolled_users(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all enrolled users who have at least one trained model."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT u.user_id, u.username 
            FROM users u
            JOIN models m ON u.user_id = m.user_id
            ORDER BY u.user_id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]


def save_evaluation_sample(typist_name: str, target_username: str, is_genuine: bool, feature_vector: List[float], db_path: str = DB_PATH) -> int:
    """
    Saves an evaluation sample into the dedicated evaluation_samples table.
    Completely decoupled from model training data to prevent data leakage.
    """
    json_vector = json.dumps(feature_vector)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluation_samples (typist_name, target_username, is_genuine, feature_vector)
            VALUES (?, ?, ?, ?)
        """, (typist_name.strip().lower(), target_username.strip().lower(), int(is_genuine), json_vector))
        conn.commit()
        return cursor.lastrowid


def get_evaluation_samples(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all labeled evaluation samples."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT eval_id, typist_name, target_username, is_genuine, feature_vector, created_at
            FROM evaluation_samples
            ORDER BY eval_id ASC
        """)
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["is_genuine"] = bool(d["is_genuine"])
            d["feature_vector"] = json.loads(d["feature_vector"])
            result.append(d)
        return result


# Auto-initialize database tables on first import
init_db()
