import os

import bcrypt
import mysql.connector
from mysql.connector import IntegrityError


def get_connection():
    """Get a connection to the MySQL database."""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "pdf_summarizer"),
    )


def init_db():
    """Initialize the database and create tables. Seed default admin if needed."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(255) NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'user',
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # Seed default admin account if no admin exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = %s", ("admin",))
    admin_count = cursor.fetchone()[0]

    if admin_count == 0:
        _create_user_internal(
            conn,
            username="admin",
            email="admin@company.com",
            full_name="Administrator",
            password="admin123",
            role="admin",
            status="approved",
        )

    cursor.close()
    conn.close()


def _hash_password(password):
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password, password_hash):
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _create_user_internal(conn, username, email, full_name, password, role="user", status="pending"):
    """Internal function to create a user with an existing connection."""
    cursor = conn.cursor()
    password_hash = _hash_password(password)

    cursor.execute(
        """INSERT INTO users (username, email, full_name, password_hash, role, status)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (username, email, full_name, password_hash, role, status),
    )
    conn.commit()
    cursor.close()


def create_user(username, email, full_name, password, role="user", status="pending"):
    """Register a new user. Returns (success, message)."""
    conn = get_connection()
    try:
        _create_user_internal(conn, username, email, full_name, password, role, status)
        return True, "Account created successfully!"
    except IntegrityError as e:
        error_msg = str(e).lower()
        if "username" in error_msg:
            return False, "Username already exists."
        elif "email" in error_msg:
            return False, "Email already registered."
        else:
            return False, f"Registration failed: {str(e)}"
    finally:
        conn.close()


def authenticate_user(username, password):
    """
    Authenticate a user by username and password.
    Returns (success, user_dict_or_message).
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        return False, "Invalid username or password."

    if not _verify_password(password, row["password_hash"]):
        return False, "Invalid username or password."

    # --- APPROVAL SYSTEM (disabled for now, uncomment to re-enable) ---
    # if row["status"] == "pending":
    #     return False, "Your account is pending admin approval. Please wait."
    #
    # if row["status"] == "rejected":
    #     return False, "Your account has been rejected. Contact the administrator."

    # Successful login
    user = {
        "username": row["username"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "status": row["status"],
    }
    return True, user


def get_all_users():
    """Get all users (for admin panel). Returns list of dicts."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, username, email, full_name, role, status, created_at FROM users ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)


def get_pending_users():
    """Get users with 'pending' status."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, username, email, full_name, role, status, created_at FROM users WHERE status = %s ORDER BY created_at ASC",
        ("pending",),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)


def update_user_status(username, status):
    """Update a user's status (approved/pending/rejected)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET status = %s WHERE username = %s", (status, username))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    return affected > 0


def update_user_role(username, role):
    """Update a user's role (admin/user)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = %s WHERE username = %s", (role, username))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    return affected > 0


def delete_user(username):
    """Delete a user by username."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    return affected > 0


def username_exists(username):
    """Check if a username is already taken."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count > 0


def email_exists(email):
    """Check if an email is already registered."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count > 0
