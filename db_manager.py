import os
import json
from datetime import datetime
from typing import Dict

# Safe dynamic import of PostgreSQL driver
try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False

# Safe dynamic import of MongoDB driver
try:
    import pymongo
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False

# Read connection string from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
KEYS_FILE = "api_keys.json"
FREE_DEMO_KEY = "DELIVERY-FREE-DEMO"
DAILY_LIMIT_DEFAULT = 100

def get_db_mode() -> str:
    if DATABASE_URL:
        if DATABASE_URL.startswith("mongodb+srv://") or DATABASE_URL.startswith("mongodb://"):
            return "mongodb" if HAS_MONGO else "json"
        elif "postgresql" in DATABASE_URL or "postgres" in DATABASE_URL:
            return "postgresql" if HAS_PG else "json"
    return "json"

def get_pg_connection():
    if get_db_mode() == "postgresql":
        return psycopg2.connect(DATABASE_URL)
    return None

def get_mongo_collection():
    if get_db_mode() == "mongodb":
        try:
            client = pymongo.MongoClient(DATABASE_URL)
            db = client.get_database("delivery_predictor")
            return db.get_collection("api_keys")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
    return None

def init_keys_db() -> Dict[str, dict]:
    mode = get_db_mode()
    
    if mode == "mongodb":
        col = get_mongo_collection()
        if col is not None:
            try:
                # Seed admin key if none exists
                if not col.find_one({"role": "admin"}):
                    master_key = os.environ.get("MASTER_API_KEY", "DEV-MASTER-KEY-81")
                    col.insert_one({
                        "_id": master_key,
                        "api_key": master_key,
                        "owner": "Master Administrator",
                        "created_at": datetime.now().isoformat(),
                        "active": True,
                        "role": "admin"
                    })
                # Seed free demo key if none exists
                if not col.find_one({"api_key": FREE_DEMO_KEY}):
                    col.insert_one({
                        "_id": FREE_DEMO_KEY,
                        "api_key": FREE_DEMO_KEY,
                        "owner": "Public Demo UI",
                        "created_at": datetime.now().isoformat(),
                        "active": True,
                        "role": "client",
                        "today_date": datetime.now().strftime("%Y-%m-%d"),
                        "today_requests": 0,
                        "daily_limit": 10000
                    })
            except Exception as e:
                print(f"Error seeding MongoDB: {e}")
        return load_keys_db()
        
    elif mode == "postgresql":
        conn = get_pg_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS api_keys (
                            api_key VARCHAR(255) PRIMARY KEY,
                            owner VARCHAR(255) NOT NULL,
                            created_at VARCHAR(255) NOT NULL,
                            active BOOLEAN NOT NULL DEFAULT TRUE,
                            role VARCHAR(50) NOT NULL DEFAULT 'client',
                            today_date VARCHAR(50) DEFAULT '',
                            today_requests INTEGER DEFAULT 0,
                            daily_limit INTEGER DEFAULT 100
                        );
                    """)
                    cur.execute("SELECT 1 FROM api_keys WHERE role = 'admin' LIMIT 1;")
                    if not cur.fetchone():
                        master_key = os.environ.get("MASTER_API_KEY", "DEV-MASTER-KEY-81")
                        cur.execute("""
                            INSERT INTO api_keys (api_key, owner, created_at, active, role, today_date, today_requests, daily_limit)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """, (master_key, "Master Administrator", datetime.now().isoformat(), True, "admin", "", 0, 100))
                    
                    cur.execute("SELECT 1 FROM api_keys WHERE api_key = %s LIMIT 1;", (FREE_DEMO_KEY,))
                    if not cur.fetchone():
                        cur.execute("""
                            INSERT INTO api_keys (api_key, owner, created_at, active, role, today_date, today_requests, daily_limit)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """, (FREE_DEMO_KEY, "Public Demo UI", datetime.now().isoformat(), True, "client", datetime.now().strftime("%Y-%m-%d"), 0, 10000))
                conn.commit()
            except Exception as e:
                print(f"Error seeding PostgreSQL: {e}")
            finally:
                conn.close()
            return load_keys_db()
            
    else:
        # JSON Fallback logic
        db = {}
        if os.path.exists(KEYS_FILE):
            try:
                with open(KEYS_FILE, "r") as f:
                    db = json.load(f)
            except Exception as e:
                db = {}

        if FREE_DEMO_KEY not in db:
            db[FREE_DEMO_KEY] = {
                "owner": "Public Demo UI",
                "created_at": datetime.now().isoformat(),
                "active": True,
                "role": "client",
                "today_date": datetime.now().strftime("%Y-%m-%d"),
                "today_requests": 0,
                "daily_limit": 10000
            }

        env_master_key = os.environ.get("MASTER_API_KEY")
        if env_master_key:
            db[env_master_key] = {
                "owner": "Environment Master Admin",
                "created_at": datetime.now().isoformat(),
                "active": True,
                "role": "admin"
            }
        else:
            has_admin = any(details.get("role") == "admin" and details.get("active", False) for details in db.values())
            if not has_admin:
                db["DEV-MASTER-KEY-81"] = {
                    "owner": "Default Master Admin",
                    "created_at": datetime.now().isoformat(),
                    "active": True,
                    "role": "admin"
                }

        save_keys_db_json(db)
        return db

def load_keys_db() -> Dict[str, dict]:
    mode = get_db_mode()
    
    if mode == "mongodb":
        col = get_mongo_collection()
        db = {}
        if col is not None:
            try:
                for doc in col.find():
                    key = doc.get("api_key") or doc.get("_id")
                    db[key] = {
                        "owner": doc.get("owner", "client"),
                        "created_at": doc.get("created_at", datetime.now().isoformat()),
                        "active": doc.get("active", True),
                        "role": doc.get("role", "client"),
                        "today_date": doc.get("today_date", ""),
                        "today_requests": doc.get("today_requests", 0),
                        "daily_limit": doc.get("daily_limit", 100)
                    }
            except Exception as e:
                print(f"Error loading from MongoDB: {e}")
        return db
        
    elif mode == "postgresql":
        conn = get_pg_connection()
        db = {}
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT api_key, owner, created_at, active, role, today_date, today_requests, daily_limit FROM api_keys;")
                    rows = cur.fetchall()
                    for row in rows:
                        db[row[0]] = {
                            "owner": row[1],
                            "created_at": row[2],
                            "active": row[3],
                            "role": row[4],
                            "today_date": row[5] or "",
                            "today_requests": row[6] or 0,
                            "daily_limit": row[7] or 100
                        }
            except Exception as e:
                print(f"Error loading from PostgreSQL: {e}")
            finally:
                conn.close()
        return db
        
    else:
        # JSON fallback load
        if not os.path.exists(KEYS_FILE):
            return init_keys_db()
        try:
            with open(KEYS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading keys JSON database: {e}")
            return {}

def save_keys_db(db_data: dict):
    mode = get_db_mode()
    
    if mode == "mongodb":
        col = get_mongo_collection()
        if col is not None:
            try:
                for key, details in db_data.items():
                    col.replace_one(
                        {"_id": key},
                        {
                            "api_key": key,
                            "owner": details.get("owner", "client"),
                            "created_at": details.get("created_at", datetime.now().isoformat()),
                            "active": details.get("active", True),
                            "role": details.get("role", "client"),
                            "today_date": details.get("today_date", ""),
                            "today_requests": details.get("today_requests", 0),
                            "daily_limit": details.get("daily_limit", 100)
                        },
                        upsert=True
                    )
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")
                
    elif mode == "postgresql":
        conn = get_pg_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    for key, details in db_data.items():
                        cur.execute("""
                            INSERT INTO api_keys (api_key, owner, created_at, active, role, today_date, today_requests, daily_limit)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (api_key) DO UPDATE SET
                                owner = EXCLUDED.owner,
                                created_at = EXCLUDED.created_at,
                                active = EXCLUDED.active,
                                role = EXCLUDED.role,
                                today_date = EXCLUDED.today_date,
                                today_requests = EXCLUDED.today_requests,
                                daily_limit = EXCLUDED.daily_limit;
                        """, (
                            key,
                            details.get("owner", "client"),
                            details.get("created_at", datetime.now().isoformat()),
                            details.get("active", True),
                            details.get("role", "client"),
                            details.get("today_date", ""),
                            details.get("today_requests", 0),
                            details.get("daily_limit", 100)
                        ))
                conn.commit()
            except Exception as e:
                print(f"Error saving to PostgreSQL: {e}")
            finally:
                conn.close()
    else:
        save_keys_db_json(db_data)

def save_keys_db_json(db_data: dict):
    try:
        with open(KEYS_FILE, "w") as f:
            json.dump(db_data, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON database: {e}")
