from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import hashlib
from app.database import db

router = APIRouter(prefix="/api/auth")

# === Auth Configuration ===
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# === Models ===
class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class TokenResponse(BaseModel):
    token: str
    user: dict

class UpdateProfile(BaseModel):
    name: str

# === Token Functions ===
def create_access_token(email: str):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    email = verify_token(credentials.credentials)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    user = db.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    
    return user

# === Routes ===
@router.post("/register", response_model=TokenResponse)
def register(data: UserRegister):
    try:
        user = db.create_user(data.email, data.name, data.password)
        token = create_access_token(user["email"])
        return {"token": token, "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=TokenResponse)
def login(data: UserLogin):
    user = db.verify_user(data.email, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(user["email"])
    return {"token": token, "user": user}

@router.post("/logout")
def logout(_: dict = Depends(get_current_user)):
    return {"message": "Logout handled client-side"}

@router.get("/stats")
def get_stats(user: dict = Depends(get_current_user)):
    stats = db.get_user_stats(int(user["id"]))
    return {"stats": stats}

@router.put("/profile")
def update_profile(update: UpdateProfile, user: dict = Depends(get_current_user)):
    success = db.update_user_profile(int(user["id"]), update.name)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update profile")
    
    return {"message": "Profile updated successfully"}
