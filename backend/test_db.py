#!/usr/bin/env python3
"""
Test script for the database functionality
"""

from app.database import db
import hashlib

def test_database():
    print("🧪 Testing Database Functionality")
    print("=" * 50)
    
    # Test 1: Create a user
    print("\n1. Creating test user...")
    try:
        user = db.create_user("test@example.com", "Test User", "password123")
        print(f"✅ User created: {user}")
    except ValueError as e:
        print(f"⚠️ User already exists: {e}")
    
    # Test 2: Verify user login
    print("\n2. Testing user login...")
    user = db.verify_user("test@example.com", "password123")
    if user:
        print(f"✅ Login successful: {user}")
    else:
        print("❌ Login failed")
    
    # Test 3: Get user by email
    print("\n3. Getting user by email...")
    user = db.get_user_by_email("test@example.com")
    if user:
        print(f"✅ User found: {user}")
    else:
        print("❌ User not found")
    
    # Test 4: Get user stats
    print("\n4. Getting user stats...")
    if user:
        stats = db.get_user_stats(int(user["id"]))
        print(f"✅ User stats: {stats}")
    
    # Test 5: Update user profile
    print("\n5. Updating user profile...")
    if user:
        success = db.update_user_profile(int(user["id"]), "Updated Test User")
        if success:
            print("✅ Profile updated successfully")
        else:
            print("❌ Profile update failed")
    
    # Test 6: Test session management
    print("\n6. Testing session management...")
    if user:
        from datetime import datetime, timedelta
        token_hash = hashlib.sha256("test-token".encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        db.store_session(int(user["id"]), token_hash, expires_at)
        print("✅ Session stored")
        
        user_id = db.validate_session(token_hash)
        if user_id:
            print(f"✅ Session validated for user ID: {user_id}")
        else:
            print("❌ Session validation failed")
    
    print("\n" + "=" * 50)
    print("🎉 Database test completed!")

if __name__ == "__main__":
    test_database() 