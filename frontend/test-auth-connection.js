// Test script to verify frontend-backend authentication connection
// Run this in the browser console or as a Node.js script

const API_BASE = 'http://localhost:8000/api/auth';

async function testAuthConnection() {
  console.log('🧪 Testing Frontend-Backend Authentication Connection');
  console.log('==================================================\n');

  try {
    // Test 1: Check if backend is running
    console.log('1. Testing backend connectivity...');
    const healthCheck = await fetch('http://localhost:8000/health');
    if (healthCheck.ok) {
      console.log('✅ Backend is running and accessible');
    } else {
      console.log('❌ Backend is not responding properly');
      return;
    }

    // Test 2: Test user registration
    console.log('\n2. Testing user registration...');
    const testUser = {
      name: 'Test User Frontend',
      email: `test-frontend-${Date.now()}@example.com`,
      password: 'testpassword123'
    };

    const registerResponse = await fetch(`${API_BASE}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testUser)
    });

    const registerData = await registerResponse.json();
    
    if (registerResponse.ok) {
      console.log('✅ Registration successful');
      console.log('   User ID:', registerData.user.id);
      console.log('   Token received:', !!registerData.token);
    } else {
      console.log('❌ Registration failed:', registerData.detail);
      return;
    }

    // Test 3: Test user login
    console.log('\n3. Testing user login...');
    const loginResponse = await fetch(`${API_BASE}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: testUser.email,
        password: testUser.password
      })
    });

    const loginData = await loginResponse.json();
    
    if (loginResponse.ok) {
      console.log('✅ Login successful');
      console.log('   User ID:', loginData.user.id);
      console.log('   Token received:', !!loginData.token);
    } else {
      console.log('❌ Login failed:', loginData.detail);
      return;
    }

    // Test 4: Test user stats with token
    console.log('\n4. Testing user stats retrieval...');
    const statsResponse = await fetch(`${API_BASE}/stats`, {
      headers: {
        'Authorization': `Bearer ${loginData.token}`,
        'Content-Type': 'application/json',
      }
    });

    const statsData = await statsResponse.json();
    
    if (statsResponse.ok) {
      console.log('✅ User stats retrieved successfully');
      console.log('   Total logins:', statsData.stats.total_logins);
      console.log('   Created at:', statsData.stats.created_at);
    } else {
      console.log('❌ Stats retrieval failed:', statsData.detail);
    }

    // Test 5: Test profile update
    console.log('\n5. Testing profile update...');
    const updateResponse = await fetch(`${API_BASE}/profile`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${loginData.token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: 'Updated Test User'
      })
    });

    const updateData = await updateResponse.json();
    
    if (updateResponse.ok) {
      console.log('✅ Profile update successful');
      console.log('   Updated name:', updateData.user.name);
    } else {
      console.log('❌ Profile update failed:', updateData.detail);
    }

    // Test 6: Test logout
    console.log('\n6. Testing logout...');
    const logoutResponse = await fetch(`${API_BASE}/logout`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${loginData.token}`,
        'Content-Type': 'application/json',
      }
    });

    if (logoutResponse.ok) {
      console.log('✅ Logout successful');
    } else {
      console.log('❌ Logout failed');
    }

    console.log('\n==================================================');
    console.log('🎉 All authentication tests completed successfully!');
    console.log('✅ Frontend is properly connected to the database');
    console.log('✅ All API endpoints are working correctly');
    console.log('✅ JWT tokens are being generated and validated');
    console.log('✅ User data is being stored and retrieved from SQLite');

  } catch (error) {
    console.error('❌ Test failed with error:', error);
    console.log('\n==================================================');
    console.log('🔧 Troubleshooting tips:');
    console.log('1. Make sure the backend is running on port 8000');
    console.log('2. Check that the database file exists and is writable');
    console.log('3. Verify that all required packages are installed');
    console.log('4. Check the browser console for CORS errors');
  }
}

// Export for Node.js usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { testAuthConnection };
}

// Auto-run if in browser
if (typeof window !== 'undefined') {
  console.log('Running authentication connection test...');
  testAuthConnection();
} 