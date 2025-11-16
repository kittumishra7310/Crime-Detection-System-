# 🚀 START HERE - Crime Detection System

## ✅ All Issues Fixed

Your system is now configured with **real authentication** and **real-time data**. No more dummy data!

## Quick Start (3 Steps)

### 1️⃣ Start the System
```bash
./start.sh
```

This will:
- Start backend on port 8000
- Start frontend on port 3000
- Open automatically in your browser

### 2️⃣ Register Your Account
1. Open http://localhost:3000
2. Click **"Create one here"**
3. Fill in the form:
   - Username: (your choice)
   - Email: (your email)
   - Password: (min 6 characters)
   - Role: **Select "Admin"** for full access
4. Click "Create Account"

### 3️⃣ Login & Use
- Login with your credentials
- Add cameras in Admin panel
- Start live detection
- Upload files for analysis

## What Was Fixed

### ❌ Before (Problems)
- Dummy login data (admin/admin123)
- Mock user in useAuth
- Fake camera data
- No real registration
- Frontend errors

### ✅ After (Fixed)
- Real user registration
- JWT authentication
- Real-time API data
- No default users
- All errors fixed

## Features Now Working

✅ **User Registration** - Create real accounts  
✅ **User Login** - JWT token auth  
✅ **Camera Management** - Add/edit/delete cameras  
✅ **Live Detection** - Real-time from webcam  
✅ **File Upload** - Analyze images/videos  
✅ **Alerts** - Real-time security alerts  
✅ **History** - View past detections  
✅ **Analytics** - Real statistics  
✅ **Admin Panel** - Manage users & cameras  

## Database Setup

The system uses MySQL. Make sure:
1. MySQL is running
2. Database credentials are correct in `Backend/config.py`
3. Database will be created automatically

Current config:
```python
DATABASE_URL = "mysql+pymysql://root:Kittu@123@localhost/surveillance_db"
```

## Troubleshooting

### Backend won't start?
```bash
cd Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### Frontend won't start?
```bash
npm install
npm run dev
```

### Can't register?
- Check backend is running on port 8000
- Check browser console for errors
- Ensure MySQL is running

### Database error?
```sql
-- Create database manually
CREATE DATABASE surveillance_db;
```

## File Structure

```
.
├── start.sh                    # One-command startup
├── test_setup.sh              # Verify setup
├── REAL_AUTH_UPDATE.md        # Detailed changes
├── QUICKSTART.md              # Quick guide
├── SETUP.md                   # Full setup guide
│
├── Backend/
│   ├── main.py                # Backend entry point
│   ├── config.py              # Configuration
│   ├── database.py            # Database models
│   ├── auth.py                # Authentication
│   └── requirements.txt       # Python dependencies
│
├── app/
│   ├── login/                 # Login page
│   ├── register/              # Registration page
│   └── dashboard/             # Main dashboard
│
├── hooks/
│   └── useAuth.ts             # Auth hook (FIXED)
│
└── services/
    └── api.ts                 # API service (FIXED)
```

## Next Steps

1. ✅ Run `./start.sh`
2. ✅ Register your admin account
3. ✅ Add cameras to the system
4. ✅ Start live detection
5. ✅ Test file upload
6. ✅ Monitor alerts

## Support

If you encounter issues:
1. Check `REAL_AUTH_UPDATE.md` for detailed info
2. Check backend logs in terminal
3. Check browser console for frontend errors
4. Ensure MySQL is running
5. Verify ports 3000 and 8000 are available

## API Documentation

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Security

✅ Passwords are hashed with bcrypt  
✅ JWT tokens for authentication  
✅ Role-based access control  
✅ 30-minute token expiration  
✅ Protected API endpoints  

---

**Ready to go!** Run `./start.sh` and register your account! 🎉
