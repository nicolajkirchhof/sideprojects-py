# Tradelog Backend

This is the backend for the Tradelog application. It is a Flask application that provides a REST API for the frontend.

## Useful Commands

### Start the application
To start the application in development mode, run the following command from the project root:
```bash
python tradelog/backend/app.py
```

### Database Migrations
This project uses Flask-Migrate to manage database migrations.

**Note:** The following commands use `set` to set environment variables on Windows. If you are using a different operating system (like Linux or macOS), use `export` instead of `set`.

**Detect changes and create a migration file:**
```bash
set FLASK_APP=tradelog.backend.app:create_app && python -m flask db migrate -m "your migration message" --directory tradelog/backend/migrations
```

**Apply the migration to the database:**
```bash
set FLASK_APP=tradelog.backend.app:create_app && python -m flask db upgrade --directory tradelog/backend/migrations
```
