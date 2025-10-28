# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app/backend

# Copy backend files
COPY backend/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend folder (optional if served via StaticFiles)
COPY frontend/ ../frontend/

# Expose port
EXPOSE 8080

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
