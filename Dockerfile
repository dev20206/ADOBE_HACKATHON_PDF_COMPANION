# Use Python 3.9 slim image for efficiency
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
# This path is correct since the Dockerfile is in the project root
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and the ML model into the container
# This copies the entire 'backend' folder into a 'backend' folder inside /app
COPY backend/ ./backend/

# Set the Python path so your application can find its modules
# The root /app should be on the path so it can find the 'backend' module
ENV PYTHONPATH=/app

# Expose the port the Flask app runs on, making it accessible
EXPOSE 5001

# This is the command that will run when the container starts.
# It starts the Flask API server, which is needed for the frontend to work.
CMD ["python", "backend/api.py"]
