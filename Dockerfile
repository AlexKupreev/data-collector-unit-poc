# Use the official Python 3.11 image from the Docker Hub
FROM python:3.11-slim

# Set environment variables
# not sure if needed PYTHONDONTWRITEBYTECODE
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

# Install Hatch
RUN pip install hatch

# Copy the project files
COPY . /app

# Install Python dependencies using Hatch
RUN hatch env create

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["hatch", "run", "uvicorn", "src.data_collector_unit_poc.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
