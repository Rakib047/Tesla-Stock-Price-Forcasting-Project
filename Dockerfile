# Use Python 3.9 slim as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# e.g., pystan (used by Prophet) might need a C compiler
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Adding jupyter here as it's needed to run notebooks, but not in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt     && pip install --no-cache-dir jupyter

# Copy the rest of the application code into the container
COPY . .

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Default command to run when the container starts (can be overridden)
# This will start a Jupyter Notebook server accessible from the host.
# Users will need to copy the URL with the token from the container logs.
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--allow-root", "--no-browser"]
