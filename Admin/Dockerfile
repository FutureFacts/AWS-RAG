# Use Python 3.12.6 as the base image
FROM python:3.12.6

# Expose port for Streamlit
EXPOSE 8083

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the host to the container
COPY . ./

# Ensure the app runs with the correct entrypoint
ENTRYPOINT ["streamlit", "run", "admin.py", "--server.port=8083", "--server.address=0.0.0.0"]


