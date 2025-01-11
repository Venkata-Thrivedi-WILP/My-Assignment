# Start with a base image that has Python installed
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the Flask application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
