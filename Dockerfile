# Use the official Python image as the base image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /usr/src/app

# Install the required packages
RUN pip install Flask faiss-cpu numpy tensorflow tensorflow_hub Pillow

RUN mkdir images && mkdir store

# Copy the rest of the application files into the container
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]
