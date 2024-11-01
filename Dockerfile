# Set the base image for a Python project
FROM python:3.10.12

WORKDIR /python_project

# Optimazing rebuildt of image using cache
COPY src/requirements.txt /python_project

# Install dependencies
RUN pip install -r requirements.txt

COPY /src /python_project

# Run the main Python file
CMD ["python", "main.py"]