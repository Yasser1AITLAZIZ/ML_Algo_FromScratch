# Set the base image for a Python project
FROM python:3.10.12

WORKDIR /python_project

COPY /src /python_project

# Install dependencies
RUN pip install -r requirements.txt

# Run the main Python file
CMD ["python", "main.py"]