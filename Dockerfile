# base env
FROM python:3.10-slim

# set the working directory
WORKDIR /app/

# copy the requirements file into the image
COPY requirements.txt .

# Install the python requirements
RUN pip install -r requirements.txt

# copy files into image
COPY  ./models/ ./models/
COPY params.yaml .
COPY /flask_app/ .

# expose the port 
EXPOSE 8000

# run the image
CMD [ "python", "./app.py" ]