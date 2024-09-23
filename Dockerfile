# base env
FROM python:3.10-slim

# Updating and installing awscli
RUN apt update -y && apt install awscli -y

# set the working directory
WORKDIR /app/

# copy the requirements file into the image
COPY requirements.txt .

# Install the python requirements
RUN pip install -r requirements.txt

# copy files into image
COPY  ./models/ ./models/
COPY app.py .
COPY data_model.py .
COPY params.yaml .

# expose the port 
EXPOSE 8000

# run the image
CMD [ "python", "./app.py" ]