# base env
FROM python:3.10-slim-buster

# Updating and installing awscli
RUN apt update -y && apt install awscli -y

# set the working directory
WORKDIR /app

# copy the requirements file into the image
COPY requirements.txt /app

# Install the python requirements
RUN pip install -r requirements.txt

# copy files into image
COPY models /app/models
COPY app.py /app
COPY static /app/static
COPY templates /app/templates


# expose the port 
EXPOSE 8000

# run the image
CMD [ "python", "./app.py" ]