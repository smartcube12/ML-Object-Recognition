# choose python version
FROM python:3.12.13

# Set the working directory in Docker Container
WORKDIR /app

# Use pip to install required python packages
RUN pip install pandas numpy matplotlib


# Copy python scripts into Docker container
#COPY utils.py /app/
COPY Convo-model.py /app/  

# Create directory for output
RUN mkdir output

# command to run python script
#ENTRYPOINT ["python", "script.py"]
# provide command line args
#CMD [args]