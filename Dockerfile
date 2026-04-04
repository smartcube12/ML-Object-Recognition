# choose python version
FROM python:3.12.13

# Set the working directory in Docker Container
WORKDIR /app

# Use pip to install required python packages
RUN pip install pandas numpy matplotlib keras tensorflow

# command to run python script
#ENTRYPOINT ["python", "script.py"]
# provide command line args
#CMD [args]
CMD ["python", "helloWorld.py"]