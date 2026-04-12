# choose python version
FROM python:3.12.13

# Set the working directory in Docker Container
WORKDIR /app

# Use pip to install required python packages
RUN pip install pandas numpy matplotlib

# Create directory for output
RUN mkdir input
RUN mkdir output

# provide command line args
#CMD [args]
CMD python graphs.py

# docker run -v ./scripts:/app <container-name>