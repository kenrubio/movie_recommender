FROM python:3.9
LABEL maintainer="Ken Rubio"
LABEL authors="Ken Rubio"

RUN python --version
RUN pip3 --version

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "main.py"]