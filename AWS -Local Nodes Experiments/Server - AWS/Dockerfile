FROM python:3.8


RUN pip install flwr
WORKDIR /usr/app/src
COPY server.py ./
CMD ["python3","./server.py"]
