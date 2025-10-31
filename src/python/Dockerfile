FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY /requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY /src/python /app/src/python

CMD ["python", "-u", "src/python/titanic_pipeline.py"]
