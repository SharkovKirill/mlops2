FROM python:3.9-slim

WORKDIR /app

COPY requirements.pip .

RUN pip install --no-cache-dir -r requirements.pip

COPY /src .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]