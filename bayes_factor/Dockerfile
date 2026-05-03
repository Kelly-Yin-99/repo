FROM python:3.11-slim

WORKDIR /app

RUN pip install scipy

COPY . .

CMD ["python", "-m", "unittest", "tests/test_bayes_factor.py"]