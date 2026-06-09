FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents.py environment.py metrics.py verification.py run_simulation.py ./

CMD ["python", "run_simulation.py"]
