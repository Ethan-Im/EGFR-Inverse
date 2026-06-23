FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgl1 libglib2.0-0 \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2.0"
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
RUN pip install --no-cache-dir rdkit pandas scipy Pillow
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

RUN pip uninstall -y gradio 2>/dev/null || true

COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
