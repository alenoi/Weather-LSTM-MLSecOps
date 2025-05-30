# docker/api/Dockerfile

# 1) Base image with Conda
FROM continuumio/miniconda3:latest

# 2) PATH beállítása
ENV PATH=/opt/conda/bin:$PATH

# 3) Conda-környezet létrehozása
COPY environment.yml ./
RUN conda env create -f environment.yml && conda clean -afy

# 4) Automatikus környezet-aktiválás minden parancshoz
SHELL ["/opt/conda/bin/conda", "run", "-n", "mlsecops", "/bin/bash", "-lc"]

# 5) Szükséges natív könyvtárak telepítése a google-re2 buildhez
RUN apt-get update && \
    apt-get install -y \
      gcc \
      g++ \
      pkg-config \
      libre2-dev \
      cmake \
      make && \
    rm -rf /var/lib/apt/lists/*

# 6) Egyéb pip-függőségek telepítése
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN conda install -y -c conda-forge google-re2
COPY requirements.txt ./
# Első-layer: alapkönyvtárak
RUN pip install --no-cache-dir "pydantic<2.0.0"
RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn
# Második-layer: ML-stack
RUN pip install --no-cache-dir torch torchvision mlflow
# Harmadik-layer: vizualizáció + monitor
RUN pip install --no-cache-dir matplotlib seaborn streamlit evidently
# Negyedik-layer: infra
RUN pip install --no-cache-dir apache-airflow python-dotenv requests email-validator>=2.0 geopy meteostat

# 7) Forráskód másolása és munkakönyvtár beállítása
COPY src/api /app/src/api
WORKDIR /app/src/api

# 8) Port
EXPOSE 8080

# 9) Konténer indítóparancs: FastAPI szerver UVicornon
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
