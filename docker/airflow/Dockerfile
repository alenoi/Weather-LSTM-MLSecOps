# docker/airflow/Dockerfile

FROM apache/airflow:2.8.1-python3.9

# 1. Emeljük meg a jogosultságokat natív csomagok telepítéséhez
USER root

# 2. Rendszercsomagok, ha kellene pl. matplotlib-hez (fontconfig, libgl stb.)
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        curl \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# 3. Vissza az airflow userre
USER airflow

# 4. Python csomagok – itt jön a heavy ML stuff
RUN pip install --no-cache-dir \
    matplotlib \
    seaborn \
    torch \
    torchvision \
    scikit-learn \
    mlflow \
    pandas \
    python-dotenv \
    requests \
    geopy \
    meteostat
