# docker/monitor/Dockerfile

# 1) Base image with Conda
FROM continuumio/miniconda3:latest
ENV PATH=/opt/conda/bin:$PATH

# 2) Conda-környezet létrehozása
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# 3) Mindig az mlsecops környezetben futtassuk a parancsokat
SHELL ["conda", "run", "-n", "mlsecops", "/bin/bash", "-lc"]

# 4) Pip-függőségek (Streamlit, Evidently, stb.)
#    Ha csak monitor fut, elég ide telepíteni
COPY requirements.txt /
RUN pip install --no-cache-dir "pydantic<2.0.0"
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir streamlit evidently

# 5) Forráskód másolása és munkakönyvtár beállítása
COPY src/monitor /app/src/monitor
WORKDIR /app/src/monitor

# 6) Port kitetes
EXPOSE 8501

# 7) Indító parancs: Streamlit app
CMD ["streamlit", "run", "monitor.py", "--server.address=0.0.0.0", "--server.port=8501"]
