FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/nesso-pxr
COPY pyproject.toml README.md LICENSE /opt/nesso-pxr/
COPY src /opt/nesso-pxr/src
COPY configs /opt/nesso-pxr/configs
COPY scripts /opt/nesso-pxr/scripts
COPY site /opt/nesso-pxr/site
COPY data /opt/nesso-pxr/data
COPY reports /opt/nesso-pxr/reports
COPY tests /opt/nesso-pxr/tests
RUN python -m pip install --no-cache-dir ".[test]"

CMD ["python", "-m", "pytest"]
