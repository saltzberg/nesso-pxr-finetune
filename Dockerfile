ARG NESSO_IMAGE=local/nesso:1.0.0@sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24
FROM ${NESSO_IMAGE}

WORKDIR /opt/nesso-pxr
COPY pyproject.toml README.md /opt/nesso-pxr/
COPY src /opt/nesso-pxr/src
COPY configs /opt/nesso-pxr/configs
COPY tests /opt/nesso-pxr/tests
RUN python -m pip install ".[test]"

ENTRYPOINT ["python", "-m"]
CMD ["pytest"]
