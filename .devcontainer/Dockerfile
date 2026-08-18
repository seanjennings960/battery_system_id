FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Keep Hatch separate from project environments while making it available to
# every user in the container.
ENV PATH="/opt/hatch/bin:${PATH}"
ENV PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=5

RUN python -m venv /opt/hatch \
    && /opt/hatch/bin/python -m pip install --no-cache-dir --upgrade pip

RUN /opt/hatch/bin/python -m pip install --no-cache-dir hatch \
    && hatch --version
