# CPU software-test image.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --extra cpu --no-install-project
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
COPY LICENSE ./
RUN uv sync --locked --extra cpu
ENTRYPOINT ["python", "-m", "tefed.main"]
CMD ["--help"]
