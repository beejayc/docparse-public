# docparse CLI image. Multi-arch (amd64 + arm64):
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/parser.Dockerfile -t docparse .
# LibreOffice (MPL-2.0) is installed as a separate OS package and only ever run as a child process.
FROM python:3.12-slim-bookworm AS build
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --no-editable \
    && uv pip install --python .venv/bin/python pip-licenses

# Licence gate on what is actually installed in the image (DESIGN R8).
RUN .venv/bin/pip-licenses --partial-match --format=plain \
      --fail-on="GPL;AGPL;LGPL;GNU General Public License;GNU Affero General Public License;GNU Lesser General Public License;GNU Library;Server Side Public License;SSPL" \
      --ignore-packages pip-licenses prettytable wcwidth \
      > /dev/null \
    && .venv/bin/python scripts/check_forbidden.py
RUN uv pip uninstall --python .venv/bin/python pip-licenses prettytable wcwidth

FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PATH="/app/.venv/bin:${PATH}" PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc \
        fonts-dejavu fonts-liberation fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 docparse
COPY --from=build /app /app
USER docparse
WORKDIR /home/docparse
ENV DOCPARSE_DATA_ROOT=/out
ENTRYPOINT ["docparse"]
CMD ["--help"]
