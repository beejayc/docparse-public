# LibreOffice (MPL-2.0) isolated in its own container/process; builds on amd64 and arm64.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc \
        fonts-dejavu fonts-liberation fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

# One profile directory per worker is passed at run time:
#   soffice --headless -env:UserInstallation=file:///tmp/lo_<worker> --convert-to pdf ...
RUN useradd --create-home --uid 1000 lo
USER lo
WORKDIR /work
ENTRYPOINT ["soffice", "--headless"]
