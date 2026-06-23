FROM python:3.11-bookworm

ARG UID=1000
ARG GID=1000

WORKDIR /opt/PrivTune/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g "${GID}" privtune \
    && useradd -m -u "${UID}" -s /usr/bin/bash -g privtune privtune \
    && chown -R privtune:privtune /opt/PrivTune/ \
    && chmod -R 755 /opt/PrivTune/ \
    && pip install pipx

USER privtune

WORKDIR /opt/PrivTune/

ENV PATH="/home/privtune/.local/bin:${PATH}"

RUN pipx ensurepath \
    && pipx install poetry \
    && poetry config virtualenvs.in-project true
