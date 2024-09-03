FROM python:3.11-buster AS builder

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv ./.venv && \
    . ./.venv/bin/activate && \
    pip install -r requirements.txt


FROM python:3.11-slim-buster AS runtime

RUN apt update && apt upgrade -y

RUN apt-get clean all

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY socratic_ui.py .

ENTRYPOINT ["python", "socratic_ui.py"]
