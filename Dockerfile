FROM python:3.11-buster as builder

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv ./.venv && \
    . ./.venv/bin/activate && \
    pip install -r requirements.txt


FROM python:3.11-slim-buster as runtime

RUN apt update && apt install -y wget

RUN install -d -m 0755 /etc/apt/keyrings && \
    wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- >> /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" >> -a /etc/apt/sources.list.d/mozilla.list > /dev/null && \
     echo '\
    Package: * \
    Pin: origin packages.mozilla.org \
    Pin-Priority: 1000 \
    ' >> /etc/apt/preferences.d/mozilla

RUN apt update && apt-get install -y firefox-esr

RUN apt-get upgrade -y

RUN apt-get clean all

RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.34.0-linux64.tar.gz && \
    chmod +x geckodriver && \
    mv geckodriver /usr/local/bin/ && \
    rm geckodriver-v0.34.0-linux64.tar.gz

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY src .
COPY templates templates
COPY datasets datasets

ENTRYPOINT ["python"]
