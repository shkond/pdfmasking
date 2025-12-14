FROM mcr.microsoft.com/devcontainers/python:3.12

# 追加システムパッケージ
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係のプリインストール(オプション)
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
