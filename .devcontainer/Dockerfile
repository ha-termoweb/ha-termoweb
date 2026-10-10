FROM mcr.microsoft.com/devcontainers/python:3.14

# System dependencies required by Home Assistant
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libturbojpeg0 \
        libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 (for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# OpenAI Codex CLI
RUN npm i -g @openai/codex

# uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# HA config directory
RUN mkdir -p /config/custom_components \
    && chown -R vscode:vscode /config
