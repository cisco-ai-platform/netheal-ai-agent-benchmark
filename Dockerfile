# NetHeal Green Agent - AAA Protocol Server
#
# Build:
#   docker build --platform linux/amd64 -t netheal-green-agent .
#
# Run:
#   docker run -p 9020:9020 netheal-green-agent \
#     --host 0.0.0.0 --port 9020 --card-url http://localhost:9020
#
# With LLM hints (example Azure OpenAI config):
#   docker run -p 9020:9020 \
#     -e AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/ \
#     -e AZURE_OPENAI_API_KEY=your-key \
#     -e AZURE_OPENAI_API_VERSION=2024-02-15-preview \
#     -e AZURE_OPENAI_DEPLOYMENT=gpt-5 \
#     netheal-green-agent --host 0.0.0.0 --port 9020 --card-url http://localhost:9020

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY netheal/ ./netheal/
COPY scenarios/ ./scenarios/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 9020

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9020/.well-known/agent.json')" || exit 1

ENTRYPOINT ["python", "-m", "netheal.aaa.cli"]
CMD ["--host", "0.0.0.0", "--port", "9020"]
