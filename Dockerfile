FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY hawthornebench ./hawthornebench
COPY configs ./configs
COPY examples ./examples
RUN pip install --no-cache-dir ".[toml]"
# API keys come from the environment at run time: docker run -e OPENAI_API_KEY ...
VOLUME ["/app/runs"]
EXPOSE 8765
ENTRYPOINT ["hawthornebench"]
CMD ["run", "-c", "configs/quickstart.toml"]
