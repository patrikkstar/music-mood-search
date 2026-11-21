# Source is an on-demand batch process; keep pyarrow out of long-running API/worker images.
FROM music-mood-search-app:latest
USER root
RUN uv pip install --python /app/.venv/bin/python 'pyarrow==21.0.0'
USER appuser
