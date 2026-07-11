FROM python:3.11-slim

WORKDIR /app

# HF Spaces' shared cpu-basic tier can misreport AVX512 support (present in
# CPUID but not actually functional under the hypervisor), which crashes
# numpy with SIGSEGV (exit 139) the moment it dispatches to that code path -
# not a Python exception, so it takes the whole app down with no traceback.
# See https://numpy.org/doc/stable/user/troubleshooting-importerror.html.
ENV NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_KNL,AVX512_KNM"

# App-only requirements (streamlit, pandas) - not the full harness's
# torch/transformers/allosaurus, which the leaderboard never imports.
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r app/requirements.txt

# Install the pronunciation_benchmark package (scoring + leaderboard modules).
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY app/ app/
COPY data/results/benchmark_results.csv data/results/g2p_predictions.csv data/results/

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
