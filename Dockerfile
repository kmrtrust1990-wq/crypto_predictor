# ক্রিপ্টো ক্যান্ডেল প্রেডিক্টর — ওয়েব ড্যাশবোর্ডের Docker ইমেজ
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

WORKDIR /app

# আগে শুধু প্যাকেজ ইনস্টল — কোড বদলালে এই ধীর ধাপটি আবার চালাতে হয় না
COPY requirements-lock.txt .
RUN pip install --root-user-action=ignore -r requirements-lock.txt

COPY . .

# root ছাড়া আলাদা ইউজার দিয়ে চালানো নিরাপদ; কোড ফাইলগুলো এই ইউজার বদলাতে পারবে না
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT', '8501'), timeout=4)"

# Render/Railway-এর মতো প্ল্যাটফর্ম PORT নিজেরা ঠিক করে দেয়
CMD ["sh", "-c", "exec streamlit run dashboard.py --server.port=${PORT} --server.address=0.0.0.0"]
