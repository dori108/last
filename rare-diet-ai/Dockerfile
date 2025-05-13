FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# 시스템 패키지 설치 (필요 시)
RUN apt-get update && apt-get install -y git && apt-get clean

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 복사
COPY . .

# 포트 설정
ENV PORT=8080

# 앱 실행
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

ENV PORT=7860
EXPOSE 7860

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
