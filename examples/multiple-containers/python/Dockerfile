# 
FROM python:3.9

WORKDIR /app

COPY . .

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 
CMD ["fastapi", "run", "app/main.py", "--port", "80"]