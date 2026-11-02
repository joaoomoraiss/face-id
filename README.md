# facecheck

## backend

```
cd backend
docker compose up -d
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

coloca algum valor no JWT_SECRET_KEY do .env senao n sobe

```
.venv\Scripts\python -m uvicorn app.main:app --reload
```

## front

```
cd frontend
npm install
npm run dev
```

abre http://127.0.0.1:5173
