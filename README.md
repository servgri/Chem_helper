# ToxMol AI

Веб-приложение и research-пайплайн для QSAR-оценки токсичности молекул на датасете **Tox21** (MoleculeNet): предсказания по 12 NR/SR-эндпоинтам, physchem/ADMET-профиль, поиск похожих структур и rule-based ретросинтез на 1–2 стадии.

## Функционал

- Ввод **SMILES** и кнопка поиска (Predict)
- Редактор структуры (**Ketcher**) с двусторонней синхронизацией SMILES ↔ molfile
- Анимированный пайплайн: Input → Fingerprint → QSAR/ADMET → NR/SR → Results
- Вкладки результатов:
  - **Predictions** — QSAR-сводка, ADMET (physchem + Lipinski/Veber), детальные **NR** / **SR**
  - **Similar** — top-N Tanimoto по Tox21
  - **Retrosynthesis** — SMARTS-маршруты (1–2 стадии)
- Django **Admin** с логом `PredictionJob`

## Стек

| Слой | Технологии |
|------|------------|
| Frontend | React 18, Vite, Tailwind CSS, Framer Motion, Ketcher |
| Backend | Django 5, Django REST Framework, django-cors-headers, Gunicorn |
| Химия / ML | RDKit, scikit-learn, LightGBM / XGBoost, FLAML, PyTorch Lightning |
| Infra | Docker Compose (`api` :8000, `web` :5173, опционально `research` Jupyter) |

## Этапы (research → web)

1. **EDA** — Tox21, баланс NR/SR, physchem (`notebooks/research.ipynb`)
2. **Featurization** — Morgan ECFP4 (2048) + physchem; графы для GNN/Transformer
3. **Обучение** — FLAML AutoML (бустинги) + Lightning (MLP / ResNet / Transformer / GNN)
4. **Отбор топовых моделей** — по `test_roc_auc` в `results/tox21_model_metrics.csv`
5. **Сервинг** — Django API загружает артефакты из `results/models/`
6. **UI** — React: ввод → инференс → вкладки Similar / Retrosynthesis

## Топовые модели (инференс)

| Target | Winner | Artifact |
|--------|--------|----------|
| NR-AhR | FLAML:lgbm | `flaml_NR-AhR_lgbm.joblib` |
| NR-Aromatase | FLAML:lgbm | `flaml_NR-Aromatase_lgbm.joblib` |
| NR-ER | FLAML:lgbm | `flaml_NR-ER_lgbm.joblib` |
| NR-PPAR-gamma | FLAML:lrl1 | `flaml_NR-PPAR-gamma_lrl1.joblib` |
| NR-AR | FLAML:xgboost | `flaml_NR-AR_xgboost.joblib` |
| SR-ARE | FLAML:lgbm | `flaml_SR-ARE_lgbm.joblib` |
| SR-HSE | FLAML:lgbm | `flaml_SR-HSE_lgbm.joblib` |
| SR-p53 | FLAML:lgbm | `flaml_SR-p53_lgbm.joblib` |
| NR-AR-LBD | Lightning_gnn | `lightning_gnn_NR-AR-LBD.pt` |
| NR-ER-LBD | Lightning_gnn | `lightning_gnn_NR-ER-LBD.pt` |
| SR-MMP | Lightning_gnn | `lightning_gnn_SR-MMP.pt` |
| SR-ATAD5 | Lightning_resnet | `lightning_resnet_SR-ATAD5.pt` |

Если `.pt` отсутствует, API использует classical fallback (см. `backend/predictions/model_registry.py`).  
`prepare_models` при первом запуске строит индекс `data/processed/tox21_fps.npz` и дообучает недостающие classical-модели (быстрый режим по умолчанию).

**ADMET:** отдельных ADMET ML-моделей в research нет — в UI это physchem + правила Lipinski/Veber плюс сводка токсичности Tox21.

## Быстрый старт (Docker)

```powershell
# API + frontend
docker compose up --build api web
```

- UI: http://127.0.0.1:5173  
- API: http://127.0.0.1:8000  
- Admin: http://127.0.0.1:8000/admin/

Суперпользователь:

```powershell
docker compose exec api python manage.py createsuperuser
```

Research Jupyter (GPU):

```powershell
docker compose up --build research
```

## Локальный запуск

### Backend

```powershell
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

cd backend
$env:PYTHONPATH = ".."
python manage.py migrate
python manage.py prepare_models
python manage.py runserver 0.0.0.0:8000
```

FLAML AutoML вместо fast-bootstrap:

```powershell
python manage.py prepare_models --flaml --time-budget 60
```

### Frontend

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

## API

| Method | Path | Назначение |
|--------|------|------------|
| GET | `/api/health/` | health + статус моделей |
| POST | `/api/molecule/parse/` | SMILES → SVG, physchem, molblock |
| POST | `/api/molecule/from-molfile/` | molfile → SMILES + SVG |
| POST | `/api/predict/` | QSAR / ADMET / NR / SR |
| POST | `/api/similar/` | top-N Tanimoto из Tox21 |
| POST | `/api/retrosynthesis/` | SMARTS-маршруты (`max_depth` 1–2) |

Пример:

```powershell
curl -X POST http://127.0.0.1:8000/api/molecule/parse/ -H "Content-Type: application/json" -d "{\"smiles\": \"CCO\"}"
```

## Переменные окружения

См. `.env` / `.env.example`. Основные:

| Var | Meaning |
|-----|---------|
| `DJANGO_DEBUG` | CORS all origins при `1` |
| `CORS_ALLOWED_ORIGINS` | origins для production |
| `TOXMOL_MODELS_DIR` | каталог моделей |
| `TOXMOL_FP_INDEX_PATH` | `tox21_fps.npz` |
| `TOXMOL_SIMILAR_TOP_N` | число похожих (default 12) |
| `TOXMOL_RETRO_MAX_DEPTH` | глубина ретросинтеза (default 2) |
| `VITE_API_URL` | URL API для фронта |
| `TOXMOL_FORCE_CPU` | CPU для API / research |

## Структура репозитория

```
backend/          Django + DRF (molecules, predictions, similarity, retrosynthesis)
frontend/         Vite React UI
src/              research helpers (qsar_utils, nn_*, flaml_train, …)
notebooks/        research.ipynb
data/             Tox21 raw + FP index
results/models/   joblib / Lightning checkpoints
```

## Данные

Tox21 скачивается в `data/raw/tox21.csv.gz` при первом `prepare_models` / запуске ноутбука.
