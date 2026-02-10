# 💳 Прогноз дефолта по кредитной карте (UCI) — End-to-End ML + Streamlit + Docker

ML-проект для предсказания **дефолта по кредитной карте** на датасете **UCI “Default of Credit Card Clients”** с полностью воспроизводимым пайплайном:

EDA → Feature Engineering → статистическая валидация → отбор фич → Optuna tuning (**LightGBM**) → **кастомный end-to-end пайплайн (raw → processed → model)** → Streamlit app → Docker.

---

## ✨ Highlights
- ✅ Понимание датасета: какие группы признаков дают основной сигнал (**repayment status**, bills, payments, credit limit, демография)
- ✅ Feature engineering, пригодный для реального инференса:
  - `LIMIT_BAL_LOG = log1p(LIMIT_BAL)`
  - `PAY_AMT* = log1p(PAY_AMT*)`
  - `BILL_AMT* = PowerTransformer(Yeo–Johnson)` (работает с отрицательными значениями)
  - `AGE_BIN` (биннинг возраста)
  - `PAY_*` стабилизация через **clipping** (`upper=3`) — уменьшает шум от экстремальных значений, сохраняя порядок категорий
- ✅ Статистическая проверка различимости классов (p-value)
- ✅ Отбор признаков:
  - greedy experiments + L1 regularization + permutation importance
- ✅ Optuna tuning + сохранение лучших параметров и **порогов, оптимизированных под Recall и F1**
- ✅ Проверка эквивалентности пайплайнов:
  - `custom_full_pipeline(raw)` ≈ `final_pipe(processed)` (одинаковое поведение на raw и processed данных)
- ✅ Деплой:
  - Streamlit app + Docker + Docker Compose

---

## 📌 Содержание
- [Обзор проекта](#-обзор-проекта)
- [Структура репозитория](#-структура-репозитория)
- [Результаты](#-результаты)
- [Как это работает (пошагово)](#-как-это-работает-пошагово)
- [Запуск локально](#️-запуск-локально)
- [Запуск через Docker](#-запуск-через-docker)
- [Артефакты](#-артефакты)
- [Roadmap / Future work](#-roadmap--future-work)
- [Лицензия](#-лицензия)

---

## 📖 Обзор проекта

**Цель:** построить воспроизводимый end-to-end ML-пайплайн для предсказания `default (0/1)`:
raw данные → обработка/feature engineering → обучение → сохранение артефактов → инференс → UI.

**Датасет:** UCI “Default of Credit Card Clients” (Тайвань, 2005)

**Что важно найти в этом датасете**
- Признаки естественно делятся на группы:
  - **Статус погашения (`PAY_*`)** — обычно самый сильный сигнал
  - **Суммы счетов (`BILL_AMT*`)** — могут быть отрицательные значения (refund/adjustments)
  - **Платежи (`PAY_AMT*`)** — сильная правосторонняя асимметрия (много маленьких платежей и немного очень больших)
  - **Кредитный лимит (`LIMIT_BAL`)** + демография (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`)
- Практическая задача — баланс между:
  - “поймать дефолтеров” (**Recall**) и
  - “не наделать слишком много ложных тревог” (**Precision/F1**)
- Поэтому в проекте сохраняются **несколько порогов** (Recall-оптимальный и F1-оптимальный), а не фиксированный 0.5.

**Подход к моделированию**
- Baseline-модель(и) для проверки сигнала и стабильности фич
- Финальная модель: **LightGBM (LGBMClassifier)** — сильный выбор для табличных данных и нелинейных зависимостей

**Основная метрика в разработке:** ROC-AUC (стабильна при дисбалансе)  
**Финальные метрики решений:** Accuracy / Precision / Recall / F1  
**Operating point:** пороги выбираются по OOF-предсказаниям (Recall-оптимальный + F1-оптимальный)

---

## 🧱 Структура репозитория
```text
.
├── artifacts/
│   ├── model_data/
│   │   ├── models/
│   │   │   ├── full_custom_final_model.joblib
│   │   │   └── full_precustom_final_model.joblib
│   │   ├── best_params.json
│   │   ├── threshold_performance.json
│   │   └── thresholds.json
│   └── final_features.json
├── dataset/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv
│   └── split/
│       ├── raw/
│       │   ├── train_set.csv
│       │   └── test_set.csv
│       └── preprocessed/
│           ├── train_set.csv
│           └── test_set.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_optuna.ipynb
├── src/
│   ├── feat_engineering.py
│   └── predict.py
├── app.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements_dev.txt
└── README.md
```

---

## 📊 Результаты

> Результаты считаются на стратифицированном train/test split и дополнительно используются out-of-fold предсказания для выбора порогов.

### Сохранённые модели
- **`full_precustom_final_model.joblib`**  
  Обучена на *уже обработанных* данных (на инференсе ожидает processed-фичи)

- **`full_custom_final_model.joblib`**  
  Полный end-to-end пайплайн: *raw → preprocessing → model* (рекомендуется для деплоя)

### Пороговая классификация
Сохраняются два порога:
- **Recall-оптимальный порог** (по умолчанию в инференсе)
- **F1-оптимальный порог**

Зачем два?
- **Recall-порог** полезен, когда “пропустить дефолт” дорого
- **F1-порог** — более сбалансированный режим (меньше false positives)

Смотри:
- `artifacts/model_data/thresholds.json`
- `artifacts/model_data/threshold_performance.json`

---

## 🧠 Как это работает (пошагово)

### 1) EDA + Feature Engineering + Feature Selection (`notebooks/01_eda.ipynb`)

Сначала EDA, чтобы понять:
- дисбаланс классов (`default=1`)
- сдвиги распределений между default и non-default
- какие группы (`PAY_*`, `BILL_AMT*`, `PAY_AMT*`) несут основной сигнал
- насколько различимость статистически “реальна”

Далее — feature engineering:

**Основные трансформации**
- `AGE_BIN` — биннинг возраста
- `LIMIT_BAL_LOG = log1p(LIMIT_BAL)` — уменьшение skew и стабилизация масштаба
- `PAY_*` — clipping до `upper=3`
  - сохраняет порядок, но снижает влияние экстремальных/шумных значений
- `BILL_AMT*` — `PowerTransformer(method="yeo-johnson")`
  - поддерживает отрицательные значения и улучшает форму распределения
- `PAY_AMT*` — `log1p(PAY_AMT*)`
  - убирает сильную асимметрию

**Связь с таргетом**
- Для числовых: корреляционная матрица (быстрый “map” сигнала)
- Для категориальных: Cramér’s V (сила ассоциации)

**Полезность признаков**
Комбинация трёх подходов:
- greedy selection (экспериментами)
- L1 regularization (разреженный отбор)
- permutation importance (финальная проверка вклада)

**Выходы**
- raw split → `dataset/split/raw/`
- preprocessed split → `dataset/split/preprocessed/`
- метаданные фич → `artifacts/final_features.json`

---

### 2) Моделирование + Optuna (LightGBM) (`notebooks/02_optuna.ipynb`)

- Протестированы несколько моделей “из коробки”
- Выбран **LGBMClassifier**
- Сделаны sanity-checks на подозрительное переобучение/утечки
- Optuna tuning → лучший скор около **0.786**
- Сохранено:
  - лучшие гиперпараметры → `artifacts/model_data/best_params.json`

**Выбор порога (по OOF)**
- Получены out-of-fold вероятности
- Выбраны пороги:
  - под Recall
  - под F1
- Сохранено:
  - `artifacts/model_data/thresholds.json`
  - `artifacts/model_data/threshold_performance.json`

---

### 3) Full Custom Pipeline (raw → processed → model)

Ключевое отличие от “ноутбук-только” проектов:
- пайплайн умеет принимать **raw ввод пользователя** и воспроизводить те же трансформации, что и при обучении.

Реализовано в:
- `src/feat_engineering.py` — логика препроцессинга + сборка пайплайна

Проверка корректности:
- `custom_full_pipeline(raw)` vs `final_pipe(processed)`  
→ поведение в проде соответствует обучению.

---

### 4) Инференс (`src/predict.py`)

`predict.py`:
- загружает модель и пороги из `artifacts/`
- принимает raw input (dict)
- возвращает вероятность + предсказание по выбранному порогу

По умолчанию используется **Recall-порог**.

---

### 5) Streamlit app (`app.py`)

UI:
- ввод raw значений
- вызов `predict()`
- отображение вероятности и решения

---

## ▶️ Запуск локально

### Установка зависимостей
```bash
pip install -r requirements.txt
# dev зависимости (опционально)
pip install -r requirements_dev.txt
```

### Запуск Streamlit
```bash
streamlit run app.py
```

Открыть:
- http://localhost:8501

---

## 🐳 Запуск через Docker

### Build
```bash
docker build -t credit-default-streamlit .
```

### Run
```bash
docker run --rm -p 8501:8501 credit-default-streamlit
```

### Или Docker Compose
```bash
docker compose up --build
```

Открыть:
- http://localhost:8501

---

## 📦 Артефакты

Хранятся в `artifacts/model_data/`:
- `best_params.json` — лучшие гиперпараметры Optuna для LightGBM
- `thresholds.json` — пороги под Recall и F1
- `threshold_performance.json` — метрики для каждого порога
- `models/full_custom_final_model.joblib` — полный пайплайн (raw → processed → model)
- `models/full_precustom_final_model.joblib` — модель, ожидающая processed-фичи

Также:
- `artifacts/final_features.json` — финальный набор фич + метаданные

---

## 🛣 Roadmap / Future work

Сейчас в проекте **не используется явное балансирование классов** (например, **SMOTE**, random oversampling/undersampling).  
Модель обучается на исходном распределении классов, а контроль качества достигается через:
- feature engineering,
- cross-validation + OOF predictions,
- и **подбор порогов** (Recall-оптимальный / F1-оптимальный).

### Планируемые улучшения
- **Балансировка дисбаланса классов**
  - Протестировать **SMOTE** и сравнить с простыми методами (random over/under-sampling).
  - Оценить, улучшает ли это **Recall/F1** без слишком большого роста false positives.
  - Важно: балансирование применять **только внутри CV-фолдов**, чтобы избежать утечек.
- **Cost-sensitive learning**
  - Использовать `class_weight` / `scale_pos_weight` (LightGBM) как лёгкую альтернативу SMOTE.
  - Тюнить веса через Optuna и сравнить компромиссы.
- **Улучшенные стратегии порогов**
  - Оптимизировать порог под бизнес-ограничения (например, минимум Recall при лимите FPR).

---

## 📄 Лицензия
MIT — см. `LICENSE`
