# PowerSpark

PowerSpark is an AI-powered outage guidance application that helps customers understand outage risk, estimated restoration timing, weather conditions, and safety steps.

## Live Website

**[Open PowerSpark](https://d2iw6ggtjrihno.cloudfront.net)**

## What It Does

PowerSpark combines:

- Historical utility and outage data
- Geographic utility-data matching
- Current weather conditions
- Storm and damage assumptions
- Machine learning restoration estimates
- Amazon Bedrock conversational AI

Users enter a location and outage scenario, then receive an estimated restoration window, outage-risk score, confidence level, and practical guidance from Sparky.

## Main Features

- Interactive React frontend with responsive desktop and mobile layouts
- Location search with utility-data matching
- Weather-aware outage estimates
- Two-stage XGBoost restoration model
- Risk, confidence, evidence, restoration, weather, preparation, and alert cards
- Conversational Sparky assistant with contextual avatar images
- Safety and preparation recommendations
- API rate limiting and duplicate-request protection
- AWS-hosted production deployment

## Technology Stack

### Frontend

- React 18
- Vite
- JavaScript
- CSS
- Leaflet and supporting geospatial packages

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- XGBoost
- joblib

### AI and Cloud

- Amazon Bedrock with Amazon Nova Lite
- Amazon S3
- Amazon CloudFront
- AWS App Runner
- Amazon ECR

## How It Works

1. The user selects a city or enters a location.
2. PowerSpark loads current weather conditions for that location.
3. The backend combines location, utility, weather, storm, damage, crew, and access information.
4. The restoration models estimate outage risk and an asset-repair window.
5. The frontend presents the estimate with confidence and preparation guidance.
6. Sparky answers follow-up questions in plain language.

Source details are not included in normal Sparky responses unless the user asks for supporting data or evidence. The application still provides evidence details in the appropriate estimate view and when requested in chat.

## Run Locally

### Prerequisites

- Node.js and npm
- Python 3.11
- AWS credentials configured if testing Amazon Bedrock locally

### Install frontend dependencies

```bash
npm install
```

### Start the frontend

```bash
npm run dev
```

The Vite development server will print the local frontend URL.

### Install API dependencies

```bash
python -m pip install -r requirements-api.txt
```

### Start the API

```bash
npm run api
```

The API runs locally at `http://127.0.0.1:8000`.

Useful endpoints:

- `GET /health`
- `POST /predict`
- `POST /chat`

## Build

```bash
npm run build
```

The production frontend is generated in `dist/`.

## Project Structure

```text
src/                    React application and styles
api/prediction_api.py   FastAPI prediction and chat API
model/                  Trained models and historical lookup data
Sparky *.png            Transparent Sparky avatar assets
scripts/                Data and model preparation utilities
Dockerfile              API container definition
requirements-api.txt    API runtime dependencies
```

## Deployment

The production frontend is deployed to Amazon S3 and served through CloudFront. The FastAPI service runs in AWS App Runner from an Amazon ECR container image.

The public frontend is:

https://d2iw6ggtjrihno.cloudfront.net

## Security Notes

The API includes:

- Per-client request rate limits
- `429 Too Many Requests` responses with retry guidance
- Restricted CORS origins
- Bedrock connection and read timeouts
- Limited AWS SDK retries
- Frontend duplicate-request prevention

The current rate limiter is in-process. A shared limiter such as AWS WAF, API Gateway, or Redis would be preferable for a larger production deployment.

## Disclaimer

PowerSpark provides informational estimates for demonstration purposes. Actual restoration timing depends on field conditions, damage assessment, crew availability, access, and utility operations.
