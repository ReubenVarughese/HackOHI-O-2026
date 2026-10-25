"""Prediction API for the trained restoration-duration models."""

from __future__ import annotations

import os
import math
import json
import time
from collections import defaultdict, deque
from pathlib import Path

from botocore.config import Config
import joblib
import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
CLASSIFIER = joblib.load(MODEL_DIR / "restoration-extended-classifier.joblib")
REGRESSOR = joblib.load(MODEL_DIR / "restoration-positive-duration-xgboost.joblib")
HISTORICAL_LOOKUP = __import__("json").loads((MODEL_DIR / "historical-lookup.json").read_text(encoding="utf-8"))
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-2")
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(
        connect_timeout=5,
        read_timeout=30,
        retries={"max_attempts": 2, "mode": "standard"},
    ),
)
CATEGORICAL = ["Asset_Type", "Inform_Wor", "Circuit", "Substation", "Event", "Incident_W"]
NUMERIC = [
    "has_access_issue", "has_special_equipment", "media_count", "latitude", "longitude",
    "report_month", "report_hour", "circuit_customer_count", "circuit_critical_customer_count",
    "circuit_fuse_count", "circuit_recloser_count", "circuit_switch_count",
    "circuit_transformer_count", "topology_depth", "nearest_primary_miles",
    "nearest_secondary_miles", "nearest_transformer_miles", "nearest_protective_miles",
    "circuit_damage_history", "asset_damage_history", "event_damage_history",
    "circuit_asset_damage_history", "labor_reference_install_hours",
    "labor_reference_removal_hours", "indirect_labor_multiplier",
    "access_labor_adder_hours", "estimated_labor_hours",
]


class PredictionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location_name: str = Field(default="User-supplied location", max_length=200)
    damage_assets: int = Field(default=4, ge=0, le=100)
    crews_available: int = Field(default=3, ge=1, le=100)
    access_percent: int = Field(default=78, ge=0, le=100)
    asset_type: str = "Unknown"
    circuit: str = "Unknown"
    event: str = "User scenario"
    wind_speed_kmh: float | None = Field(default=None, ge=0, le=300)
    precipitation_mm: float | None = Field(default=None, ge=0, le=500)
    matched_utility_data: bool = False
    use_historical_lookup: bool = True
    circuit_customer_count: int = Field(default=0, ge=0)
    circuit_critical_customer_count: int = Field(default=0, ge=0)
    circuit_fuse_count: int = Field(default=0, ge=0)
    circuit_recloser_count: int = Field(default=0, ge=0)
    circuit_switch_count: int = Field(default=0, ge=0)
    circuit_transformer_count: int = Field(default=0, ge=0)
    topology_depth: int = Field(default=0, ge=0)
    nearest_primary_miles: float = Field(default=1.0, ge=0)
    nearest_secondary_miles: float = Field(default=1.0, ge=0)
    nearest_transformer_miles: float = Field(default=1.0, ge=0)
    nearest_protective_miles: float = Field(default=1.0, ge=0)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    prediction: PredictionRequest


app = FastAPI(title="AEP Outage Concierge Prediction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "https://d2iw6ggtjrihno.cloudfront.net",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

RATE_LIMITS = {"/predict": (30, 60), "/chat": (10, 60)}
rate_history: dict[tuple[str, str], deque[float]] = defaultdict(deque)


@app.middleware("http")
async def limit_api_requests(request: Request, call_next):
    limit_config = RATE_LIMITS.get(request.url.path)
    if limit_config is None:
        return await call_next(request)
    client_id = request.client.host if request.client else "unknown"
    limit, window_seconds = limit_config
    now = time.monotonic()
    history = rate_history[(client_id, request.url.path)]
    while history and now - history[0] >= window_seconds:
        history.popleft()
    if len(history) >= limit:
        retry_after = max(1, round(window_seconds - (now - history[0])))
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again shortly."},
            headers={"Retry-After": str(retry_after)},
        )
    history.append(now)
    return await call_next(request)


def build_input(request: PredictionRequest) -> pd.DataFrame:
    circuit_key, circuit_history, lookup_mode = resolve_historical_context(request)
    network_record, _ = resolve_network_context(request)
    asset_history = HISTORICAL_LOOKUP["circuit_assets"].get(
        f"{circuit_key}:{request.asset_type}", {}
    )

    def supplied_or_lookup(field: str, default: int) -> int:
        supplied = getattr(request, field)
        return supplied if supplied > 0 else int(network_record.get(field.removeprefix("circuit_"), default))

    values = {
        "Asset_Type": request.asset_type, "Inform_Wor": "User-reported damage",
        "Circuit": request.circuit, "Substation": "Unknown", "Event": request.event,
        "Incident_W": "User scenario", "has_access_issue": int(request.access_percent < 60),
        "has_special_equipment": 0, "media_count": 0, "latitude": request.latitude,
        "longitude": request.longitude, "report_month": 1, "report_hour": 12,
        "circuit_customer_count": supplied_or_lookup("circuit_customer_count", 0),
        "circuit_critical_customer_count": request.circuit_critical_customer_count,
        "circuit_fuse_count": supplied_or_lookup("circuit_fuse_count", 0),
        "circuit_recloser_count": supplied_or_lookup("circuit_recloser_count", 0),
        "circuit_switch_count": supplied_or_lookup("circuit_switch_count", 0),
        "circuit_transformer_count": supplied_or_lookup("circuit_transformer_count", 0),
        "topology_depth": request.topology_depth,
        "nearest_primary_miles": request.nearest_primary_miles,
        "nearest_secondary_miles": request.nearest_secondary_miles,
        "nearest_transformer_miles": request.nearest_transformer_miles,
        "nearest_protective_miles": request.nearest_protective_miles,
        "circuit_damage_history": circuit_history.get("damage_count", 0) if request.use_historical_lookup else 0,
        "asset_damage_history": asset_history.get("damage_count", 0) if request.use_historical_lookup else 0,
        "event_damage_history": 0,
        "circuit_asset_damage_history": 0, "labor_reference_install_hours": 0.215,
        "labor_reference_removal_hours": 0.045, "indirect_labor_multiplier": 4.5,
        "access_labor_adder_hours": int(request.access_percent < 60) * 0.5,
        "estimated_labor_hours": 1.0 + request.damage_assets * 0.25 + (100 - request.access_percent) * 0.02,
    }
    return pd.DataFrame([{column: values[column] for column in CATEGORICAL + NUMERIC}])


def resolve_historical_context(request: PredictionRequest) -> tuple[str, dict, str]:
    requested_key = "".join(character for character in request.circuit if character.isdigit())
    if requested_key and requested_key in HISTORICAL_LOOKUP["circuits"]:
        return requested_key, HISTORICAL_LOOKUP["circuits"][requested_key], "circuit"
    nearest_key = None
    nearest_distance = float("inf")
    for key, record in HISTORICAL_LOOKUP["circuits"].items():
        distance = math.hypot(
            (request.latitude - record["latitude"]) * 69.0,
            (request.longitude - record["longitude"]) * 53.0,
        )
        if distance < nearest_distance:
            nearest_key, nearest_distance = key, distance
    if nearest_key is not None and nearest_distance <= 15:
        return nearest_key, HISTORICAL_LOOKUP["circuits"][nearest_key], "nearest_coordinate"
    return requested_key or "unknown", {}, "unavailable"


def resolve_network_context(request: PredictionRequest) -> tuple[dict, str]:
    circuit_key, record, lookup_mode = resolve_historical_context(request)
    if lookup_mode == "circuit":
        return record, "exact_circuit"
    if lookup_mode == "nearest_coordinate":
        return record, "nearby_regional"
    return {}, "generic"


def weather_similarity_score(request: PredictionRequest) -> tuple[int, list[str]]:
    """Estimate transferability from current, storm, and climate-proxy similarity."""
    event_name = request.event.lower()
    storm_targets = (
        ("extreme", 80.0, 18.0)
        if "extreme" in event_name
        else ("severe", 55.0, 12.0)
        if "severe" in event_name
        else ("moderate", 40.0, 6.0)
    )
    _, target_wind, target_precipitation = storm_targets
    signals: list[tuple[int, int]] = []
    matched: list[str] = []
    if request.wind_speed_kmh is not None:
        wind_similarity = max(0.0, 1.0 - abs(request.wind_speed_kmh - target_wind) / 45.0)
        signals.append((round(wind_similarity * 40), 40))
        if wind_similarity >= 0.7:
            matched.append("current wind")
    if request.precipitation_mm is not None:
        precipitation_similarity = max(
            0.0,
            1.0 - abs(request.precipitation_mm - target_precipitation) / 20.0,
        )
        signals.append((round(precipitation_similarity * 25), 25))
        if precipitation_similarity >= 0.7:
            matched.append("current precipitation")

    reference_latitude = 37.27
    reference_longitude = -79.94
    climate_distance = math.hypot(
        (request.latitude - reference_latitude) / 4.0,
        (request.longitude - reference_longitude) / 8.0,
    )
    climate_similarity = max(0.0, 1.0 - climate_distance)
    signals.append((round(climate_similarity * 35), 35))
    if climate_similarity >= 0.7:
        matched.append("long-term climate proxy")

    total_weight = sum(weight for _, weight in signals)
    return (round(sum(score for score, _ in signals) / total_weight * 100), matched)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "two-stage-restoration", "chat_model": BEDROCK_MODEL_ID}


def fallback_chat(question: str, prediction: dict[str, object]) -> str:
    risk = prediction["risk"]
    window = prediction["restoration_window_hours"]
    if "prepare" in question.lower():
        score = risk["score"]
        event = str(prediction.get("scenario", "")).lower()
        if "extreme" in event or score >= 70:
            return "Prepare for an extended outage by gathering water, food, medication, and lighting. Use generators only outdoors, secure loose items, and follow official emergency instructions."
        if "severe" in event or score >= 45:
            return "Prepare for a possible outage by gathering supplies for 24–48 hours, charging essential devices, securing outdoor items, and monitoring official safety updates."
        return "Make a simple plan by charging essential devices, keeping medication and lighting ready, and following official safety updates."
    return f"The current estimate is {risk['label'].lower()} outage risk at {risk['score']}%, with a {window['low']}–{window['high']} hour asset-repair window if an outage occurs. This is an estimate, not a confirmed outage or restoration time."


def preparation_chat(question: str, prediction: dict[str, object]) -> str | None:
    if "prepare" not in question.lower():
        return None
    risk = prediction["risk"]
    score = risk["score"]
    event = str(prediction.get("scenario", "")).lower()
    if "extreme" in event or score >= 70:
        return "Prepare for an extended outage. Gather water, food, medication, and lighting, and charge essential devices. Use generators only outdoors and away from windows, secure loose items, avoid downed lines or flooded areas, and keep an emergency communication plan."
    if "severe" in event or score >= 45:
        return "Prepare for a possible outage. Gather supplies for 24–48 hours, charge essential devices, unplug sensitive electronics, secure outdoor items, keep vehicles fueled, and monitor official safety updates."
    return "Make a simple plan. Charge essential devices, keep medication and lighting ready, and monitor official safety and utility updates."


def data_explanation_chat(question: str, prediction: dict[str, object]) -> str | None:
    lower = question.lower()
    if not any(term in lower for term in ("data", "source", "evidence", "nearby", "support")):
        return None
    sources = prediction.get("data_sources", {})
    risk = prediction["risk"]
    window = prediction["restoration_window_hours"]
    location = prediction.get("location_name", "the selected location")
    network = sources.get("utility_network", "utility network evidence was unavailable")
    history = sources.get("historical_damage", "historical damage evidence was unavailable")
    weather = sources.get("weather", "storm assumptions")
    scenario = prediction.get("scenario", "the selected scenario")
    return (
        f"For {location}, I used {network}. The historical damage lookup was {history}, "
        f"and the weather input was {weather}. I also used the {scenario.lower()} scenario, "
        f"{prediction.get('damage_assets', 0)} damaged assets, {prediction.get('crews_available', 0)} available crews, "
        f"and {prediction.get('access_percent', 0)}% access. These inputs produced a "
        f"{risk['score']}% {risk['label'].lower()} risk and a {window['low']}–{window['high']} hour "
        "asset-repair estimate. It estimates repair work, not a confirmed customer outage duration."
    )


def confidence_chat(question: str, prediction: dict[str, object]) -> str | None:
    lower = question.lower()
    if not any(term in lower for term in ("confidence", "confident", "certainty", "reliable")):
        return None
    risk = prediction["risk"]
    confidence = risk["confidence"]
    data_mode = prediction.get("data_mode", "generic_fallback")
    if data_mode == "matched":
        support = "The location has a direct utility or circuit match."
    elif data_mode == "regional_match":
        support = "The location has a nearby regional utility match rather than an exact circuit match."
    else:
        support = "The estimate uses generalized assumptions because a nearby utility match was not available."
    limitation = (
        "Confidence is also limited because field conditions, damage reports, and crew availability can change."
    )
    return (
        f"Confidence in this estimate is {confidence}%. {support} "
        f"The current estimate is {risk['score']}% {risk['label'].lower()} outage risk. {limitation} "
        "This confidence describes the estimate's supporting evidence; it does not confirm that an outage is occurring."
    )


def specialized_guidance_chat(question: str, prediction: dict[str, object]) -> str | None:
    lower = question.lower()
    risk = prediction["risk"]
    window = prediction["restoration_window_hours"]
    score = risk["score"]
    event = str(prediction.get("scenario", "selected scenario")).lower()
    wind = prediction.get("wind_speed_kmh")
    rain = prediction.get("precipitation_mm")

    if "medical" in lower or "life support" in lower or "refrigerated medication" in lower:
        return (
            "If you rely on powered medical equipment, contact your healthcare provider or emergency "
            "service before an outage occurs to make an individualized backup plan. Notify your utility "
            "if a medical-needs program is available, keep backup power charged, and never wait for "
            "the estimate if equipment power becomes unsafe."
        )
    if "generator" in lower:
        return (
            "Use a generator only outdoors, well away from doors, windows, and vents. Never connect it "
            "to a home outlet or electrical panel unless a qualified electrician installed an approved "
            "transfer switch. Keep it dry, refuel only when cool, and use battery-powered carbon-monoxide "
            "alarms. Follow the manufacturer instructions and local emergency guidance."
        )
    if "asset repair duration" in lower or "repair duration" in lower:
        return (
            f"The {window['low']}–{window['high']} hour estimate describes time for utility asset repair "
            "under this scenario. It is not a promise that every customer will be without power for that "
            "entire period, and it does not confirm that an outage is active. Field assessment, switching, "
            "travel, crew availability, and additional damage can change the result."
        )
    if "weather" in lower and ("affect" in lower or "impact" in lower or "included" in lower):
        weather = (
            f"Live conditions included {wind:.0f} km/h wind and {rain:.1f} mm of precipitation."
            if isinstance(wind, (int, float)) and isinstance(rain, (int, float))
            else "Live weather was not available, so the estimate used the selected storm assumptions."
        )
        return f"{weather} Weather can increase the expected repair effort when wind or precipitation is elevated. The current estimate is {score}% {risk['label'].lower()} risk."
    if "assumption" in lower or "missing" in lower:
        return (
            f"The scenario assumes {prediction.get('damage_assets', 0)} damaged assets, "
            f"{prediction.get('crews_available', 0)} available crews, {prediction.get('access_percent', 0)}% "
            f"access, and a {event} outlook. The estimate does not know live crew dispatch, newly reported "
            "damage, exact customer-level outage status, or changing road conditions unless those are supplied."
        )
    if "extreme" in lower and ("storm" in lower or "change" in lower):
        return (
            "An extreme storm would generally increase outage risk and extend the repair window because "
            "of stronger winds, more widespread damage, access constraints, and competing restoration work. "
            "Select Extreme storm in the scenario controls to recalculate the estimate with those conditions."
        )
    if "safe" in lower or "safety" in lower:
        return (
            "Stay away from downed lines, damaged equipment, and flooded areas. Keep phones, lighting, "
            "medication, and essential devices ready. Use generators outdoors only, and follow official "
            "utility and emergency-management instructions."
        )
    if "compare" in lower and ("columbus" in lower or "roanoke" in lower):
        return (
            "I can explain the current location's estimate, but a comparison requires running a second "
            "location through the same scenario assumptions. Compare the two risk percentages, confidence "
            "levels, evidence modes, and restoration windows rather than treating either estimate as a confirmed outage duration."
        )
    return None


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    prediction = predict(request.prediction)
    preparation_answer = preparation_chat(request.question, prediction)
    if preparation_answer:
        return {"answer": preparation_answer, "mode": "prepared-guidance", "model": BEDROCK_MODEL_ID}
    confidence_answer = confidence_chat(request.question, prediction)
    if confidence_answer:
        return {"answer": confidence_answer, "mode": "confidence-explanation", "model": BEDROCK_MODEL_ID}
    data_answer = data_explanation_chat(request.question, prediction)
    if data_answer:
        return {"answer": data_answer, "mode": "data-explanation", "model": BEDROCK_MODEL_ID}
    specialized_answer = specialized_guidance_chat(request.question, prediction)
    if specialized_answer:
        return {"answer": specialized_answer, "mode": "specialized-guidance", "model": BEDROCK_MODEL_ID}
    facts = json.dumps(
        {
            "question": request.question,
            "prediction": prediction,
        },
        ensure_ascii=True,
    )
    system = (
        "You are Sparky, the PowerSpark outage assistant. Answer calmly and directly. "
        "Use only the supplied prediction JSON; never invent outages, weather, utility facts, "
        "risk values, restoration times, or capabilities. Explain uncertainty. "
        "Do not mention data sources, regional datasets, historical records, or how the estimate "
        "was calculated unless the user explicitly asks for supporting data, evidence, or sources. "
        "When explicitly asked what data supports the estimate, name the searched location, utility network "
        "source, historical damage source, weather source, damage/crew/access assumptions, and "
        "storm outlook from the JSON. Do not replace those specific sources with a generic phrase. "
        "Say when information is estimated or unavailable. For data questions, explain the role "
        "of each supplied input rather than only listing source names. Keep the answer under 150 words."
    )
    try:
        response = bedrock_runtime.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": facts}]}],
            inferenceConfig={"maxTokens": 180, "temperature": 0.2},
        )
        answer = response["output"]["message"]["content"][0]["text"].strip()
        return {"answer": answer, "mode": "bedrock", "model": BEDROCK_MODEL_ID}
    except (BotoCoreError, ClientError) as error:
        return {
            "answer": fallback_chat(request.question, prediction),
            "mode": "fallback",
            "model": BEDROCK_MODEL_ID,
            "fallback_reason": f"Bedrock request unavailable: {error.__class__.__name__}",
        }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    features = build_input(request)
    circuit_key, circuit_history, lookup_mode = resolve_historical_context(request)
    network_record, network_mode = resolve_network_context(request)
    has_circuit_history = bool(circuit_history)
    extended_probability = float(CLASSIFIER.predict_proba(features)[0, 1])
    positive_hours = max(0.0, float(REGRESSOR.predict(features)[0]))
    expected_hours = extended_probability * positive_hours
    event_name = request.event.lower()
    storm_duration_adjustment = (
        14.0
        if "extreme" in event_name
        else 7.0
        if "severe" in event_name
        else 0.0
    )
    weather_adjustment = 0.0
    if request.wind_speed_kmh is not None and request.wind_speed_kmh >= 45:
        weather_adjustment += 3.0
    if request.precipitation_mm is not None and request.precipitation_mm >= 10:
        weather_adjustment += 2.0
    expected_hours += weather_adjustment + storm_duration_adjustment
    raw_low = max(1, round(expected_hours * 0.65))
    raw_high = max(raw_low + 2, round(expected_hours * 1.45 + 2))
    if "extreme" in event_name:
        low = min(max(raw_low, 16), 96)
        high = min(max(raw_high, low + 24), 168)
    elif "severe" in event_name:
        low = min(max(raw_low, 8), 36)
        high = min(max(raw_high, low + 16), 72)
    else:
        low = min(max(raw_low, 4), 12)
        high = min(max(raw_high, low + 8), 36)
    has_local_historical_evidence = lookup_mode in {"circuit", "nearest_coordinate"}
    confidence = 85 if request.matched_utility_data or has_local_historical_evidence else 48
    similarity_score, similarity_signals = weather_similarity_score(request)
    if not request.matched_utility_data and lookup_mode == "unavailable":
        confidence += round(similarity_score * 0.1)
    confidence = max(30, confidence - round(abs(extended_probability - 0.5) * 8))
    weather_risk = 0
    if request.wind_speed_kmh is not None:
        weather_risk += 12 if request.wind_speed_kmh >= 65 else 7 if request.wind_speed_kmh >= 45 else 3 if request.wind_speed_kmh >= 30 else 0
    if request.precipitation_mm is not None and request.precipitation_mm >= 10:
        weather_risk += 5
    risk_score = min(
        96,
        round(
            18
            + extended_probability * 45
            + min(request.damage_assets, 10) * 3
            + (
                18
                if "extreme" in event_name
                else 9
                if "severe" in event_name
                else 0
            )
            + weather_risk
            + max(0, 60 - request.access_percent) * 0.15
        ),
    )
    risk_label = "High" if risk_score >= 70 else "Moderate" if risk_score >= 45 else "Low"
    utility_source = (
        "Matched utility request features"
        if request.matched_utility_data
        else "Exact circuit aggregate"
        if network_mode == "exact_circuit"
        else "Nearby regional aggregate"
        if network_mode == "nearby_regional"
        else "Generic network fallback"
    )
    historical_source = (
        "Aggregated circuit history"
        if request.use_historical_lookup and lookup_mode == "circuit"
        else "Nearest aggregated circuit history"
        if request.use_historical_lookup and lookup_mode == "nearest_coordinate"
        else "Unavailable for this location"
    )
    confirmed_facts = [
        f"Location searched: {request.location_name}.",
        f"Coordinates received: {request.latitude:.4f}, {request.longitude:.4f}.",
        f"Utility evidence source: {utility_source}.",
        f"Weather input: {'live request conditions supplied' if request.wind_speed_kmh is not None else 'no live weather supplied'}.",
    ]
    model_estimates = [
        f"{risk_label} outage risk estimate ({risk_score}%).",
        f"Asset repair restoration window: {low}–{high} hours.",
        f"Extended-repair probability: {extended_probability:.1%}.",
    ]
    assumptions = [
        f"{request.damage_assets} damaged assets assumed.",
        f"{request.crews_available} crews assumed available.",
        f"Road/access condition set to {request.access_percent}%.",
        "The target is asset repair duration, not customer outage duration.",
    ]
    customer_actions = [
        "Keep emergency lighting, medication, and device chargers ready.",
        "Follow official utility and emergency-management updates.",
    ]
    fallbacks = []
    if not request.matched_utility_data:
        fallbacks.append("No direct utility match was supplied; aggregate or generic network evidence was used.")
    if lookup_mode == "unavailable":
        fallbacks.append("No historical circuit was found within the 15-mile lookup radius.")
    if request.wind_speed_kmh is None:
        fallbacks.append("Live weather was not supplied; weather impact was not added.")
    return {
        "model_version": "two-stage-restoration-2026-10-24",
        "location_name": request.location_name,
        "damage_assets": request.damage_assets,
        "crews_available": request.crews_available,
        "access_percent": request.access_percent,
        "risk": {"score": risk_score, "label": risk_label, "confidence": confidence},
        "scenario": request.event,
        "restoration_window_hours": {"low": low, "high": high},
        "extended_repair_probability": round(extended_probability, 3),
        "model_positive_duration_hours": round(positive_hours, 2),
        "confidence": confidence,
        "data_mode": (
            "matched"
            if request.matched_utility_data or network_mode == "exact_circuit"
            else "regional_match"
            if network_mode == "nearby_regional"
            else "generic_fallback"
        ),
        "data_sources": {
            "utility_network": (
                "matched request features"
                if request.matched_utility_data
                else "exact circuit aggregate"
                if network_mode == "exact_circuit"
                else "nearby regional aggregate"
                if network_mode == "nearby_regional"
                else "generic fallback"
            ),
            "weather": "live request weather" if request.wind_speed_kmh is not None else "storm assumptions",
            "historical_damage": (
                "aggregated circuit lookup"
                if request.use_historical_lookup and lookup_mode == "circuit" and has_circuit_history
                else "nearest coordinate lookup"
                if request.use_historical_lookup and lookup_mode == "nearest_coordinate"
                else "unavailable for this location"
            ),
            "similarity": (
                f"Weather/storm/climate similarity proxy ({', '.join(similarity_signals)})"
                if similarity_signals
                else "No similarity signal available"
            ),
        },
        "weather_adjustment_hours": weather_adjustment,
        "confirmed_facts": confirmed_facts,
        "model_estimates": model_estimates,
        "customer_actions": customer_actions,
        "assumptions": assumptions,
        "fallbacks": fallbacks,
        "limitations": [
            "This predicts asset repair duration, not customer outage duration.",
            "Generic fallback features are used when utility/network data is unavailable.",
        ],
    }
