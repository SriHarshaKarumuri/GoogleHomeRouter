"""FastAPI application with Google GenAI function-calling router and real CT Trains API."""

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.genai import types
from pydantic import BaseModel


load_dotenv()

# Import tools
from tools.train_api import check_train_tickets, search_stations
from tools.food_checker import check_food_prices

# Conversation session store keyed by session id using types.Content objects
_session_histories: dict[str, list[types.Content]] = {}

# Tool registry mapping function names to implementations
FUNCTION_REGISTRY = {
    "check_train_tickets": check_train_tickets,
    "search_stations": search_stations,
    "check_food_prices": check_food_prices,
}

# Gemini function declarations
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="check_train_tickets",
        description=(
            "Search for trains between two Indian railway stations on a given date using the CT Trains API. "
            "This covers ALL Indian Railways trains (all classes, all routes). "
            "First call 'search_stations' to get station codes if you only have city/station names. "
            "Then call this function with the station codes and date in DD-MM-YYYY format. "
            "If user mentions a specific train number or name (e.g., '20677', 'Vande Bharat'), pass it via train_number param. "
            "If user asks for a specific class (SL=Sleeper, CC=AC Chair Car, EC=Executive Chair, 3A=3rd AC, 2A=2nd AC, 1A=1st AC), pass it via travel_class param. "
            "Returns detailed availability, fares in Rs, departure/arrival times, duration, and running days."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "origin": types.Schema(type=types.Type.STRING, description="Origin station name, city, or code"),
                "destination": types.Schema(type=types.Type.STRING, description="Destination station name, city, or code"),
                "date": types.Schema(type=types.Type.STRING, description="Travel date in DD-MM-YYYY format"),
                "train_number": types.Schema(
                    type=types.Type.STRING,
                    description="Optional: train number or name to filter results (e.g. '20677' or 'vande bharat')",
                ),
                "travel_class": types.Schema(
                    type=types.Type.STRING,
                    description="Optional: travel class code — SL, 2S, CC, 3E, 3A, 2A, 1A, EC",
                ),
            },
            required=["origin", "destination", "date"],
        ),
    ),
    types.FunctionDeclaration(
        name="search_stations",
        description=(
            "Search for Indian railway stations by name, city, or station code using the CT Trains API. "
            "Returns station codes (e.g., CHENNAI -> MAS, MUMBAI -> CSTM/LTT/BDTS) and station names. "
            "Use this FIRST before calling check_train_tickets whenever the user provides station or city names instead of codes."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "search_string": types.Schema(
                    type=types.Type.STRING, description="Station name, city name, or partial code to search"
                ),
            },
            required=["search_string"],
        ),
    ),
    types.FunctionDeclaration(
        name="check_food_prices",
        description="Check food prices for a specific dish across restaurants",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "dish_name": types.Schema(type=types.Type.STRING, description="Name of the dish to search for"),
                "restaurant_name": types.Schema(
                    type=types.Type.STRING, description="Optional restaurant name to filter results"
                ),
            },
            required=["dish_name"],
        ),
    ),
]

TOOLS = [types.Tool(function_declarations=TOOL_DECLARATIONS)]

_SYSTEM_PROMPT = """You are a train enquiry assistant with access to CT Trains API for Indian Railways.

RULES:
1. Always call search_stations for origin AND destination BEFORE calling check_train_tickets.
2. Use dates in DD-MM-YYYY format; default to tomorrow.
3. Keep replies brief (user hears via TTS).
4. For food prices use check_food_prices.
5. Never refuse train or food queries."""


class DialogflowPayload(BaseModel):
    """Standard Dialogflow webhook payload with optional session tracking."""
    queryResult: dict[str, Any]
    sessionId: str = "default-session"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for initialization."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable is required")

    from google import genai
    app.state.client = genai.Client(api_key=api_key)
    yield


app = FastAPI(
    title="Voice Router API",
    description="Agentic backend for Google Assistant / Dialogflow webhook requests",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/voice-router")
async def voice_router(payload: DialogflowPayload):
    """Handle Dialogflow webhook requests using Gemini function calling with conversation context."""
    query_text = payload.queryResult.get("queryText", "").strip()
    session_id = payload.sessionId or "default-session"

    if not query_text:
        raise HTTPException(status_code=400, detail="queryText is required")

    client = app.state.client
    config = types.GenerateContentConfig(
        tools=TOOLS,
        system_instruction=[types.Part(text=_SYSTEM_PROMPT)]
    )
    history = _session_histories.setdefault(session_id, [])
    history.append(types.Content(parts=[types.Part(text=query_text)], role="user"))

    def _is_quota_error(err):
        s = str(err)
        return "429" in s or "RESOURCE_EXHAUSTED" in s or "UNAVAILABLE" in s or "NOT_FOUND" in s

    try:
        response = client.models.generate_content(model="gemini-3.5-flash", contents=history, config=config)
    except Exception as e:
        if _is_quota_error(e):
            return await _direct_tool_response(query_text, session_id)
        raise HTTPException(status_code=500, detail=f"GenAI request failed: {str(e)}")

    function_calls = list(response.function_calls) if response.function_calls else []
    raw_fc_parts: list[types.Part] = []
    if not function_calls and response.candidates:
        for c in response.candidates:
            for p in c.content.parts or []:
                if p.function_call:
                    function_calls.append(p.function_call)
                    raw_fc_parts.append(p)
    elif function_calls and response.candidates:
        seen = set()
        for c in response.candidates:
            for p in c.content.parts or []:
                if p.function_call and id(p.function_call) not in seen:
                    seen.add(id(p.function_call))
                    raw_fc_parts.append(p)

    if function_calls:
        if not raw_fc_parts and response.candidates:
            raw_fc_parts = [p for p in (response.candidates[0].content.parts or []) if p.function_call]
        history.append(types.Content(parts=raw_fc_parts or [types.Part(function_call=fc) for fc in function_calls], role="model"))

        for fc in function_calls:
            fn_name = fc.name
            args = fc.args or {}
            func = FUNCTION_REGISTRY.get(fn_name)
            if func:
                try:
                    result = func(**args)
                except Exception as ex:
                    result = f"Error: {ex}"
                history.append(types.Content(
                    parts=[types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result}))],
                    role="user",
                ))

        # Retry Gemini after tool results — fall back to direct parsing on quota error
        try:
            response = client.models.generate_content(model="gemini-3.5-flash", contents=history, config=types.GenerateContentConfig(tools=TOOLS))
        except Exception as e:
            if _is_quota_error(e):
                return await _direct_tool_response(query_text, session_id)
            return {"fulfillmentText": f"Sorry, an error occurred processing your request: {str(e)}"}

    fulfillment_text = "".join(part.text for part in response.parts if part.text)
    if not fulfillment_text:
        fulfillment_text = "I'm sorry, I couldn't find relevant information for your request."

    history.append(types.Content(parts=[types.Part(text=fulfillment_text)], role="model"))
    _session_histories[session_id] = history[-12:] if len(history) > 12 else history
    return {"fulfillmentText": fulfillment_text}


async def _direct_tool_response(query_text: str, session_id: str) -> dict:
    """Fallback when Gemini quota is exhausted — parse query and call tools directly."""
    import re as _re
    from datetime import datetime, timedelta

    q_lower = query_text.lower()
    today = datetime.now()
    tomorrows_date = (today + timedelta(days=1)).strftime("%d-%m-%Y")
    orig, dest, date = "", "", tomorrows_date

    m_from_to = _re.search(r'(?:from|[转到])\s*([^\s,]+?)\s*(?:to|到)\s*([^\s,]+)', q_lower)
    if m_from_to:
        orig = m_from_to.group(1).strip().title()
        dest = m_from_to.group(2).strip().title()

    m_date = _re.search(r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})', q_lower)
    if m_date:
        day = m_date.group(1).zfill(2)
        months = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
                  'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
        date = f"{day}-{months.get(m_date.group(2),'01')}-{m_date.group(3)}"

    if any(k in q_lower for k in ["train", "trains", "火车", "票", "查一", "车"]):
        try:
            result = check_train_tickets(origin=orig or query_text, destination=dest, date=date, train_number="", travel_class="")
            history = _session_histories.setdefault(session_id, [])
            history.append(types.Content(parts=[types.Part(text=result)], role="model"))
            _session_histories[session_id] = history[-12:]
            return {"fulfillmentText": result}
        except Exception as ex:
            return {"fulfillmentText": f"Train service temporarily unavailable: {ex}"}

    if any(k in q_lower for k in ["food", "price", "dish", "餐", "菜", "价格"]):
        try:
            result = check_food_prices(dish_name=query_text)
            return {"fulfillmentText": result}
        except Exception as ex:
            return {"fulfillmentText": f"Food service temporarily unavailable: {ex}"}

    return {"fulfillmentText": "Gemini AI service is currently busy. Please try again later."}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
