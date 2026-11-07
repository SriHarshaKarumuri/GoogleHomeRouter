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

_SYSTEM_PROMPT = """You are a helpful Indian railway enquiry assistant integrated with a live CT Trains (confirmTkt) API. This API provides real-time train schedule, seat availability, and fare information for ALL Indian Railways trains.

Your capabilities:
- Search for trains between any two Indian railway stations (e.g., Chennai to Tirupati, Delhi to Mumbai)
- Show train schedules, seat availability, fares, running days
- Find railway stations by city or station name (search by city English names like 'chennai', 'mumbai', 'delhi')
- Help users find specific trains by number or name (e.g., 'Vande Bharat', 'Shatabdi', 'Rajdhani')
- Report the availability of tickets for a particular train daily to the user until he says to stop the reminder.
- Ask for missing information politely before making calls

RULES:
1. You HAVE ACCESS TO A LIVE INDIAN RAILWAY API. ALWAYS USE IT when the user asks about train travel in India.
2. When the user gives a route like "chennai to tirupati", FIRST call search_stations for BOTH origin and destination to get station codes, THEN call check_train_tickets with those codes.
3. If the user provides a date, use it directly (convert to DD-MM-YYYY format if needed, e.g., "tomorrow" -> next day).
4. If the user mentions a train number or name (e.g., "20677", "Vande Bharat"), pass it as train_number to check_train_tickets.
5. If the user asks about a specific class (SL=Sleeper, CC=AC Chair Car, EC=Executive Chair, 3A=3rd AC, 2A=2nd AC, 1A=1st AC), pass it as travel_class.
6. If the user says "which train is fastest/cheapest/best", first fetch all trains then give your recommendation based on the returned data.
7. If the user's request is missing key details (no date, no route), ASK the user for clarification in natural language BEFORE calling any function. Do not guess dates or routes.
8. If the user asks a follow-up like "what about SL class?" after already getting train results, re-call check_train_tickets with the same origin/destination/date but add travel_class="SL" parameter.
9. Keep responses concise and conversational — the user will hear this as speech via text-to-speech.
10. NEVER refuse to help with Indian train queries. You MUST use the available tools."""


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

    # Maintain per-session conversation history
    history = _session_histories.setdefault(session_id, [])

    # Append user message (ensure it's a Content object)
    history.append(types.Content(parts=[types.Part(text=query_text)], role="user"))

    # Force tool use via system instructions if necessary
    config = types.GenerateContentConfig(
        tools=TOOLS,
        system_instruction=[types.Part(text=_SYSTEM_PROMPT)]
    )

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=history,
            config=config,
        )
    except Exception as e:
        err_str = str(e)
        # Detect API errors (quota exceeded, 503, etc.) and fall back to direct tool execution
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str:
            q_lower = query_text.lower()
            # Check for train-related intent (English or Chinese)
            if any(k in q_lower for k in ["train", "trains", "火车", "票", "查一", "车"]):
                import re as _re
                from datetime import datetime, timedelta
                today = datetime.now()
                tomorrows_date = (today + timedelta(days=1)).strftime("%d-%m-%Y")
                # Extract from/to pattern
                m_from_to = _re.search(r'(?:from|[转到])\s*([^\s,]+?)\s*(?:to|到)\s*([^\s,]+)', q_lower)
                orig, dest = "", ""
                date = tomorrows_date
                if m_from_to:
                    orig = m_from_to.group(1).strip().title()
                    dest = m_from_to.group(2).strip().title()
                # Extract date if present
                m_date = _re.search(r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})', q_lower)
                if m_date:
                    day = m_date.group(1).zfill(2)
                    months = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
                              'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
                    date = f"{day}-{months.get(m_date.group(2),'01')}-{m_date.group(3)}"
                # Try to execute with parsed params or full query as fallback
                try:
                    result_text = check_train_tickets(origin=orig or query_text, destination=dest, date=date, train_number="", travel_class="")
                    return {"fulfillmentText": result_text}
                except Exception as ex:
                    return {"fulfillmentText": f"Train service temporarily unavailable: {ex}"}
            # Check for food-related intent
            elif any(k in q_lower for k in ["food", "price", "dish", "餐", "菜", "价格"]):
                try:
                    result_text = check_food_prices(dish_name=query_text)
                    return {"fulfillmentText": result_text}
                except Exception as ex:
                    return {"fulfillmentText": f"Food service temporarily unavailable: {ex}"}
        raise HTTPException(status_code=500, detail=f"GenAI request failed: {err_str}")

    # Extract function calls — prefer resp.function_calls; fall back to resp.candidates
    function_calls = list(response.function_calls) if response.function_calls else []
    raw_fc_parts: list[types.Part] = []
    if not function_calls and response.candidates:
        for c in response.candidates:
            for p in c.content.parts or []:
                if p.function_call:
                    function_calls.append(p.function_call)
                    raw_fc_parts.append(p)
    elif function_calls:
        # Re-link function_calls to their original Parts (with thought_signature) via candidates
        if response.candidates:
            seen = set()
            for c in response.candidates:
                for p in c.content.parts or []:
                    if p.function_call and id(p.function_call) not in seen:
                        seen.add(id(p.function_call))
                        raw_fc_parts.append(p)

    if function_calls:
        # Preserve original Parts (includes thought_signature required by API)
        if not raw_fc_parts and response.candidates:
            raw_fc_parts = [p for p in (response.candidates[0].content.parts or []) if p.function_call]
        history.append(types.Content(parts=raw_fc_parts or [types.Part(function_call=fc) for fc in function_calls], role="model"))

        # Execute tools and collect results
        tool_results = []
        for fc in function_calls:
            fn_name = fc.name
            args = fc.args or {}
            func = FUNCTION_REGISTRY.get(fn_name)
            if func:
                try:
                    result = func(**args)
                    tool_results.append({"name": fn_name, "result": result})
                except Exception as e:
                    tool_results.append({"name": fn_name, "result": f"Error executing {fn_name}: {str(e)}"})
            else:
                tool_results.append({"name": fn_name, "result": f"Unknown function: {fn_name}"})

        # Append tool results to history and call Gemini again to get final response
        for tr in tool_results:
            history.append(types.Content(
                parts=[types.Part(function_response=types.FunctionResponse(
                    name=tr["name"],
                    response={"result": tr["result"]}
                ))],
                role="user",
            ))

        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=history,
                config=types.GenerateContentConfig(tools=TOOLS),
            )
        except Exception as e:
            return {"fulfillmentText": f"Sorry, an error occurred processing your request: {str(e)}"}

    # Extract text from final response
    fulfillment_text = "".join(part.text for part in response.parts if part.text)
    if not fulfillment_text:
        fulfillment_text = "I'm sorry, I couldn't find relevant information for your request."

    # Append assistant text response to history (keep last 6 round-trips for context)
    history.append(types.Content(parts=[types.Part(text=fulfillment_text)], role="model"))
    _session_histories[session_id] = history[-12:] if len(history) > 12 else history

    return {"fulfillmentText": fulfillment_text}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
