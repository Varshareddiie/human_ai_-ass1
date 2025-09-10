

import os
import json
from pathlib import Path
from typing import List, Optional, TypedDict, Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory

# -----------------------------------------------------------------------------
# ENV & PAGE CONFIG
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Hotel QA Agent",
    page_icon="🏨",
    layout="centered",
)

# -----------------------------------------------------------------------------
# DATA LOADING (ONCE PER SESSION) + UPLOAD/SAVE SUPPORT
# -----------------------------------------------------------------------------
REQUIRED_COLS = {
    "hotel id": "hotel_id",
    "hotel name": "hotel_name",
    "city": "city",
    "country": "country",
    "lat": "lat",
    "lon": "lon",
    "star rating": "star_rating",
    "cleanliness base": "cleanliness_base",
    "comfort base": "comfort_base",
    "facilities base": "facilities_base",
}

NUMERIC_COLS = ["star_rating", "cleanliness_base", "comfort_base", "facilities_base", "lat", "lon"]
TEXT_COLS = ["city", "country", "hotel_name"]  # 'hotel_id' may be numeric

@st.cache_data(show_spinner=False)
def load_and_normalize_hotels(csv_path: str) -> pd.DataFrame:
    # Load
    df = pd.read_csv(csv_path)

    # Standardize columns: lowercase, strip & map to internal names
    col_map = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in REQUIRED_COLS:
            col_map[c] = REQUIRED_COLS[key]
    df = df.rename(columns=col_map)

    # Ensure all required columns exist
    missing = [v for v in REQUIRED_COLS.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Normalize text columns and create __norm copies for matching
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[f"{c}__norm"] = df[c].str.lower()

    # Numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing key numerics to avoid weird comparisons
    df = df.dropna(subset=["star_rating", "cleanliness_base", "comfort_base", "facilities_base"])

    # Helpful display column order
    preferred = [
        "hotel_id", "hotel_name", "city", "country",
        "star_rating", "cleanliness_base", "comfort_base", "facilities_base",
        "lat", "lon"
    ]
    display_cols = [c for c in preferred if c in df.columns]
    df = df[display_cols + [c for c in df.columns if c not in display_cols]]

    return df

# -----------------------------------------------------------------------------
# STREAMLIT STATE
# -----------------------------------------------------------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    if "df_loaded" not in st.session_state:
        st.session_state.df_loaded = False
    if "hotels_df" not in st.session_state:
        st.session_state.hotels_df = None

init_state()

# -----------------------------------------------------------------------------
# TOOL: EXACTLY ONE TOOL (query_hotels)
# -----------------------------------------------------------------------------
def clamp_limit(n: Optional[int]) -> int:
    try:
        n = int(n) if n is not None else 5
    except Exception:
        n = 5
    return max(1, min(10, n))

SORTABLE_MAP = {
    "star_rating": "star_rating",
    "cleanliness": "cleanliness_base",
    "comfort": "comfort_base",
    "facilities": "facilities_base",
    # Allow aliases:
    "cleanliness_base": "cleanliness_base",
    "comfort_base": "comfort_base",
    "facilities_base": "facilities_base",
}

@tool("query_hotels", return_direct=False)
def query_hotels_tool(
    city: Optional[str] = None,
    country: Optional[str] = None,
    min_star: Optional[float] = None,
    min_cleanliness: Optional[float] = None,
    min_comfort: Optional[float] = None,
    min_facilities: Optional[float] = None,
    sort_by: Optional[str] = "star_rating",
    limit: Optional[int] = 5,
) -> str:
    """
    Query the hotels dataset with optional filters and sorting.

    Args:
        city (str, optional): case-insensitive city filter
        country (str, optional): case-insensitive country filter
        min_star (float, optional): minimum star rating
        min_cleanliness (float, optional): minimum cleanliness_base
        min_comfort (float, optional): minimum comfort_base
        min_facilities (float, optional): minimum facilities_base
        sort_by (str, optional): one of {'star_rating','cleanliness','comfort','facilities'}
        limit (int, optional): number of rows to return (clamped to [1,10])

    Returns:
        JSON string with keys:
          - "results": list of rows (dicts)
          - "count": int
          - "note": str (present when no results to guide user)
          - "args_used": dict of effective args
    """
    if "hotels_df" not in st.session_state or st.session_state.hotels_df is None:
        return json.dumps({
            "results": [],
            "count": 0,
            "note": "Dataset not loaded. Please ensure hotels.csv is available and the app re-ran.",
            "args_used": {}
        })

    df = st.session_state.hotels_df.copy()

    # Normalize filters
    eff_city = (city or "").strip()
    eff_country = (country or "").strip()

    # Case-insensitive match by using precomputed __norm
    if eff_city:
        df = df[df["city__norm"] == eff_city.lower()]
    if eff_country:
        df = df[df["country__norm"] == eff_country.lower()]

    # Numeric thresholds
    if min_star is not None:
        df = df[df["star_rating"] >= float(min_star)]
    if min_cleanliness is not None:
        df = df[df["cleanliness_base"] >= float(min_cleanliness)]
    if min_comfort is not None:
        df = df[df["comfort_base"] >= float(min_comfort)]
    if min_facilities is not None:
        df = df[df["facilities_base"] >= float(min_facilities)]

    # Sorting
    sort_key = SORTABLE_MAP.get((sort_by or "").strip().lower(), "star_rating")
    df = df.sort_values(by=sort_key, ascending=False, kind="mergesort")

    # Clamp limit
    limit_val = clamp_limit(limit)
    df = df.head(limit_val)

    # Prepare results
    display_cols = [
        c for c in [
            "hotel_name", "city", "country",
            "star_rating", "cleanliness_base", "comfort_base", "facilities_base",
            "lat", "lon"
        ] if c in df.columns
    ]

    if df.empty:
        note = (
            "No hotels matched your query. Try relaxing filters (e.g., lower thresholds), "
            "removing city/country constraints, or choosing a different sort key."
        )
        return json.dumps({
            "results": [],
            "count": 0,
            "note": note,
            "args_used": {
                "city": eff_city or None,
                "country": eff_country or None,
                "min_star": min_star,
                "min_cleanliness": min_cleanliness,
                "min_comfort": min_comfort,
                "min_facilities": min_facilities,
                "sort_by": sort_key,
                "limit": limit_val,
            }
        })

    rows = df[display_cols].to_dict(orient="records")
    return json.dumps({
        "results": rows,
        "count": len(rows),
        "args_used": {
            "city": eff_city or None,
            "country": eff_country or None,
            "min_star": min_star,
            "min_cleanliness": min_cleanliness,
            "min_comfort": min_comfort,
            "min_facilities": min_facilities,
            "sort_by": sort_key,
            "limit": limit_val,
        }
    })

TOOLS = [query_hotels_tool]
tool_node = ToolNode(TOOLS)

# -----------------------------------------------------------------------------
# LANGGRAPH STATE & MODEL
# -----------------------------------------------------------------------------
class MessagesState(TypedDict):
    messages: List[AnyMessage]

def get_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY not found in environment (.env).")
        st.info("Create a .env file with one line:\nOPENAI_API_KEY=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")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

SYSTEM_PROMPT = (
    "You are a helpful Hotel QA agent.\n"
    "- You have exactly ONE tool: query_hotels. Use it whenever the user asks a data question.\n"
    "- Parse free-text queries into structured tool args (city, country, minimum thresholds, sort_by, limit).\n"
    "- Sort_by can be one of: star_rating, cleanliness, comfort, facilities.\n"
    "- Always produce a short natural-language summary followed by a compact table in plain text.\n"
    "- If there are no results, clearly say so and suggest alternative queries (e.g., relax thresholds or remove filters).\n"
    "- Do NOT call any other tools or browse the web.\n"
)

def call_model(state: MessagesState, config: RunnableConfig) -> dict:
    llm = get_model().bind_tools(TOOLS)
    msgs: List[AnyMessage] = state["messages"]
    prefixed = []
    if not msgs or not isinstance(msgs[0], SystemMessage):
        prefixed.append(SystemMessage(content=SYSTEM_PROMPT))
    prefixed.extend(msgs)
    resp = llm.invoke(prefixed, config=config)
    return {"messages": [resp]}

# Build graph
graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_node("tools", tool_node)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition, {"tools": "tools", "end": END})
graph.add_edge("tools", "model")
app_graph = graph.compile()

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🏨 Hotel QA Agent")
st.caption("Find your ideal stay! I can help you search for hotels by city or country, filter by star ratings, and provide all the details you need to book the perfect room.")
st.markdown("---")

# ----- Dataset controls (upload once, save to data/hotels.csv, then load once per session)
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DEFAULT_CSV_PATH = DATA_DIR / "hotels.csv"

with st.expander("📂 Dataset (upload or load)", expanded=not st.session_state.df_loaded):
    st.write(
        "Upload your `hotels.csv` here once. It will be saved to `data/hotels.csv` "
        "so future runs auto-load it."
    )

    # Upload-and-save flow
    uploaded = st.file_uploader("Upload hotels.csv", type=["csv"])
    if uploaded is not None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temp_df = pd.read_csv(uploaded)
            temp_df.to_csv(DEFAULT_CSV_PATH, index=False)
            st.success(f"✅ Saved to {DEFAULT_CSV_PATH.relative_to(HERE)}")
        except Exception as e:
            st.error(f"Failed to save uploaded CSV: {e}")

    # Allow custom path too (fallback / advanced)
    csv_path_input = st.text_input(
        "Path to hotels.csv",
        value=str(DEFAULT_CSV_PATH if DEFAULT_CSV_PATH.exists() else "data/hotels.csv"),
    )

    # Load/Reload button
    if st.button("Load/Reload Dataset"):
        try:
            st.session_state.hotels_df = load_and_normalize_hotels(csv_path_input)
            st.session_state.df_loaded = True
            st.success(f"Loaded {len(st.session_state.hotels_df)} rows.")
        except Exception as e:
            st.session_state.df_loaded = False
            st.error(f"Failed to load dataset: {e}")

# Clear chat button + example
left, right = st.columns([1, 5])
with left:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.rerun()
with right:
    st.info("Example: *Top 5 hotels in Paris by cleanliness*")

st.markdown("---")

# Show prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("What would you like to know?")
if user_msg:
    # Guard: ensure dataset is loaded
    if not st.session_state.df_loaded:
        st.error("Please load the dataset first (use the Dataset section above).")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.chat_history.add_message(HumanMessage(content=user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build the conversation for the graph
                    history_msgs: List[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            history_msgs.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant":
                            history_msgs.append(AIMessage(content=m["content"]))

                    # Invoke the graph
                    result = app_graph.invoke({"messages": history_msgs})

                    # Extract the last assistant message (after tool runs)
                    final_msgs: List[AnyMessage] = result["messages"]

                    last_ai = None
                    for msg in reversed(final_msgs):
                        if isinstance(msg, AIMessage):
                            last_ai = msg
                            break

                    if last_ai is None:
                        ai_text = "I couldn't generate a response. Please try rephrasing your query."
                        st.markdown(ai_text)
                    else:
                        ai_text = last_ai.content
                        st.markdown(ai_text)

                    # Store assistant response
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.session_state.chat_history.add_message(AIMessage(content=ai_text))

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Check your API key, dataset path (Dataset section), and internet connection (for the LLM).")










