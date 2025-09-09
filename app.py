"""
Grad Director AI Chatbot -> Hotel QA Agent (Assignment-Ready)

Implements a LangGraph agent with exactly ONE custom tool that queries a hotels.csv
dataset, and a Streamlit chat interface.

Features required by the assignment:
- Loads hotels.csv once per session and normalizes columns
- Exactly one tool: query_hotels (structured args; filters, thresholds, sort, limit)
- LangGraph orchestration with tool invocation
- Natural-language answers + tabular text summary
- Handles no-match cases with explicit guidance
- Clamp limit to [1, 10]
- Uses OPENAI_API_KEY from environment (.env)

Run: `streamlit run app.py`
"""

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
# These globals are kept for reference; the loader below handles aliases flexibly.
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
    """
    Load CSV and map flexible/alias headers to internal names expected by the app.
    Accepts common variants like: latitude/longitude, stars, review_score_cleanliness, etc.
    """
    import re

    df = pd.read_csv(csv_path)

    # ---- Flexible column mapping ----
    # internal name -> acceptable aliases (all compared lowercased & stripped)
    ALIASES = {
        "hotel_id": ["hotel id", "hotel_id", "id"],
        "hotel_name": ["hotel name", "hotel_name", "name", "hotel"],
        "city": ["city", "town"],
        "country": ["country"],
        "lat": ["lat", "latitude", "geo_latitude", "latitude_deg"],
        "lon": ["lon", "lng", "longitude", "geo_longitude", "longitude_deg"],
        "star_rating": ["star rating", "star_rating", "stars", "rating", "hotel_star_rating"],
        "cleanliness_base": [
            "cleanliness base", "cleanliness_base",
            "cleanliness", "review_score_cleanliness", "review_scores_cleanliness",
            "cleanliness score"
        ],
        "comfort_base": [
            "comfort base", "comfort_base",
            "comfort", "review_score_comfort", "review_scores_comfort",
            "comfort score"
        ],
        "facilities_base": [
            "facilities base", "facilities_base",
            "facilities", "review_score_facilities", "review_scores_facilities",
            "facilities score", "amenities score"
        ],
    }

    # lookup normalized csv headers -> original header
    orig_cols = list(df.columns)
    norm_to_orig = {c.strip().lower(): c for c in orig_cols}

    col_map: Dict[str, str] = {}  # original_name -> internal_name
    used_norms = set()

    def first_hit(candidates):
        # exact normalized name match
        for cand in candidates:
            key = cand.strip().lower()
            if key in norm_to_orig:
                return norm_to_orig[key]
        # loose regex match (handles e.g., "review score (cleanliness)")
        for cand in candidates:
            pat = re.compile(rf"\b{re.escape(cand)}\b", re.I)
            for norm, orig in norm_to_orig.items():
                if pat.search(norm) and norm not in used_norms:
                    return orig
        return None

    required_order = [
        "hotel_id", "hotel_name", "city", "country",
        "lat", "lon",
        "star_rating", "cleanliness_base", "comfort_base", "facilities_base",
    ]
    missing_internal = []
    for internal in required_order:
        hit = first_hit(ALIASES[internal])
        if hit:
            col_map[hit] = internal
            used_norms.add(hit.strip().lower())
        else:
            missing_internal.append(internal)

    if missing_internal:
        seen = ", ".join(orig_cols[:20])  # show first 20 headers for help
        raise ValueError(
            "Missing required columns in CSV: "
            f"{missing_internal}. \n\nDetected CSV headers include: {seen}"
        )

    # rename to internal names
    df = df.rename(columns=col_map)

    # ---- Normalize text columns & add __norm copies for matching ----
    for c in ["city", "country", "hotel_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[f"{c}__norm"] = df[c].str.lower()

    # ---- Numeric coercion ----
    for c in ["star_rating", "cleanliness_base", "comfort_base", "facilities_base", "lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Require scores to exist for comparisons
    df = df.dropna(subset=["star_rating", "cleanliness_base", "comfort_base", "facilities_base"])

    # ---- Order helpful display columns ----
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

    # Case-insensitive matches
    if eff_city and "city__norm" in df.columns:
        df = df[df["city__norm"] == eff_city.lower()]
    if eff_country and "country__norm" in df.columns:
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
st.caption("Ask about hotels by city/country, ratings, and more (LangGraph + one tool).")
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

