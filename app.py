"""Arabic RAG Assistant chat app.   Run:  streamlit run app.py"""

from __future__ import annotations

import html

import streamlit as st

from arag.config import DEFAULT_MODELS, Settings
from arag.index import Hit, Index, embedder_for
from arag.llm import LLMError, make_llm
from arag.pipeline import Answer, ask
from arag.query import QueryTranslator
from arag.rerank import make_reranker
from arag.text import is_arabic

st.set_page_config(page_title="Arabic RAG Assistant", page_icon=":material/travel_explore:", layout="wide")

EXAMPLES = [
    "How many hotel gyms were there in Doha in 2023?",
    "كم بلغت قيمة صادرات قطر إلى الصين في 2022؟",
    "How many tons of cargo were received at Doha International Airport in May 2014?",
    "كم جريدة يومية كانت تطلع في 2023؟",
    "How many chemistry laboratory tests did Hamad Medical Corporation do in 2018?",
    "كم سعر كوب الكرك في سوق واقف؟",
]
PROVIDERS = {"ollama": "Local model (Ollama)", "anthropic": "Claude API"}


@st.cache_resource(show_spinner="Loading the index ...")
def get_index(index_dir: str, retrieval: str) -> Index:
    settings = Settings.from_env().with_(retrieval=retrieval)
    return Index.load(settings.index_dir, embedder_for(settings))


def text_block(text: str) -> None:
    direction = "rtl" if is_arabic(text) else "ltr"
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(f"<div dir='{direction}' style='font-size:1.08rem;line-height:1.7'>{body}</div>", unsafe_allow_html=True)


def source_card(n: int, hit: Hit, shown: str) -> None:
    """A cited source: its titles, a link, and the rows the model saw (all rows one click away if they were cut)."""
    c = hit.chunk
    years = f" · {', '.join(c.years)}" if c.years and c.kind == "rows" else ""
    with st.expander(f"[{n}]  {c.title_en}{years}"):
        st.markdown(f"<div dir='rtl'>{html.escape(c.title_ar)}</div>", unsafe_allow_html=True)
        st.markdown(f"[Open the dataset on data.gov.qa]({c.url})")
        if shown != c.text:
            st.caption("The rows the model saw")
        st.code(shown, language=None, wrap_lines=True, height=220 if shown == c.text else "content")
        if shown != c.text:
            with st.popover("All rows in this chunk"):
                st.code(c.text, language=None, wrap_lines=True)


def render(ans: Answer) -> None:
    shown = ans.context or [h.chunk.text for h in ans.hits]
    if ans.status == "answered":
        text_block(ans.text)
        st.caption(f"{ans.seconds:.1f} s · every number above was found in its cited source")
        for n in ans.cited:
            source_card(n, ans.hits[n - 1], shown[n - 1])
    elif ans.status == "not_found":
        st.info(ans.text, icon=":material/search_off:")
    else:  # ungrounded
        st.warning(ans.text, icon=":material/gpp_maybe:")
        st.caption("Numbers the sources did not contain: " + ", ".join(ans.ungrounded_numbers))
        for n, hit in enumerate(ans.hits[:3], 1):
            source_card(n, hit, shown[n - 1])
    if ans.extra_queries:
        st.caption("Also searched: " + " · ".join(ans.extra_queries))
    with st.expander("All retrieved chunks", icon=":material/list:"):
        for n, h in enumerate(ans.hits, 1):
            st.markdown(f"**[{n}]** `{h.chunk.id}`")
            st.text(h.chunk.text[:400])


# ------------------------------------------------------------------ sidebar
settings = Settings.from_env()
with st.sidebar:
    st.markdown("### Arabic RAG Assistant")
    st.caption("مساعد الإحصاءات · Qatar's official open statistics, in Arabic or English")
    provider = st.selectbox("Model", list(PROVIDERS), format_func=PROVIDERS.get,
                            index=list(PROVIDERS).index(settings.llm_provider) if settings.llm_provider in PROVIDERS else 0)
    model = st.text_input("Model name", settings.llm_model if provider == settings.llm_provider else DEFAULT_MODELS[provider])
    retrieval = st.radio("Retrieval", ["bm25", "hybrid"], index=0 if settings.retrieval == "bm25" else 1, horizontal=True,
                         help="hybrid adds bge-m3 embeddings via Ollama (`ollama pull bge-m3`, then rebuild the index)")
    rerank_options = ["rows", "llm", "none"]
    rerank = st.radio("Rerank", rerank_options, horizontal=True,
                      index=rerank_options.index(settings.rerank) if settings.rerank in rerank_options else 0,
                      help="rows: rank chunks by their best single row (fast, default) · llm: the model picks the "
                           "chunks that hold the answer (one extra call) · none: BM25 order")
    top_k = st.slider("Sources per question", 3, 10, settings.top_k)
    translate = st.toggle("Query translation", settings.translate_query,
                          help="Also search with the question translated into the other language (one extra model call)")
    focus = st.slider("Rows shown per source", 0, 20, settings.focus_rows,
                      help="Show the model only the rows of each table that best match the question. 0 = all rows.")
    st.divider()
    st.caption("Try a question")
    pressed = [q for q in EXAMPLES if st.button(q, width="stretch")]  # draw every button, then pick
    clicked = pressed[0] if pressed else None
    st.divider()
    st.caption("Data: National Planning Council, State of Qatar, via data.gov.qa (CC BY 4.0). "
               "Check the linked dataset before relying on a figure.")

# ------------------------------------------------------------------ main
st.title("Arabic RAG Assistant")
st.caption("Ask about population, labour, health, energy, trade, transport and more. "
           "Every figure in an answer links to the dataset it came from.")

try:
    index = get_index(str(settings.index_dir), retrieval)
except FileNotFoundError as e:
    st.error(str(e), icon=":material/error:")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for q, ans in st.session_state.history:
    with st.chat_message("user"):
        text_block(q)
    with st.chat_message("assistant"):
        render(ans)

question = st.chat_input("Ask a question in Arabic or English ...") or clicked
if question:
    with st.chat_message("user"):
        text_block(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching 1,432 datasets ..."):
                llm = make_llm(settings.with_(llm_provider=provider, llm_model=model))
                ans = ask(question, index, llm, top_k, make_reranker(rerank, llm), focus,
                          QueryTranslator(llm) if translate else None)
        except LLMError as e:
            st.error(str(e), icon=":material/error:")
            st.stop()
        render(ans)
    st.session_state.history.append((question, ans))
