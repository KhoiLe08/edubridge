"""Streamlit MVP for generating grounded EduBridge learning materials."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st


st.set_page_config(page_title="EduBridge", page_icon="📚", layout="wide")


TOPICS = {
    "Hàm số bậc nhất": "linear_functions",
    "Định lý Pythagoras": "pythagorean_theorem",
    "Xác suất cơ bản": "basic_probability",
}

DEFAULT_REQUESTS = {
    "Hàm số bậc nhất": (
        "Hãy tạo một phiếu học tập tập trung vào hệ số góc và đồ thị."
    ),
    "Định lý Pythagoras": (
        "Hãy tạo một phiếu học tập giải thích định lý Pythagoras và các bài "
        "toán tìm cạnh."
    ),
    "Xác suất cơ bản": (
        "Hãy tạo một phiếu học tập về không gian mẫu, biến cố và xác suất cơ bản."
    ),
}


@st.cache_resource(show_spinner=False)
def _load_retriever() -> Callable[..., list[dict[str, Any]]]:
    """Initialize and cache the existing RAG model and Chroma connection."""
    from rag import _get_collection, _get_model, retrieve

    # Validate the committed index before downloading/loading the ML model.
    _get_collection()
    _get_model()
    return retrieve


def _initialize_session_state() -> None:
    """Set initial widget values and result placeholders for this session."""
    default_topic = next(iter(TOPICS))
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = default_topic
    if "previous_topic" not in st.session_state:
        st.session_state.previous_topic = st.session_state.selected_topic
    if "teacher_request" not in st.session_state:
        st.session_state.teacher_request = DEFAULT_REQUESTS[
            st.session_state.selected_topic
        ]
    if "learning_pack" not in st.session_state:
        st.session_state.learning_pack = None
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []


def _clear_results() -> None:
    """Remove generated output that no longer matches the current inputs."""
    st.session_state.learning_pack = None
    st.session_state.retrieved_chunks = []


def _handle_topic_change() -> None:
    """Update the suggestion without overwriting a teacher's custom request."""
    new_topic = st.session_state.selected_topic
    previous_topic = st.session_state.get("previous_topic")
    current_request = st.session_state.get("teacher_request", "")
    previous_default = DEFAULT_REQUESTS.get(previous_topic, "")

    if not current_request.strip() or current_request == previous_default:
        st.session_state.teacher_request = DEFAULT_REQUESTS[new_topic]

    st.session_state.previous_topic = new_topic
    _clear_results()


def _friendly_error(exc: Exception) -> str:
    """Return a short user-facing error message without a traceback."""
    message = str(exc).strip() or type(exc).__name__
    if "GEMINI_API_KEY" in message:
        return (
            "Thiếu GEMINI_API_KEY. Hãy thêm khóa API vào tệp .env rồi thử lại."
        )
    if len(message) > 400:
        message = message[:397] + "..."
    return message


def _format_distance(value: Any) -> str:
    """Format a retrieval distance defensively for display."""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "Không có"


def _render_sources(chunks: list[dict[str, Any]]) -> None:
    """Render retrieved attribution directly from Chroma metadata."""
    st.markdown("## Nguồn đã sử dụng")
    st.caption(
        "Thông tin nguồn dưới đây đến trực tiếp từ metadata truy xuất, "
        "không phải do LLM tạo ra."
    )

    for position, chunk in enumerate(chunks, start=1):
        source_id = str(chunk.get("source_id") or "Không rõ nguồn")
        title = str(chunk.get("title") or "Không có tiêu đề")
        chunk_id = str(chunk.get("chunk_id") or position)
        expander_title = f"{source_id} — {title}"

        with st.expander(
            expander_title,
            expanded=False,
            key=f"source_{chunk_id}_{position}",
        ):
            left_column, right_column = st.columns(2)
            with left_column:
                st.markdown(f"**Source title:** {title}")
                st.markdown(f"**Source ID:** {source_id}")
                st.markdown(f"**Topic:** {chunk.get('topic') or 'Không có'}")
                st.markdown(f"**Author:** {chunk.get('author') or 'Không có'}")
                st.markdown(
                    f"**Publisher:** {chunk.get('publisher') or 'Không có'}"
                )
            with right_column:
                st.markdown(f"**License:** {chunk.get('license') or 'Không có'}")
                st.markdown(
                    f"**Retrieval distance:** "
                    f"{_format_distance(chunk.get('distance'))}"
                )
                st.markdown(
                    f"**Chunk index:** {chunk.get('chunk_index', 'Không có')}"
                )

                source_url = str(chunk.get("source_url") or "").strip()
                if source_url:
                    st.markdown(f"**Source URL:** [{source_url}]({source_url})")
                else:
                    st.markdown("**Source URL:** Không có")

            st.markdown("**Retrieved text:**")
            st.text(str(chunk.get("text") or ""))


def _render_transparency_section() -> None:
    """Explain the compact RAG pipeline used to produce the result."""
    st.divider()
    st.subheader("How this result was created")
    st.markdown(
        "**Open educational resources** → **Chunking** → "
        "**Multilingual embeddings** → **Chroma semantic retrieval** → "
        "**Gemini** → **Learning pack**"
    )
    st.caption(
        "Sources are taken directly from retrieved metadata rather than "
        "generated by the LLM."
    )


def main() -> None:
    """Render the EduBridge Streamlit application."""
    _initialize_session_state()

    st.title("📚 EduBridge")
    st.subheader(
        "Generate temporary learning materials from openly licensed "
        "educational resources."
    )
    st.write(
        "EduBridge retrieves relevant content from verified open educational "
        "resources and uses an LLM to generate a temporary learning sheet "
        "for teachers."
    )

    st.divider()
    topic_column, request_column = st.columns([1, 2])
    with topic_column:
        selected_topic_display = st.selectbox(
            "Chủ đề",
            options=list(TOPICS),
            key="selected_topic",
            on_change=_handle_topic_change,
        )
    with request_column:
        teacher_request = st.text_area(
            "Yêu cầu của giáo viên",
            key="teacher_request",
            height=130,
            on_change=_clear_results,
        )

    generate_clicked = st.button("Generate Learning Pack", type="primary")

    if generate_clicked:
        _clear_results()
        if not teacher_request.strip():
            st.error("Vui lòng nhập yêu cầu của giáo viên trước khi tạo tài liệu.")
        else:
            selected_topic_key = TOPICS[selected_topic_display]

            try:
                with st.spinner("Đang tìm tài liệu phù hợp..."):
                    retrieve = _load_retriever()
                    retrieved_chunks = retrieve(
                        query=teacher_request.strip(),
                        top_k=4,
                        topic=selected_topic_key,
                    )
            except Exception as exc:
                st.error(f"Không thể truy xuất tài liệu: {_friendly_error(exc)}")
            else:
                if not retrieved_chunks:
                    st.error(
                        "Không tìm thấy tài liệu phù hợp cho chủ đề đã chọn. "
                        "Vui lòng điều chỉnh yêu cầu và thử lại."
                    )
                else:
                    try:
                        with st.spinner("Đang tạo phiếu học tập..."):
                            from generator import generate_learning_pack

                            learning_pack = generate_learning_pack(
                                topic_name=selected_topic_display,
                                user_request=teacher_request.strip(),
                                retrieved_chunks=retrieved_chunks,
                            )
                    except Exception as exc:
                        st.error(
                            "Không thể tạo phiếu học tập: "
                            f"{_friendly_error(exc)}"
                        )
                    else:
                        st.session_state.learning_pack = learning_pack
                        st.session_state.retrieved_chunks = retrieved_chunks

    if st.session_state.learning_pack:
        st.divider()
        st.markdown(st.session_state.learning_pack)
        _render_sources(st.session_state.retrieved_chunks)

    _render_transparency_section()


if __name__ == "__main__":
    main()
