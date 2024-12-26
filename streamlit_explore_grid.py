import streamlit as st

# Hardcoded data
CLASS_HIERARCHY = [
    {
        "id": 9,
        "name": "Electrical and Scientific Apparatus",
        "terms": [
            {"term": "Computer hardware", "likelihood": "High"},
            {"term": "Mobile phones", "likelihood": "High"},
            {"term": "Software for AI training", "likelihood": "Medium"},
            {"term": "Virtual reality headsets", "likelihood": "High"},
        ]
    },
    {
        "id": 10,
        "name": "Medical Apparatus",
        "terms": [
            {"term": "Surgical masks", "likelihood": "High"},
            {"term": "Defibrillators", "likelihood": "High"},
            {"term": "Patient monitors", "likelihood": "Medium"},
            {"term": "Insulin pumps", "likelihood": "Medium"},
        ]
    },
]

def get_likelihood_icon(likelihood: str) -> str:
    """
    Returns an HTML <span> that draws a colored circle with a symbol:
      - High   -> dark green circle + white exclamation mark
      - Medium -> light green circle + black question mark
    """
    if likelihood == "High":
        return (
            "<span style='"
            "display:inline-block; width:16px; height:16px; "
            "background-color:darkgreen; border-radius:50%; color:white; "
            "text-align:center; font-weight:bold; line-height:16px; "
            "vertical-align:middle;'>!</span>"
        )
    elif likelihood == "Medium":
        return (
            "<span style='"
            "display:inline-block; width:16px; height:16px; "
            "background-color:lightgreen; border-radius:50%; color:black; "
            "text-align:center; font-weight:bold; line-height:16px; "
            "vertical-align:middle;'>?</span>"
        )
    else:
        return ""

def main():
    st.set_page_config(layout="wide")
    st.title("Trademark Classes — Icons First")

    col_chat, col_hierarchy = st.columns([2,1], gap="medium")

    with col_chat:
        st.write("### Chat or Main Content")
        user_input = st.text_input("Chat with the system...")
        st.write(f"*User typed:* {user_input}")

    with col_hierarchy:
        st.write("### USPTO Class Hierarchy")
        st.write("Expand a class to view/select its terms (icon first).")

        if "class_selections" not in st.session_state:
            st.session_state["class_selections"] = {}
        if "term_selections" not in st.session_state:
            st.session_state["term_selections"] = {}

        for class_info in CLASS_HIERARCHY:
            class_id = class_info["id"]
            class_name = class_info["name"]
            terms = class_info["terms"]

            with st.expander(f"Class {class_id}: {class_name}", expanded=False):
                # Class-level checkbox
                is_class_selected = st.checkbox(
                    f"Select entire Class {class_id}",
                    key=f"class_select_{class_id}",
                    value=st.session_state["class_selections"].get(class_id, False),
                )
                st.session_state["class_selections"][class_id] = is_class_selected

                # Term-level checkboxes + icons
                for term_info in terms:
                    term = term_info["term"]
                    likelihood = term_info["likelihood"]
                    icon_html = get_likelihood_icon(likelihood)

                    term_key = f"term_select_{class_id}_{term}"
                    current_value = st.session_state["term_selections"].get(term_key, False)

                    # Use columns to align checkbox & label
                    row_col1, row_col2 = st.columns([0.08, 0.92])
                    with row_col1:
                        # Hide the checkbox label entirely
                        term_selected = st.checkbox(
                            label="", 
                            label_visibility="hidden",  # or "collapsed"
                            key=term_key,
                            value=current_value,
                            help=f"Term: {term}, Likelihood: {likelihood}"
                        )
                        st.session_state["term_selections"][term_key] = term_selected

                    with row_col2:
                        # Icon first, then text
                        label_html = (
                            f"<span style='vertical-align:middle; margin-right:6px;'>"
                            f"{icon_html}</span>"
                            f"<span style='vertical-align:middle;'>{term}</span>"
                        )
                        st.markdown(label_html, unsafe_allow_html=True)

    st.write("## Debug: Current Selections")
    st.write("Class Selections:", st.session_state["class_selections"])
    st.write("Term Selections:", {
        k: v for k, v in st.session_state["term_selections"].items() if v
    })

if __name__ == "__main__":
    main()
