import streamlit as st
import json

level_icons = ["🟣", "🔵", "🟢", "🟡", "🟠",  "🔴", "⚫", "⚪", "🟤"]

# Page configuration
st.set_page_config(page_title="Taxonomy Viewer", layout="wide")

# Sidebar layout
uploaded_file = st.sidebar.file_uploader("", type=["json"])

st.title("🌳 Taxonomy Viewer")

# Cache the JSON loading
@st.cache_data
def load_taxonomy(file):
    return json.load(file)

# Helper function to count leaves
def count_leaves(node):
    if isinstance(node, dict):
        children = node.get("children", [])
        if not children:
            return 1
        return sum(count_leaves(child) for child in children)
    elif isinstance(node, list):
        return sum(count_leaves(item) for item in node)
    return 0

# Main rendering function
def render_node(node, total_leaves, level=0, max_level=10, min_freq=0):
    if level >= max_level:
        return

    icon = level_icons[level % len(level_icons)]

    if isinstance(node, dict):
        name = node.get("name", "Unnamed")
        info = node.get("info", {})
        children = node.get("children", [])

        leaf_count = count_leaves(node) if children else 1  # count 1 for leaf nodes
        
        percent = (leaf_count / total_leaves * 100) if total_leaves > 0 else 0

        if percent < min_freq:
            return  # Skip nodes below the percentage threshold

        percent = (leaf_count / total_leaves * 100) if total_leaves > 0 and children else 0

        label = f"{icon} {name}"
        if children:
            label += f" — {leaf_count} errors ({percent:.2f}%)"

        with st.expander(label):
            if info:
                if "description" in info and info["description"]:
                    st.write("**Description:**", info["description"])
                else:
                    st.write("**Info:**")
                    st.json(info)

            sorted_children = sorted(children, key=lambda c: count_leaves(c), reverse=True)
            for child in sorted_children:
                render_node(child, total_leaves, level + 1, max_level, min_freq)

    elif isinstance(node, list):
        for item in node:
            render_node(item, total_leaves, level, max_level, min_freq)

def get_max_depth(node, level=0):
    if isinstance(node, dict):
        children = node.get("children", [])
        if not children:
            return level + 1
        return max(get_max_depth(child, level + 1) for child in children)
    elif isinstance(node, list):
        return max(get_max_depth(item, level) for item in node)
    return level


if uploaded_file:
    try:
        taxonomy = load_taxonomy(uploaded_file)
        total_leaves = count_leaves(taxonomy)
        max_depth = get_max_depth(taxonomy)

        st.sidebar.markdown("---")  # horizontal line

        max_display_level = st.sidebar.slider("**Max depth to display**", min_value=1, max_value=max_depth, value=max_depth-1)

        min_freq = st.sidebar.number_input(
            "**Minimum frequency to display (%)**",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            format="%.1f"
        )

        st.sidebar.markdown("---")  # horizontal line

        with st.sidebar.expander("Legend:",):
            for i in range(max_depth):
                st.markdown(f"{level_icons[i % len(level_icons)]} Level {i + 1}")


        render_node(taxonomy, total_leaves, max_level=max_display_level, min_freq=min_freq)

    except Exception as e:
        st.error(f"❌ Failed to load taxonomy: {e}")