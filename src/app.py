import io
from typing import List, Optional

import requests
from PIL import Image
import streamlit as st


WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


def search_wikimedia_images(query: str, max_results: int = 10) -> List[str]:
    """Return a list of image URLs from Wikimedia Commons for a query."""
    if not query:
        return []

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": max_results,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 1200,
        "iiurlheight": 1200,
        "gsrnamespace": 6,  # File namespace
    }

    try:
        response = requests.get(WIKIMEDIA_API, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return []

    data = response.json()
    pages = (data.get("query", {}).get("pages", {}) if isinstance(data, dict) else {})
    image_urls: List[str] = []
    for page in pages.values():
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        info = imageinfo[0]
        # Prefer thumburl if present, else url
        url = info.get("thumburl") or info.get("url")
        if url:
            image_urls.append(url)
    return image_urls


def fetch_image(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Person of Interest - Image Search", layout="wide")
    st.title("Person of Interest - Image Search")
    st.caption("Search images from Wikimedia Commons. Use sidebar to tune parameters.")

    # Sidebar hyperparameters
    with st.sidebar:
        st.header("Search Settings")
        max_results = st.slider("Max results", min_value=1, max_value=50, value=12, step=1)
        num_columns = st.slider("Grid columns", min_value=1, max_value=6, value=3, step=1)
        top_k = st.slider("Top K", min_value=1, max_value=6, value=3, step=1)
        show_captions = st.toggle("Show URLs as captions", value=False)
        fetch_full_images = st.toggle("Fetch and display full images", value=False)

    query = st.text_input("Enter your image query", placeholder="e.g., sunrise mountains", value="")

    if st.button("Search") or (query and "auto_search" in st.session_state):
        st.session_state["auto_search"] = True
        with st.spinner("Searching images..."):
            urls = search_wikimedia_images(query, max_results=max_results)

        if not urls:
            st.warning("No images found. Try refining your query.")
            return

        # Display as a grid
        cols = st.columns(num_columns)
        for idx, url in enumerate(urls):
            col = cols[idx % num_columns]
            with col:
                if fetch_full_images:
                    image = fetch_image(url)
                    if image is not None:
                        st.image(image, use_container_width=True)
                    else:
                        st.image(url, use_container_width=True)
                else:
                    st.image(url, use_container_width=True)
                if show_captions:
                    st.caption(url)


if __name__ == "__main__":
    main()


