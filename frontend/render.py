"""Escaping helpers for the parts of the UI that emit raw HTML.

Streamlit's default rendering escapes HTML, but the score bars, badges, evidence
blocks and library rows need `unsafe_allow_html=True` for their layout. Every
value interpolated into those blocks is untrusted: titles come from PDF metadata,
quotes and rationales from the model, prior-art fields from OpenAlex.

Kept out of app.py so it can be imported without executing the Streamlit script.
"""

import html


def esc(value) -> str:
    """Escape a value for interpolation into an unsafe_allow_html block.

    Unescaped, a PDF whose metadata title is `<img src=x onerror=...>` runs
    script in every browser that opens the library.
    """
    return html.escape(str(value if value is not None else ""), quote=True)


def safe_url(url) -> str:
    """A link, or empty if it is not an ordinary web address.

    The DOI arrives from OpenAlex, so the scheme is checked rather than trusted:
    a `javascript:` value in a link would run on click.
    """
    url = str(url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else ""
