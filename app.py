"""Streamlit application for browsing and managing collected sailing logs.

The previous revision of this file only contained the inner loop responsible for
rendering each row of the inventory table. Because the surrounding control
structure was missing, Python raised an ``IndentationError``.  This module now
provides a full, well-structured Streamlit application that can be executed
directly with ``streamlit run app.py``.

The app expects two CSV files inside ``STORAGE_ROOT``:

* ``master.csv`` – the authoritative inventory of every imported log.
* ``index.csv`` – an optional secondary index that mirrors ``master.csv``.

Both CSVs are optional.  When either file is missing, the application simply
shows an empty state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DEFAULT_STORAGE_ROOT = Path("storage").resolve()
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", DEFAULT_STORAGE_ROOT)).resolve()
MASTER_CSV_NAME = os.getenv("MASTER_CSV_NAME", "master.csv")
INDEX_CSV_NAME = os.getenv("INDEX_CSV_NAME", "index.csv")

# The minimum set of columns the app expects in the inventory.
INVENTORY_COLUMNS = [
    "canonical_filename",
    "drive_path",
    "file_hash_sha256",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` restricted to :data:`INVENTORY_COLUMNS`.

    Missing columns are added with empty string values so that downstream code
    can rely on their presence without having to guard every access.
    """

    for column in INVENTORY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[INVENTORY_COLUMNS]


def load_inventory(csv_path: Path) -> pd.DataFrame:
    """Load the inventory ``csv_path`` as a :class:`~pandas.DataFrame`.

    When the CSV does not exist, an empty dataframe with the expected columns is
    returned.  Any parsing errors are surfaced to the user via Streamlit and an
    empty dataframe is returned as a safe fallback.
    """

    if not csv_path.exists():
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive UI feedback
        st.error(f"Não foi possível ler o inventário '{csv_path}': {exc}")
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    return _ensure_dataframe_columns(df)


def remove_row_csv(path: Path, predicate: Callable[[pd.Series], bool]) -> None:
    """Remove rows from ``path`` that satisfy ``predicate``.

    If the CSV does not exist, the function returns silently.  When the file is
    present, it is re-written without the rows that match the predicate.
    """

    if not path.exists():
        return

    df = pd.read_csv(path)
    mask = df.apply(predicate, axis=1)
    df = df.loc[~mask]
    df.to_csv(path, index=False)


def resolve_path(raw_path: str) -> Path:
    """Return an absolute path for ``raw_path`` relative to ``STORAGE_ROOT``."""

    raw_path = raw_path or ""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return STORAGE_ROOT / candidate


def humanize_filename(value: str) -> str:
    """Return a human-friendly representation for ``value``."""

    return value or "(sem nome)"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render_inventory(df_view: pd.DataFrame, index_csv_path: Path) -> None:
    """Render the inventory table and row-level actions."""

    if df_view.empty:
        st.info("Nenhum arquivo para este filtro.")
        return

    for i, row in df_view.iterrows():
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])

        display_name = humanize_filename(row.get("canonical_filename", ""))
        drive_path = str(row.get("drive_path", ""))
        file_hash = str(row.get("file_hash_sha256", ""))

        with col1:
            st.write(f"**{display_name}**")
            st.caption(drive_path or "(caminho não informado)")

        with col2:
            try:
                file_abs = resolve_path(drive_path)
                if file_abs.exists():
                    with file_abs.open("rb") as stream:
                        st.download_button(
                            "Baixar",
                            data=stream.read(),
                            file_name=file_abs.name,
                            key=f"dl_{i}",
                        )
                else:
                    st.button("Baixar", disabled=True, key=f"dl_{i}")
            except Exception:  # pragma: no cover - download is best-effort
                st.button("Baixar", disabled=True, key=f"dl_{i}")

        with col3:
            if st.button("Excluir", key=f"rm_{i}"):
                try:
                    file_abs = resolve_path(drive_path)
                    if file_abs.exists():
                        file_abs.unlink()

                    # Attempt to clean empty parent directories up to three
                    # levels above the deleted file.
                    parent = file_abs.parent
                    for _ in range(3):
                        if (
                            parent.is_dir()
                            and parent != parent.parent
                            and not any(parent.iterdir())
                            and str(parent).startswith(str(STORAGE_ROOT))
                        ):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break

                    def predicate(series: pd.Series) -> bool:
                        return (
                            str(series.get("canonical_filename", "")) == display_name
                            and str(series.get("file_hash_sha256", "")) == file_hash
                        )

                    master_csv_path = STORAGE_ROOT / MASTER_CSV_NAME
                    remove_row_csv(master_csv_path, predicate)
                    remove_row_csv(index_csv_path, predicate)

                    st.success("Arquivo excluído e inventário atualizado.")
                    st.experimental_rerun()
                except Exception as exc:  # pragma: no cover - defensive UI
                    st.error(f"Falha ao excluir: {exc}")

        with col4:
            st.code(file_hash[:8] if file_hash else "—", language=None)


def main() -> None:
    """Entry point for ``streamlit run app.py``."""

    st.set_page_config(page_title="Sailing Logs Collector", layout="wide")
    st.title("Inventário de logs importados")

    master_csv_path = STORAGE_ROOT / MASTER_CSV_NAME
    index_csv_path = STORAGE_ROOT / INDEX_CSV_NAME

    df_master = load_inventory(master_csv_path)
    df_view = df_master.sort_values("canonical_filename")

    render_inventory(df_view, index_csv_path)


if __name__ == "__main__":
    main()

