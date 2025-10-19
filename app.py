"""Streamlit application for managing the Sailing Logs Collector storage."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DEFAULT_STORAGE_ROOT = Path("storage").resolve()
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", DEFAULT_STORAGE_ROOT)).resolve()
MASTER_CSV_NAME = os.getenv("MASTER_CSV_NAME", "master.csv")
INDEX_CSV_NAME = os.getenv("INDEX_CSV_NAME", "index.csv")
REGATTA_CONFIG_NAME = os.getenv("REGATTA_CONFIG_NAME", "regattas.json")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", os.getenv("ADMIN_PASS", "admin"))
TIMEZONE_NAME = os.getenv("TIMEZONE", "UTC")

try:
    LOCAL_TZ = ZoneInfo(TIMEZONE_NAME)
except Exception:  # pragma: no cover - invalid timezone definitions are rare
    LOCAL_TZ = ZoneInfo("UTC")
    TIMEZONE_WARNING = True
else:
    TIMEZONE_WARNING = False

ACCEPTED_EXTENSIONS = ["gpx", "csv", "fit", "tcx", "nmea", "zip"]

REGATTA_CONFIG_PATH = STORAGE_ROOT / REGATTA_CONFIG_NAME


INVENTORY_COLUMNS = [
    "received_at_utc",
    "log_date",
    "log_datetime_utc",
    "log_datetime_source",
    "regatta",
    "regatta_date",
    "athlete_name",
    "class",
    "contact",
    "source_channel",
    "raw_filename",
    "canonical_filename",
    "drive_path",
    "file_hash_sha256",
    "file_ext",
    "src_format_guess",
    "parse_status",
    "notes",
]


def ensure_storage_root() -> None:
    """Create the storage root if it does not exist."""

    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


def load_regatta_config() -> dict[str, list[str]]:
    """Load regatta options defined by administrators."""

    if not REGATTA_CONFIG_PATH.exists():
        return {}

    try:
        with REGATTA_CONFIG_PATH.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except Exception as exc:  # pragma: no cover - defensive UI feedback
        st.error(f"Não foi possível ler as configurações de regata: {exc}")
        return {}

    regattas = data.get("regattas", []) if isinstance(data, dict) else []
    mapping: dict[str, list[str]] = {}
    for entry in regattas:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        classes = entry.get("classes", [])
        if not isinstance(classes, list):
            classes = []
        cleaned = []
        for value in classes:
            value_str = str(value).strip()
            if value_str:
                cleaned.append(value_str)
        mapping[str(name)] = cleaned
    return mapping


def save_regatta_config(mapping: dict[str, list[str]]) -> None:
    """Persist the regatta options for collector dropdowns."""

    REGATTA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "regattas": [
            {"name": name, "classes": classes}
            for name, classes in sorted(mapping.items(), key=lambda item: item[0].lower())
        ]
    }
    with REGATTA_CONFIG_PATH.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_slug_pattern = re.compile(r"[^a-z0-9]+")


def slugify(value: str, fallback: str = "sem-nome") -> str:
    """Return a filesystem-friendly slug for ``value``."""

    value = value or ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = _slug_pattern.sub("-", normalized).strip("-")
    return normalized or fallback


def regatta_folder_name(regatta: str, regatta_date: Optional[date]) -> str:
    """Return the folder name for a regatta."""

    prefix_date = regatta_date.strftime("%Y-%m") if regatta_date else datetime.now(LOCAL_TZ).strftime("%Y-%m")
    return f"{prefix_date}_{slugify(regatta, fallback='regata')}"


def _ensure_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in INVENTORY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[INVENTORY_COLUMNS]


def load_inventory(csv_path: Path) -> pd.DataFrame:
    """Load an inventory CSV and ensure it exposes the expected columns."""

    if not csv_path.exists():
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive UI feedback
        st.error(f"Não foi possível ler o inventário '{csv_path}': {exc}")
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    return _ensure_dataframe_columns(df)


def append_inventory_row(csv_path: Path, row: dict) -> None:
    """Append ``row`` to ``csv_path`` ensuring headers are preserved."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {column: row.get(column, "") for column in INVENTORY_COLUMNS}
    df = pd.DataFrame([ordered])

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def remove_row_csv(path: Path, predicate) -> None:
    """Remove rows from ``path`` matching ``predicate``."""

    if not path.exists():
        return

    df = pd.read_csv(path)
    if df.empty:
        return
    mask = df.apply(predicate, axis=1)
    df = df.loc[~mask]
    df.to_csv(path, index=False)


def resolve_path(raw_path: str) -> Path:
    raw_path = raw_path or ""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return STORAGE_ROOT / candidate


def humanize_filename(value: str) -> str:
    return value or "(sem nome)"


def compute_sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def trigger_rerun() -> None:
    """Trigger a Streamlit rerun supporting both legacy and current APIs."""

    rerun = getattr(st, "experimental_rerun", None)
    if rerun is not None:
        rerun()
    else:  # pragma: no cover - depends on Streamlit runtime
        st.rerun()


def parse_gpx_datetime(data: bytes) -> Optional[datetime]:
    """Extract the first ``<time>`` entry from a GPX file."""

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    # ``time`` tags may appear in multiple namespaces.
    for elem in root.iter():
        if elem.tag.endswith("time") and elem.text:
            try:
                parsed = datetime.fromisoformat(elem.text.replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed.astimezone(timezone.utc)
    return None


def detect_log_datetime(extension: str, data: bytes, submitted_date: Optional[date]) -> tuple[Optional[datetime], str]:
    """Return ``(datetime_utc, source)`` for the uploaded ``data``."""

    extension = (extension or "").lstrip(".").lower()
    if extension == "gpx":
        dt = parse_gpx_datetime(data)
        if dt:
            return dt, "file"

    if submitted_date:
        dt = datetime.combine(submitted_date, datetime.min.time(), tzinfo=LOCAL_TZ).astimezone(timezone.utc)
        return dt, "form"

    return datetime.now(timezone.utc), "upload"


def ensure_regatta_directories(base: Path) -> None:
    for name in ["_INBOX", "_NORMALIZED", "_REPORTS"]:
        (base / name).mkdir(parents=True, exist_ok=True)


def _append_to_inbox(inbox_dir: Path, original_name: str, data: bytes) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = f"{timestamp}_{original_name}"
    inbox_path = inbox_dir / safe_name
    with inbox_path.open("wb") as stream:
        stream.write(data)


def store_log_file(regatta_dir: Path, log_date: date, athlete_slug: str, canonical_name: str, data: bytes) -> Path:
    normalized_dir = regatta_dir / "_NORMALIZED" / log_date.strftime("%Y-%m-%d") / "@Atletas" / athlete_slug
    normalized_dir.mkdir(parents=True, exist_ok=True)
    destination = normalized_dir / canonical_name
    with destination.open("wb") as stream:
        stream.write(data)
    return destination


def build_zip(rows: pd.DataFrame) -> Optional[bytes]:
    """Return a zip archive containing the files referenced in ``rows``."""

    files_added = False
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for _, row in rows.iterrows():
            drive_path = row.get("drive_path", "")
            if not drive_path:
                continue
            file_path = resolve_path(str(drive_path))
            if not file_path.exists():
                continue

            storage_parts = Path(drive_path).parts
            regatta_folder = storage_parts[0] if storage_parts else slugify(row.get("regatta", "regata"))

            log_date_value = row.get("log_date", "")
            if pd.isna(log_date_value):
                log_date = ""
            else:
                log_date = str(log_date_value).strip()
            filename = str(row.get("canonical_filename") or Path(drive_path).name)

            arc_parts = [regatta_folder]
            if log_date:
                arc_parts.append(log_date)
            arc_parts.append(filename)

            arcname = "/".join(arc_parts)
            archive.write(file_path, arcname=arcname)
            files_added = True

    if not files_added:
        return None

    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Streamlit UI helpers
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
        log_date = row.get("log_date", "")

        with col1:
            st.write(f"**{display_name}**")
            st.caption(
                "\n".join(
                    filter(
                        None,
                        [
                            drive_path or "(caminho não informado)",
                            f"Data do log: {log_date}" if log_date else "",
                        ],
                    )
                )
            )

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
                    trigger_rerun()
                except Exception as exc:  # pragma: no cover - defensive UI
                    st.error(f"Falha ao excluir: {exc}")

        with col4:
            st.code(file_hash[:8] if file_hash else "—", language=None)


# ---------------------------------------------------------------------------
# Collector workflow
# ---------------------------------------------------------------------------

@dataclass
class UploadResult:
    filename: str
    file_hash: str
    log_date: date
    message: str
    success: bool


def process_uploads(
    *,
    regatta: str,
    regatta_date: date,
    athlete_name: str,
    sail_class: str,
    contact: str,
    files: Iterable[st.runtime.uploaded_file_manager.UploadedFile],
    master_df: pd.DataFrame,
) -> List[UploadResult]:
    results: List[UploadResult] = []

    existing_hashes = set(str(value) for value in master_df["file_hash_sha256"].dropna())
    regatta_dir = STORAGE_ROOT / regatta_folder_name(regatta, regatta_date)
    ensure_regatta_directories(regatta_dir)

    master_csv_path = STORAGE_ROOT / MASTER_CSV_NAME
    index_csv_path = regatta_dir / INDEX_CSV_NAME

    athlete_slug = slugify(athlete_name)

    for uploaded in files:
        raw_name = uploaded.name
        data = uploaded.read()
        file_hash = compute_sha256(data)

        if file_hash in existing_hashes:
            results.append(
                UploadResult(
                    filename=raw_name,
                    file_hash=file_hash,
                    log_date=regatta_date,
                    message="Arquivo já existe no inventário (deduplicado).",
                    success=False,
                )
            )
            continue

        extension = Path(raw_name).suffix.lower().lstrip(".")
        log_dt, source = detect_log_datetime(extension, data, regatta_date)
        log_dt_local = log_dt.astimezone(LOCAL_TZ) if log_dt else datetime.now(LOCAL_TZ)
        log_date = log_dt_local.date()
        canonical_name = f"{log_date.strftime('%Y-%m-%d')}_{athlete_slug}_{file_hash[:8]}"
        if extension:
            canonical_name += f".{extension}"

        _append_to_inbox(regatta_dir / "_INBOX", raw_name, data)
        destination = store_log_file(regatta_dir, log_date, athlete_slug, canonical_name, data)

        drive_path = destination.relative_to(STORAGE_ROOT).as_posix()
        received_at = datetime.now(timezone.utc)

        row = {
            "received_at_utc": received_at.isoformat(),
            "log_date": log_date.isoformat(),
            "log_datetime_utc": log_dt.isoformat() if log_dt else "",
            "log_datetime_source": source,
            "regatta": regatta,
            "regatta_date": regatta_date.isoformat() if regatta_date else "",
            "athlete_name": athlete_name,
            "class": sail_class,
            "contact": contact,
            "source_channel": "web",
            "raw_filename": raw_name,
            "canonical_filename": canonical_name,
            "drive_path": drive_path,
            "file_hash_sha256": file_hash,
            "file_ext": extension,
            "src_format_guess": extension,
            "parse_status": "pending",
            "notes": "",
        }

        append_inventory_row(master_csv_path, row)
        append_inventory_row(index_csv_path, row)
        existing_hashes.add(file_hash)

        results.append(
            UploadResult(
                filename=canonical_name,
                file_hash=file_hash,
                log_date=log_date,
                message="Upload processado com sucesso.",
                success=True,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def collector_tab(master_df: pd.DataFrame) -> None:
    st.subheader("Coletor de logs")
    st.write(
        "Envie seus arquivos de rastreamento para que possamos organizar o inventário da regata."
    )

    regatta_config = load_regatta_config()

    regatta = ""
    regatta_names = sorted(regatta_config)
    if regatta_names:
        placeholder = "Selecione uma regata"
        regatta_option = st.selectbox(
            "Regata *",
            [placeholder] + regatta_names,
            key="collector_regatta_select",
        )
        regatta = "" if regatta_option == placeholder else regatta_option
        st.session_state.pop("collector_regatta_text", None)
    else:
        regatta = st.text_input("Regata *", key="collector_regatta_text")
        st.session_state.pop("collector_regatta_select", None)

    classes_for_regatta = regatta_config.get(regatta, []) if regatta else []
    class_lookup: dict[str, str] = {}
    if classes_for_regatta:
        for value in classes_for_regatta:
            cleaned_value = value.strip()
            if not cleaned_value:
                continue
            normalized = cleaned_value.casefold()
            class_lookup.setdefault(normalized, value)

    sail_class = ""
    class_key = "collector_class_select"
    regatta_state_key = "collector_selected_regatta"
    class_placeholder = "Selecione a classe"
    if classes_for_regatta:
        previous_regatta = st.session_state.get(regatta_state_key)
        if regatta != previous_regatta:
            st.session_state.pop(class_key, None)
        selected_option = st.selectbox(
            "Classe *",
            [class_placeholder] + classes_for_regatta,
            key=class_key,
        )
        if selected_option == class_placeholder:
            sail_class = ""
        else:
            selected_clean = selected_option.strip()
            normalized = selected_clean.casefold()
            sail_class = class_lookup.get(normalized, selected_clean)
        st.session_state[regatta_state_key] = regatta
        st.session_state.pop("collector_class_text", None)
    elif regatta or not regatta_names:
        sail_class = st.text_input("Classe", key="collector_class_text").strip()
        st.session_state.pop(class_key, None)
        st.session_state.pop(regatta_state_key, None)
    else:
        st.session_state.pop(class_key, None)
        st.session_state.pop("collector_class_text", None)
        st.session_state.pop(regatta_state_key, None)

    submit_disabled = (not regatta.strip()) or (
        classes_for_regatta and not sail_class.strip()
    )

    with st.form("collector_form"):
        regatta_date = st.date_input("Data da regata *", value=date.today())
        athlete_name = st.text_input("Nome do atleta *")
        contact = st.text_input("Contato (e-mail ou telefone) *")
        uploaded_files = st.file_uploader(
            "Arquivos de log *",
            accept_multiple_files=True,
            type=ACCEPTED_EXTENSIONS,
            help="Formatos aceitos: GPX, CSV, FIT, TCX, NMEA ou ZIP.",
        )
        submitted = st.form_submit_button("Enviar logs", disabled=submit_disabled)

    if submitted:
        if not regatta.strip():
            st.error("Informe a regata.")
            return
        if not athlete_name.strip():
            st.error("Informe o nome do atleta.")
            return
        if classes_for_regatta and not sail_class.strip():
            st.error("Selecione uma classe disponível para a regata escolhida.")
            return
        if not contact.strip():
            st.error("Informe um contato para confirmação.")
            return
        if not uploaded_files:
            st.error("Envie pelo menos um arquivo de log.")
            return

        results = process_uploads(
            regatta=regatta,
            regatta_date=regatta_date,
            athlete_name=athlete_name,
            sail_class=sail_class,
            contact=contact,
            files=uploaded_files,
            master_df=master_df,
        )

        success = [result for result in results if result.success]
        duplicates = [result for result in results if not result.success]

        if success:
            st.success("Uploads concluídos!")
            for result in success:
                st.write(
                    f"• **{result.filename}** – protocolo `{result.file_hash[:8]}` – data {result.log_date.isoformat()}"
                )

        if duplicates:
            st.warning("Alguns arquivos foram ignorados por já existirem no inventário:")
            for result in duplicates:
                st.write(
                    f"• {result.filename} – hash `{result.file_hash[:8]}`"
                )


def admin_tab() -> None:
    st.subheader("Administração")

    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        with st.form("admin_login"):
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar")
        if submitted:
            if password == ADMIN_PASSWORD:
                st.session_state["admin_authenticated"] = True
                trigger_rerun()
            else:
                st.error("Senha incorreta.")
        return

    regatta_config = load_regatta_config()

    st.markdown("### Configuração do coletor")
    with st.expander(
        "Regatas e classes disponíveis no coletor",
        expanded=not regatta_config,
    ):
        if not regatta_config:
            st.info("Nenhuma regata configurada. Adicione uma nova regata abaixo.")

        regatta_names = sorted(regatta_config)
        selector_options = ["(Nova regata)"] + regatta_names
        selected_option = st.selectbox(
            "Selecione uma regata para configurar",
            selector_options,
            key="admin_regatta_selector",
        )
        is_new = selected_option == "(Nova regata)"
        widget_suffix = slugify(selected_option or "nova-regata")
        if is_new:
            widget_suffix = f"new-{widget_suffix}"
        default_name = "" if is_new else selected_option
        default_classes = (
            ", ".join(regatta_config.get(selected_option, [])) if not is_new else ""
        )

        delete_clicked = False
        with st.form(f"regatta_form_{widget_suffix}"):
            regatta_name = st.text_input(
                "Nome da regata",
                value=default_name,
                key=f"regatta_name_{widget_suffix}",
            )
            classes_value = st.text_input(
                "Classes disponíveis (separe por vírgula)",
                value=default_classes,
                help="Ex.: HPE30, HPE25",
                key=f"regatta_classes_{widget_suffix}",
            )
            submitted_config = st.form_submit_button("Salvar regata")
            if not is_new:
                delete_clicked = st.form_submit_button("Remover regata")

        if delete_clicked and not is_new:
            if selected_option in regatta_config:
                regatta_config.pop(selected_option)
                save_regatta_config(regatta_config)
                st.success("Regata removida do coletor.")
                trigger_rerun()
                return

        if submitted_config:
            regatta_name_clean = regatta_name.strip()
            if not regatta_name_clean:
                st.error("Informe o nome da regata.")
            else:
                classes = []
                for part in classes_value.split(","):
                    value = part.strip()
                    if value and value not in classes:
                        classes.append(value)

                if is_new:
                    if regatta_name_clean in regatta_config:
                        st.error("Já existe uma regata com esse nome.")
                    else:
                        regatta_config[regatta_name_clean] = classes
                        save_regatta_config(regatta_config)
                        st.success("Regata adicionada ao coletor.")
                        trigger_rerun()
                        return
                else:
                    original_name = selected_option
                    if (
                        regatta_name_clean != original_name
                        and regatta_name_clean in regatta_config
                    ):
                        st.error("Já existe uma regata com esse nome.")
                    else:
                        if regatta_name_clean != original_name:
                            regatta_config.pop(original_name, None)
                        regatta_config[regatta_name_clean] = classes
                        save_regatta_config(regatta_config)
                        st.success("Regata atualizada no coletor.")
                        trigger_rerun()
                        return

        if regatta_config:
            st.caption(
                "Regatas configuradas: "
                + ", ".join(name for name in sorted(regatta_config))
            )

    st.info("Use o seletor abaixo para visualizar as regatas disponíveis.")

    regatta_dirs = [
        path
        for path in STORAGE_ROOT.iterdir()
        if path.is_dir() and path.name not in {".streamlit"}
    ] if STORAGE_ROOT.exists() else []

    if not regatta_dirs:
        st.warning("Nenhuma regata encontrada no armazenamento.")
        if st.button("Sair"):
            st.session_state["admin_authenticated"] = False
            trigger_rerun()
        return

    regatta_dirs.sort(key=lambda p: p.name)
    regatta_options = {path.name: path for path in regatta_dirs}
    selected_label = st.selectbox("Regata", list(regatta_options.keys()))
    regatta_dir = regatta_options[selected_label]
    index_csv_path = regatta_dir / INDEX_CSV_NAME
    df_regatta = load_inventory(index_csv_path)

    if df_regatta.empty:
        st.info("Nenhum arquivo disponível para esta regata.")
    else:
        dates = sorted(value for value in df_regatta["log_date"].dropna().unique())
        date_options = ["Todos"] + dates
        selected_date = st.selectbox("Filtrar por data", date_options, format_func=lambda x: x)

        if selected_date != "Todos":
            df_view = df_regatta[df_regatta["log_date"] == selected_date]
        else:
            df_view = df_regatta

        st.markdown("### Inventário")
        st.dataframe(df_view)

        if selected_date != "Todos":
            zip_bytes = build_zip(df_view)
            st.download_button(
                "Baixar ZIP do dia",
                data=zip_bytes,
                file_name=f"{regatta_dir.name}_{selected_date}.zip",
                mime="application/zip",
                disabled=zip_bytes is None,
            )
        zip_all = build_zip(df_regatta)
        st.download_button(
            "Baixar ZIP completo da regata",
            data=zip_all,
            file_name=f"{regatta_dir.name}.zip",
            mime="application/zip",
            disabled=zip_all is None,
        )

        st.markdown("### Ações por arquivo")
        render_inventory(df_view, index_csv_path)

    if st.button("Sair"):
        st.session_state["admin_authenticated"] = False
        trigger_rerun()


def main() -> None:
    st.set_page_config(page_title="Sailing Logs Collector", layout="wide")
    ensure_storage_root()

    if TIMEZONE_WARNING:
        st.warning(
            "Fuso horário configurado não pôde ser carregado. Utilizando UTC como padrão."
        )

    st.title("Sailing Logs Collector")

    master_csv_path = STORAGE_ROOT / MASTER_CSV_NAME
    master_df = load_inventory(master_csv_path)

    tab_collector, tab_admin = st.tabs(["Coletor", "Admin"])

    with tab_collector:
        collector_tab(master_df)

    with tab_admin:
        admin_tab()


if __name__ == "__main__":
    main()
