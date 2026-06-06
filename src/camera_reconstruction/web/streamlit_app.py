from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from camera_reconstruction.services.reconstruction_service import (
    ReconstructionConfig,
    ReconstructionError,
    ReconstructionResult,
    ReconstructionService,
)
from camera_reconstruction.storage.files import ensure_directory, save_upload_bytes, unique_image_path


st.set_page_config(
    layout="wide",
    page_title="Camera PnP Reconstruction",
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VAR_DIR = PROJECT_ROOT / "var"
UPLOADS_DIR = ensure_directory(VAR_DIR / "uploads")
RESULTS_DIR = ensure_directory(VAR_DIR / "results")


def main() -> None:
    _initialize_navigation_state()
    theme = str(st.session_state.current_theme).lower()

    _apply_dashboard_styles(theme)
    page, theme = _render_navigation()
    _render_header()

    if page == "Guide":
        _render_guide()
        return

    config, uploaded_image, run_clicked = _render_sidebar()

    if run_clicked:
        if uploaded_image is None:
            st.sidebar.error("Please upload a checkerboard image before running PnP.")
        else:
            _run_reconstruction(uploaded_image, config)

    result = st.session_state.get("reconstruction_result")
    if result is None:
        _render_empty_state()
        return

    _render_result(result)


def _query_value(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def _resolve_view() -> str:
    requested_view = _query_value("view", "dashboard")
    return "Guide" if requested_view.lower() == "guide" else "Dashboard"


def _resolve_theme() -> str:
    requested_theme = _query_value("theme", "dark")
    return "light" if requested_theme.lower() == "light" else "dark"


def _initialize_navigation_state() -> None:
    if "current_view" not in st.session_state:
        st.session_state.current_view = _resolve_view()
    if "current_theme" not in st.session_state:
        st.session_state.current_theme = "Light" if _resolve_theme() == "light" else "Dark"

    if dict(st.query_params):
        st.query_params.clear()


def _render_navigation() -> tuple[str, str]:
    st.session_state.current_view = "Guide" if st.session_state.current_view == "Guide" else "Dashboard"
    st.session_state.current_theme = "Light" if str(st.session_state.current_theme).lower() == "light" else "Dark"

    with st.sidebar:
        st.markdown(
            """
            <div class="nav-panel">
              <span>Workspace</span>
              <strong>Camera Reconstruction</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.segmented_control(
            "View mode",
            ["Dashboard", "Guide"],
            key="current_view",
            required=True,
            label_visibility="collapsed",
            width="stretch",
        )
        st.markdown('<div class="theme-panel-label">Interface theme</div>', unsafe_allow_html=True)
        theme = st.segmented_control(
            "Interface theme",
            ["Dark", "Light"],
            key="current_theme",
            required=True,
            label_visibility="collapsed",
            width="stretch",
        )

    return str(page or "Dashboard"), str(theme or "Dark").lower()


def _render_header() -> None:
    st.markdown(
        """
        <div class="app-title">
          <div>
            <span>Perspective-n-Point Dashboard</span>
            <div class="title-main">Camera PnP Reconstruction</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_guide() -> None:
    st.markdown(
        """
        <div class="st-guide-hero">
          <span>Usage guide</span>
          <h2>Perspective-n-Point reconstruction workflow</h2>
          <p>Use this guide to prepare a checkerboard image, run pose estimation, and interpret the camera reconstruction output.</p>
        </div>
        <div class="st-guide-grid">
          <article>
            <strong>01</strong>
            <h3>Prepare the image</h3>
            <p>Use a clear checkerboard photo where the internal corners are visible, sharp, and not blocked by reflections or motion blur.</p>
          </article>
          <article>
            <strong>02</strong>
            <h3>Set board parameters</h3>
            <p>Corners X and Corners Y are internal-corner counts. Square size is the physical size of one square in a consistent unit.</p>
          </article>
          <article>
            <strong>03</strong>
            <h3>Run pose estimation</h3>
            <p>Leave the intrinsic matrix at its default estimate when calibration is unavailable, or enter fx, fy, cx, and cy if known.</p>
          </article>
          <article>
            <strong>04</strong>
            <h3>Review the result</h3>
            <p>The overlay shows projected X/Y/Z axes. Lower RMSE, mean error, and max error indicate a tighter reprojection fit.</p>
          </article>
        </div>
        <div class="st-guide-note">
          <h3>Interpretation and limitations</h3>
          <p>This application estimates camera pose from known 3D checkerboard points and their detected 2D image positions. The result should be reported as a Perspective-n-Point reconstruction, including R, tvec, C_world, the projection matrix, and reprojection error. A single 2D pixel alone does not define a unique 3D point without depth or an additional geometric constraint.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[ReconstructionConfig, object | None, bool]:
    with st.sidebar:
        st.header("Input")
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

        st.subheader("Checkerboard Params")
        col_x, col_y = st.columns(2)
        with col_x:
            corners_x = st.number_input("Corners X", min_value=2, value=9, step=1)
        with col_y:
            corners_y = st.number_input("Corners Y", min_value=2, value=6, step=1)

        col_square, col_axis = st.columns(2)
        with col_square:
            square_size = st.number_input("Square size", min_value=0.000001, value=1.0, step=0.1, format="%.6f")
        with col_axis:
            axis_length = st.number_input("Axis length", min_value=0.000001, value=3.0, step=0.1, format="%.6f")

        with st.expander("Intrinsic Matrix K", expanded=False):
            use_custom_k = st.checkbox("Use custom intrinsic matrix", value=False)
            k_col_1, k_col_2 = st.columns(2)
            with k_col_1:
                fx = st.number_input("fx", min_value=0.000001, value=736.0, step=10.0, format="%.6f")
                cx = st.number_input("cx", value=368.0, step=10.0, format="%.6f")
            with k_col_2:
                fy = st.number_input("fy", min_value=0.000001, value=736.0, step=10.0, format="%.6f")
                cy = st.number_input("cy", value=368.0, step=10.0, format="%.6f")
            skew = st.number_input("skew", value=0.0, step=0.01, format="%.6f")

        with st.expander("Distortion", expanded=False):
            distortion_text = st.text_input("Coefficients", value="", placeholder="k1, k2, p1, p2, k3")

        run_clicked = st.button("Run PnP", type="primary", use_container_width=True)

    try:
        distortion = _parse_distortion_coefficients(distortion_text)
    except ValueError:
        if run_clicked:
            st.sidebar.error("Distortion coefficients must be comma-separated numbers.")
        distortion = ()
        run_clicked = False

    config = ReconstructionConfig(
        corners_x=int(corners_x),
        corners_y=int(corners_y),
        square_size=float(square_size),
        fx=float(fx) if use_custom_k else None,
        fy=float(fy) if use_custom_k else None,
        cx=float(cx) if use_custom_k else None,
        cy=float(cy) if use_custom_k else None,
        skew=float(skew) if use_custom_k else 0.0,
        distortion_coefficients=distortion,
        axis_length=float(axis_length),
    )
    return config, uploaded_image, run_clicked


def _run_reconstruction(uploaded_image: object, config: ReconstructionConfig) -> None:
    with st.spinner("Solving PnP and rendering the 3D axis overlay..."):
        try:
            upload_path = save_upload_bytes(uploaded_image.getvalue(), UPLOADS_DIR, uploaded_image.name)
            output_path = unique_image_path(RESULTS_DIR, "overlay.png")
            result = ReconstructionService().reconstruct(upload_path, output_path, config)
        except (ValueError, ReconstructionError) as exc:
            st.error(str(exc))
            return

    st.session_state["reconstruction_result"] = result
    st.success("Reconstruction completed.")


def _render_empty_state() -> None:
    left, right = st.columns([0.65, 0.35], gap="large")
    with left:
        st.info("Upload a checkerboard image and click Run PnP to render the 3D coordinate axes.")
    with right:
        st.markdown("### Evaluation")
        metric_cols = st.columns(3)
        metric_cols[0].metric("RMSE", "-")
        metric_cols[1].metric("Mean Error", "-")
        metric_cols[2].metric("Max Error", "-")


def _render_result(result: ReconstructionResult) -> None:
    image_col, eval_col = st.columns([0.65, 0.35], gap="large")

    with image_col:
        st.markdown("### 3D Axis Overlay")
        overlay_rgb = _read_overlay_rgb(result.overlay_image_path)
        st.image(overlay_rgb, use_container_width=True)
        st.download_button(
            "Download Overlay Image",
            data=result.overlay_image_path.read_bytes(),
            file_name=result.overlay_image_path.name,
            mime="image/png",
            use_container_width=True,
        )

    with eval_col:
        st.markdown("### Evaluation")
        metric_row_1 = st.columns(3)
        metric_row_1[0].metric("RMSE", f"{result.pnp.reprojection_rmse:.6g} px")
        metric_row_1[1].metric("Mean Error", f"{result.pnp.reprojection_mean:.6g} px")
        metric_row_1[2].metric("Max Error", f"{result.pnp.reprojection_max:.6g} px")

        metric_row_2 = st.columns(2)
        metric_row_2[0].metric("Detected Points", f"{result.checkerboard.image_points.shape[0]}")
        metric_row_2[1].metric("Image Size", f"{result.image_width} x {result.image_height}")

        st.markdown("### Camera Center")
        st.latex(_vector_latex(r"\mathbf{C}_{world}", result.pnp.camera_center_world))

    matrix_tab, points_tab = st.tabs(["Ma trận (Matrices)", "Dữ liệu điểm (2D/3D Points)"])

    with matrix_tab:
        _render_matrices_tab(result)

    with points_tab:
        _render_points_tab(result)


def _render_matrices_tab(result: ReconstructionResult) -> None:
    k_col, r_col = st.columns(2, gap="large")
    with k_col:
        st.latex(_matrix_latex(r"\mathbf{K}", result.intrinsics.matrix()))
        st.latex(_vector_latex(r"\mathbf{t}", result.pnp.tvec))
        st.latex(_vector_latex(r"\mathbf{r}", result.pnp.rvec))
    with r_col:
        st.latex(_matrix_latex(r"\mathbf{R}", result.pnp.R))
        st.latex(_vector_latex(r"\mathbf{C}_{world}", result.pnp.camera_center_world))

    st.latex(_matrix_latex(r"\mathbf{P} = K[R|t]", result.pnp.projection_matrix))


def _render_points_tab(result: ReconstructionResult) -> None:
    camera_points = (result.pnp.R @ result.checkerboard.object_points.T).T + result.pnp.tvec.reshape(1, 3)
    points_df = pd.DataFrame(
        {
            "Image 2D [u, v]": [_format_tuple(point) for point in result.checkerboard.image_points],
            "World 3D [X, Y, Z]": [_format_tuple(point) for point in result.checkerboard.object_points],
            "Camera 3D [Xc, Yc, Zc]": [_format_tuple(point) for point in camera_points],
            "Reprojected 2D [u, v]": [_format_tuple(point) for point in result.pnp.reprojected_points],
            "Error px": [f"{value:.6g}" for value in result.pnp.reprojection_errors],
        }
    )
    st.dataframe(points_df, use_container_width=True, height=360, hide_index=True)


def _read_overlay_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Could not read the rendered overlay image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _parse_distortion_coefficients(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    normalized = value.replace(";", ",").replace("\n", ",")
    return tuple(float(part.strip()) for part in normalized.split(",") if part.strip())


def _matrix_latex(name: str, matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=float):
        rows.append(" & ".join(_latex_number(value) for value in row))
    body = r" \\ ".join(rows)
    return rf"{name} = \begin{{bmatrix}} {body} \end{{bmatrix}}"


def _vector_latex(name: str, vector: np.ndarray) -> str:
    rows = [rf"{_latex_number(value)}" for value in np.asarray(vector, dtype=float).reshape(-1)]
    body = r" \\ ".join(rows)
    return rf"{name} = \begin{{bmatrix}} {body} \end{{bmatrix}}"


def _latex_number(value: float) -> str:
    formatted = f"{float(value):.6g}"
    if "e" not in formatted and "E" not in formatted:
        return formatted

    mantissa, exponent = formatted.lower().split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _format_tuple(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in np.asarray(values, dtype=float).reshape(-1)) + "]"


def _theme_tokens_css(theme: str) -> str:
    palettes = {
        "dark": {
            "bg": "#003049",
            "panel": "#082f49",
            "card": "#0b3a55",
            "surface": "#123f5b",
            "input": "#071f31",
            "text": "#fdf0d5",
            "muted": "#c9dde8",
            "border": "#669bbc",
            "header": "rgba(0, 48, 73, 0.9)",
            "shadow": "0 18px 42px rgba(0, 18, 28, 0.34)",
            "soft-accent": "rgba(102, 155, 188, 0.22)",
            "table-header": "#0f425f",
        },
        "light": {
            "bg": "#fdf0d5",
            "panel": "#f5e5bd",
            "card": "#fff8e8",
            "surface": "#f0dcae",
            "input": "#fffaf0",
            "text": "#003049",
            "muted": "#405f72",
            "border": "#669bbc",
            "header": "rgba(253, 240, 213, 0.9)",
            "shadow": "0 18px 42px rgba(0, 48, 73, 0.18)",
            "soft-accent": "rgba(193, 18, 31, 0.1)",
            "table-header": "#f2dfb9",
        },
    }
    palette = palettes["light"] if theme == "light" else palettes["dark"]
    declarations = "\n".join(f"            --cr-{name}: {value};" for name, value in palette.items())
    return f"""
        <style>
          :root {{
{declarations}
            --cr-accent: #c1121f;
            --cr-accent-dark: #780000;
            --cr-cream: #fdf0d5;
            --cr-blue: #003049;
            --cr-steel: #669bbc;
            --background-color: var(--cr-bg);
            --secondary-background-color: var(--cr-card);
            --text-color: var(--cr-text);
            --primary-color: var(--cr-accent);
            --border-color: var(--cr-border);
          }}
        </style>
    """


def _apply_dashboard_styles(theme: str) -> None:
    st.markdown(_theme_tokens_css(theme), unsafe_allow_html=True)
    st.markdown(
        """
        <style>
          :root {
            --cr-accent: #c1121f;
            --cr-accent-dark: #780000;
            --cr-cream: #fdf0d5;
            --cr-blue: #003049;
            --cr-steel: #669bbc;
          }

          .stApp,
          div[data-testid="stAppViewContainer"] {
            background:
              radial-gradient(circle at top left, var(--cr-soft-accent), transparent 34rem),
              var(--cr-bg);
            color: var(--cr-text);
          }

          div[data-testid="stHeader"] {
            background: var(--cr-header) !important;
          }

          div[data-testid="stDeployButton"],
          .stDeployButton,
          button[title="Deploy"],
          a[title="Deploy"],
          [aria-label="Deploy"],
          [data-testid="stToolbar"] a[href*="deploy"] {
            display: none !important;
          }

          section[data-testid="stSidebar"] {
            background:
              linear-gradient(180deg, var(--cr-panel), color-mix(in srgb, var(--cr-panel) 84%, var(--cr-surface) 16%));
            border-right: 1px solid var(--cr-border);
          }

          section[data-testid="stSidebar"] h1,
          section[data-testid="stSidebar"] h2,
          section[data-testid="stSidebar"] h3,
          section[data-testid="stSidebar"] p,
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] span {
            color: var(--cr-text);
          }

          h1, h2, h3, h4,
          div[data-testid="stMarkdownContainer"],
          div[data-testid="stMarkdownContainer"] * {
            color: var(--cr-text);
          }

          div[data-testid="stAlert"] {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--cr-card) 88%, var(--cr-steel) 12%);
          }

          div[data-testid="stAlert"] * {
            color: var(--cr-text) !important;
          }

          div[data-testid="stFileUploader"] section {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--cr-card) 88%, var(--cr-steel) 12%);
          }

          div[data-testid="stFileUploader"] section * {
            color: var(--cr-text) !important;
          }

          div[data-testid="stFileUploader"] button,
          div[data-testid="stFileUploader"] [role="button"] {
            border: 1px solid var(--cr-accent-dark) !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-dark)) !important;
            color: var(--cr-cream) !important;
          }

          div[data-testid="stFileUploader"] button *,
          div[data-testid="stFileUploader"] [role="button"] * {
            color: var(--cr-cream) !important;
            fill: var(--cr-cream) !important;
          }

          div[data-baseweb="input"],
          div[data-baseweb="base-input"],
          input,
          textarea {
            background: var(--cr-input) !important;
            color: var(--cr-text) !important;
            border-color: var(--cr-border) !important;
          }

          input::placeholder,
          textarea::placeholder {
            color: var(--cr-muted) !important;
          }

          div[data-baseweb="input"] button,
          div[data-baseweb="base-input"] button {
            color: var(--cr-text) !important;
          }

          section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            border: 1px solid var(--cr-accent-dark);
            background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-dark));
            color: var(--cr-cream);
            font-weight: 850;
            transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
          }

          section[data-testid="stSidebar"] div[data-testid="stButton"] button * {
            color: var(--cr-cream) !important;
          }

          section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            box-shadow: 0 12px 24px rgba(120, 0, 0, 0.24);
            transform: translateY(-1px);
          }

          .app-title {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 18px;
            background: linear-gradient(135deg, #003049, #0b3a55);
            box-shadow: var(--cr-shadow);
          }

          .app-title span {
            color: #669bbc !important;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }

          .app-title .title-main {
            margin: 4px 0 0;
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1.2;
            letter-spacing: 0;
            color: #fdf0d5 !important;
          }

          div[data-testid="stMetric"] {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 12px;
            background: linear-gradient(180deg, var(--cr-card), color-mix(in srgb, var(--cr-card) 84%, var(--cr-bg) 16%));
            box-shadow: 0 10px 24px rgba(0, 48, 73, 0.16);
          }

          div[data-testid="stMetric"] * {
            color: var(--cr-text) !important;
          }

          div[data-testid="stImage"] {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 10px;
            background: var(--cr-blue);
            box-shadow: 0 16px 34px rgba(0, 48, 73, 0.28);
          }

          div[data-testid="stDownloadButton"] {
            margin-top: 10px;
          }

          div[data-testid="stDownloadButton"] button {
            border: 1px solid var(--cr-accent-dark);
            border-radius: 8px;
            background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-dark));
            color: var(--cr-cream);
            font-weight: 850;
            transition: background 160ms ease, transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
          }

          div[data-testid="stDownloadButton"] button:hover {
            border-color: var(--cr-accent-dark);
            background: var(--cr-accent-dark);
            color: var(--cr-cream);
            box-shadow: 0 12px 24px rgba(120, 0, 0, 0.28);
            transform: translateY(-1px);
          }

          .nav-panel {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #003049, #0b3a55);
            box-shadow: 0 12px 28px rgba(0, 48, 73, 0.18);
          }

          .nav-panel span {
            display: block;
            color: #669bbc !important;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }

          .nav-panel strong {
            display: block;
            margin-top: 2px;
            color: #fdf0d5 !important;
            font-size: 0.95rem;
          }

          .theme-panel-label {
            border: 1px solid var(--cr-border);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            padding: 10px 10px 2px;
            margin-top: 12px;
            background: linear-gradient(180deg, var(--cr-card), color-mix(in srgb, var(--cr-card) 84%, var(--cr-bg) 16%));
            color: var(--cr-muted) !important;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] {
            margin: 10px 0 14px;
          }

          section[data-testid="stSidebar"] div[data-testid="stButtonGroup"] {
            width: 100%;
            margin: 10px 0 14px;
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [role="radiogroup"],
          section[data-testid="stSidebar"] div[data-testid="stButtonGroup"] > div,
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
            width: 100%;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 5px;
            background: var(--cr-card);
            box-shadow: var(--cr-shadow);
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button,
          section[data-testid="stSidebar"] div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-segmented_control"],
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label,
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [role="radio"] {
            min-height: 38px;
            width: 100%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid transparent !important;
            border-radius: 7px !important;
            background: var(--cr-input) !important;
            color: var(--cr-text) !important;
            font-weight: 850 !important;
            transition: background 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button *,
          section[data-testid="stSidebar"] div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-segmented_control"] *,
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label *,
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [role="radio"] * {
            color: inherit !important;
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button:hover,
          section[data-testid="stSidebar"] div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-segmented_control"]:hover,
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label:hover,
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [role="radio"]:hover {
            border-color: var(--cr-accent) !important;
            background: var(--cr-soft-accent) !important;
            transform: translateY(-1px);
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button[aria-selected="true"],
          section[data-testid="stSidebar"] div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"],
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label:has(input:checked),
          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
            border-color: var(--cr-accent-dark) !important;
            background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-dark)) !important;
            color: var(--cr-cream) !important;
            box-shadow: 0 9px 18px rgba(120, 0, 0, 0.22);
          }

          section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label:has(input:checked) * {
            color: var(--cr-cream) !important;
          }

          .view-switch {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
            margin: 10px 0 16px;
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 5px;
            background: var(--cr-card);
            box-shadow: var(--cr-shadow);
          }

          .view-switch a {
            min-height: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid transparent;
            border-radius: 7px;
            color: var(--cr-text) !important;
            font-weight: 850;
            text-decoration: none;
            transition: background 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease;
          }

          .view-switch a:hover {
            border-color: var(--cr-accent);
            background: var(--cr-soft-accent);
            transform: translateY(-1px);
          }

          .view-switch a.active {
            border-color: var(--cr-accent-dark);
            background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-dark));
            color: #fdf0d5 !important;
            box-shadow: 0 9px 18px rgba(120, 0, 0, 0.22);
          }

          .theme-panel {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 10px;
            margin: -4px 0 18px;
            background: linear-gradient(180deg, var(--cr-card), color-mix(in srgb, var(--cr-card) 84%, var(--cr-bg) 16%));
            box-shadow: var(--cr-shadow);
          }

          .theme-panel > span {
            display: block;
            margin-bottom: 8px;
            color: var(--cr-muted) !important;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }

          .theme-switch {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
          }

          .theme-switch a {
            min-height: 34px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            border: 1px solid var(--cr-border);
            border-radius: 7px;
            background: var(--cr-input);
            color: var(--cr-text) !important;
            font-size: 0.88rem;
            font-weight: 850;
            text-decoration: none;
            transition: background 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
          }

          .theme-switch a:hover {
            border-color: var(--cr-accent);
            background: var(--cr-soft-accent);
            transform: translateY(-1px);
          }

          .theme-switch a.active {
            border-color: var(--cr-accent-dark);
            background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-dark));
            color: var(--cr-cream) !important;
            box-shadow: 0 9px 18px rgba(120, 0, 0, 0.22);
          }

          .theme-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
            border: 1px solid currentColor;
          }

          .theme-dot.dark {
            background: #003049;
          }

          .theme-dot.light {
            background: #fdf0d5;
          }

          div[data-testid="stTabs"] button,
          div[data-testid="stTabs"] [role="tab"] {
            color: var(--cr-muted) !important;
          }

          div[data-testid="stTabs"] button[aria-selected="true"],
          div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--cr-accent) !important;
          }

          div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--cr-accent) !important;
          }

          div[data-testid="stDataFrame"] {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            overflow: hidden;
            background: var(--cr-card);
          }

          .st-guide-hero,
          .st-guide-note,
          .st-guide-grid article {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            background: linear-gradient(180deg, var(--cr-card), color-mix(in srgb, var(--cr-card) 90%, var(--cr-bg) 10%));
            box-shadow: 0 12px 28px rgba(0, 48, 73, 0.16);
          }

          .st-guide-hero,
          .st-guide-note {
            padding: 18px 20px;
            margin-bottom: 16px;
          }

          .st-guide-hero span {
            color: var(--cr-accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }

          .st-guide-hero h2,
          .st-guide-note h3,
          .st-guide-grid h3 {
            margin: 4px 0 0;
            letter-spacing: 0;
            color: var(--cr-text);
          }

          .st-guide-hero p,
          .st-guide-note p,
          .st-guide-grid p {
            color: var(--cr-muted);
            margin-bottom: 0;
          }

          .st-guide-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 16px;
          }

          .st-guide-grid article {
            padding: 14px;
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
          }

          .st-guide-grid article:hover {
            border-color: var(--cr-accent);
            box-shadow: 0 16px 34px rgba(120, 0, 0, 0.16);
            transform: translateY(-2px);
          }

          .st-guide-grid strong {
            color: var(--cr-accent);
          }

          .katex,
          .katex * {
            color: var(--cr-text) !important;
          }

          code,
          pre {
            border-radius: 7px;
            background: color-mix(in srgb, var(--cr-card) 82%, var(--cr-steel) 18%) !important;
            color: var(--cr-text) !important;
          }

          @media (max-width: 900px) {
            .st-guide-grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
