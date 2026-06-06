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
    _apply_dashboard_styles()
    page = _render_navigation()
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


def _render_navigation() -> str:
    with st.sidebar:
        return st.radio("View", ["Dashboard", "Guide"], horizontal=True)


def _render_header() -> None:
    st.markdown(
        """
        <div class="app-title">
          <div>
            <span>Perspective-n-Point Dashboard</span>
            <h1>Camera PnP Reconstruction</h1>
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
          <h2>Quy trinh chay thi nghiem PnP</h2>
          <p>Trang nay tom tat cac buoc de upload checkerboard, chay solvePnP va doc ket qua phuc vu bao cao.</p>
        </div>
        <div class="st-guide-grid">
          <article>
            <strong>01</strong>
            <h3>Chuan bi anh</h3>
            <p>Chon anh checkerboard ro net, thay du cac goc trong, khong bi che khuat va khong qua mo.</p>
          </article>
          <article>
            <strong>02</strong>
            <h3>Nhap checkerboard</h3>
            <p>Corners X va Corners Y la so goc trong. Square size la kich thuoc mot o vuong theo cung mot don vi.</p>
          </article>
          <article>
            <strong>03</strong>
            <h3>Chay PnP</h3>
            <p>Neu khong co camera calibration, co the de mac dinh K. Neu co thong so that, nhap fx, fy, cx, cy.</p>
          </article>
          <article>
            <strong>04</strong>
            <h3>Doc ket qua</h3>
            <p>Anh overlay hien truc X/Y/Z. RMSE, Mean Error va Max Error cang nho thi phep chieu lai cang khop.</p>
          </article>
        </div>
        <div class="st-guide-note">
          <h3>Luu y cho bao cao</h3>
          <p>Chuong trinh uoc luong pose camera tu cac cap diem 2D/3D da biet tren checkerboard. Mot pixel 2D rieng le chi tao ra mot tia 3D neu khong co depth hoac rang buoc hinh hoc.</p>
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


def _apply_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
          .app-title {
            border: 1px solid #d7e0e8;
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 18px;
            background: #ffffff;
          }
          .app-title span {
            color: #0b6f76;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }
          .app-title h1 {
            margin: 4px 0 0;
            font-size: 1.75rem;
            letter-spacing: 0;
          }
          div[data-testid="stMetric"] {
            border: 1px solid #d7e0e8;
            border-radius: 8px;
            padding: 12px;
            background: #ffffff;
          }
          div[data-testid="stImage"] {
            border: 1px solid #111923;
            border-radius: 8px;
            padding: 10px;
            background: #101820;
          }
          .st-guide-hero,
          .st-guide-note,
          .st-guide-grid article {
            border: 1px solid #d7e0e8;
            border-radius: 8px;
            background: #ffffff;
          }
          .st-guide-hero,
          .st-guide-note {
            padding: 18px 20px;
            margin-bottom: 16px;
          }
          .st-guide-hero span {
            color: #0b6f76;
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
          }
          .st-guide-hero p,
          .st-guide-note p,
          .st-guide-grid p {
            color: #607083;
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
          }
          .st-guide-grid strong {
            color: #0b6f76;
          }
          @media (max-width: 900px) {
            .st-guide-grid {
              grid-template-columns: 1fr;
            }
          }
          code, pre {
            font-family: Consolas, "SFMono-Regular", "Courier New", monospace;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
