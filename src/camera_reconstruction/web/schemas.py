from __future__ import annotations

from dataclasses import dataclass

from camera_reconstruction.services.reconstruction_service import ReconstructionConfig


DEFAULT_FORM_VALUES = {
    "corners_x": "9",
    "corners_y": "6",
    "square_size": "1.0",
    "fx": "",
    "fy": "",
    "cx": "",
    "cy": "",
    "skew": "0.0",
    "distortion_coefficients": "",
    "axis_length": "",
}


@dataclass(frozen=True)
class ReconstructionForm:
    corners_x: int
    corners_y: int
    square_size: float
    fx: float | None
    fy: float | None
    cx: float | None
    cy: float | None
    skew: float
    distortion_coefficients: tuple[float, ...]
    axis_length: float | None

    @classmethod
    def from_raw(cls, values: dict[str, str | None]) -> "ReconstructionForm":
        return cls(
            corners_x=_parse_int(values.get("corners_x"), "corners_x", minimum=2),
            corners_y=_parse_int(values.get("corners_y"), "corners_y", minimum=2),
            square_size=_parse_float(values.get("square_size"), "square_size", minimum=0.0, strict_min=True),
            fx=_parse_optional_float(values.get("fx"), "fx", minimum=0.0, strict_min=True),
            fy=_parse_optional_float(values.get("fy"), "fy", minimum=0.0, strict_min=True),
            cx=_parse_optional_float(values.get("cx"), "cx"),
            cy=_parse_optional_float(values.get("cy"), "cy"),
            skew=_parse_optional_float(values.get("skew"), "skew") or 0.0,
            distortion_coefficients=_parse_float_list(values.get("distortion_coefficients")),
            axis_length=_parse_optional_float(values.get("axis_length"), "axis_length", minimum=0.0, strict_min=True),
        )

    def to_config(self) -> ReconstructionConfig:
        return ReconstructionConfig(
            corners_x=self.corners_x,
            corners_y=self.corners_y,
            square_size=self.square_size,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            skew=self.skew,
            distortion_coefficients=self.distortion_coefficients,
            axis_length=self.axis_length,
        )


def form_values(**overrides: str | None) -> dict[str, str]:
    values = DEFAULT_FORM_VALUES.copy()
    for key, value in overrides.items():
        values[key] = "" if value is None else str(value)
    return values


def _parse_int(value: str | None, name: str, minimum: int | None = None) -> int:
    if value is None or value.strip() == "":
        raise ValueError(f"{name} is required")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _parse_float(
    value: str | None,
    name: str,
    minimum: float | None = None,
    strict_min: bool = False,
) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"{name} is required")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if minimum is not None:
        if strict_min and parsed <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
        if not strict_min and parsed < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _parse_optional_float(
    value: str | None,
    name: str,
    minimum: float | None = None,
    strict_min: bool = False,
) -> float | None:
    if value is None or value.strip() == "":
        return None
    return _parse_float(value, name, minimum=minimum, strict_min=strict_min)


def _parse_float_list(value: str | None) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        return ()

    normalized = value.replace(";", ",").replace("\n", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("distortion_coefficients must be comma-separated numbers") from exc

