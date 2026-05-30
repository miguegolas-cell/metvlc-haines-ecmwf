from pathlib import Path
from datetime import datetime, timezone
import json
import math

import numpy as np
import xarray as xr
from shapely.geometry import shape, Point


DATA_DIR = Path("data")
DOCS_DIR = Path("docs")

GRIB_FILE = DATA_DIR / "test_ecmwf_haines.grib2"
CV_GEOJSON = DOCS_DIR / "cv.geojson"

OUT_GEOJSON = DOCS_DIR / "haines.geojson"
OUT_METADATA = DOCS_DIR / "metadata.json"


def saturation_vapor_pressure_hpa(temp_c):
    """
    Presión de vapor de saturación en hPa.
    Fórmula de Magnus.
    """
    return 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))


def dewpoint_from_temp_rh(temp_c, rh):
    """
    Calcula punto de rocío en ºC a partir de temperatura ºC y HR %.
    """
    rh = max(1.0, min(100.0, rh))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)


def haines_stability_700_500(t700_c, t500_c):
    """
    Componente A del Haines alto:
    T700 - T500
    """
    diff = t700_c - t500_c

    if diff <= 17:
        return 1
    elif diff <= 21:
        return 2
    else:
        return 3


def haines_moisture_700(t700_c, td700_c):
    """
    Componente B del Haines alto:
    depresión del punto de rocío en 700 hPa.
    """
    depression = t700_c - td700_c

    if depression <= 14:
        return 1
    elif depression <= 20:
        return 2
    else:
        return 3


def haines_level(total):
    if total <= 3:
        return "Bajo"
    elif total <= 5:
        return "Moderado"
    else:
        return "Alto"


def haines_color(total):
    if total <= 3:
        return "#2b83ba"
    elif total == 4:
        return "#abdda4"
    elif total == 5:
        return "#fdae61"
    else:
        return "#d7191c"


def load_cv_geometry():
    if not CV_GEOJSON.exists():
        raise FileNotFoundError("No existe docs/cv.geojson")

    data = json.loads(CV_GEOJSON.read_text(encoding="utf-8"))

    if data["type"] == "FeatureCollection":
        geoms = [shape(feature["geometry"]) for feature in data["features"]]
    elif data["type"] == "Feature":
        geoms = [shape(data["geometry"])]
    else:
        geoms = [shape(data)]

    return geoms


def point_inside_any(lon, lat, geometries):
    p = Point(lon, lat)
    return any(g.contains(p) or g.touches(p) for g in geometries)


def main():
    if not GRIB_FILE.exists():
        raise FileNotFoundError(f"No existe {GRIB_FILE}")

    print("Abriendo GRIB ECMWF...")
    ds = xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    print(ds)

    if "t" not in ds:
        raise ValueError("No encuentro la variable de temperatura 't'")

    if "r" not in ds:
        raise ValueError("No encuentro la variable de humedad relativa 'r'")

    levels = list(ds["isobaricInhPa"].values)

    needed = [700, 500]

    for level in needed:
        if level not in levels:
            raise ValueError(f"No encuentro el nivel {level} hPa en el GRIB. Niveles disponibles: {levels}")

    cv_geoms = load_cv_geometry()

    t700 = ds["t"].sel(isobaricInhPa=700)
    t500 = ds["t"].sel(isobaricInhPa=500)
    r700 = ds["r"].sel(isobaricInhPa=700)

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    features = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            lon_f = float(lon)
            lat_f = float(lat)

            if not point_inside_any(lon_f, lat_f, cv_geoms):
                continue

            t700_c = float(t700.values[i, j]) - 273.15
            t500_c = float(t500.values[i, j]) - 273.15
            rh700 = float(r700.values[i, j])

            if np.isnan(t700_c) or np.isnan(t500_c) or np.isnan(rh700):
                continue

            td700_c = dewpoint_from_temp_rh(t700_c, rh700)

            a = haines_stability_700_500(t700_c, t500_c)
            b = haines_moisture_700(t700_c, td700_c)
            total = a + b

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon_f, lat_f]
                },
                "properties": {
                    "haines": int(total),
                    "nivel": haines_level(total),
                    "color": haines_color(total),
                    "A_estabilidad": int(a),
                    "B_sequedad": int(b),
                    "T700_C": round(t700_c, 1),
                    "T500_C": round(t500_c, 1),
                    "HR700": round(rh700, 0),
                    "Td700_C": round(td700_c, 1),
                    "depresion_700": round(t700_c - td700_c, 1)
                }
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    OUT_GEOJSON.write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    valid_time = ""

    if "valid_time" in ds:
        try:
            valid_time = str(ds["valid_time"].values)
        except Exception:
            valid_time = ""

    metadata = {
        "producto": "Índice de Haines ECMWF",
        "fuente": "ECMWF Open Data",
        "tipo": "Haines alto: estabilidad 700-500 hPa + sequedad 700 hPa",
        "archivo": str(GRIB_FILE),
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "valid_time": valid_time,
        "puntos": len(features),
        "variables": ["t", "r"],
        "niveles": [700, 500],
        "nota": "Cálculo experimental a partir de temperatura y humedad relativa ECMWF."
    }

    OUT_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Puntos generados: {len(features)}")
    print(f"Archivo: {OUT_GEOJSON}")
    print(f"Archivo: {OUT_METADATA}")


if __name__ == "__main__":
    main()
