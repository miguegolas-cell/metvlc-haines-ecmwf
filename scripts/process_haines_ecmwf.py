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


def dewpoint_from_temp_rh(temp_c, rh):
    """
    Calcula punto de rocío en ºC a partir de temperatura ºC y HR %.
    """
    rh = max(1.0, min(100.0, rh))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)


def haines_bajo_A(t950_c, t850_c):
    """
    Haines bajo:
    A = T950 - T850
    """
    diff = t950_c - t850_c

    if diff <= 3:
        return 1
    elif diff <= 7:
        return 2
    else:
        return 3


def haines_bajo_B(t850_c, td850_c):
    """
    Haines bajo:
    B = T850 - Td850
    """
    depression = t850_c - td850_c

    if depression <= 5:
        return 1
    elif depression <= 9:
        return 2
    else:
        return 3


def haines_medio_A(t850_c, t700_c):
    """
    Haines medio:
    A = T850 - T700
    """
    diff = t850_c - t700_c

    if diff <= 5:
        return 1
    elif diff <= 10:
        return 2
    else:
        return 3


def haines_medio_B(t850_c, td850_c):
    """
    Haines medio:
    B = T850 - Td850
    """
    depression = t850_c - td850_c

    if depression <= 5:
        return 1
    elif depression <= 12:
        return 2
    else:
        return 3


def haines_alto_A(t700_c, t500_c):
    """
    Haines alto:
    A = T700 - T500
    """
    diff = t700_c - t500_c

    if diff <= 17:
        return 1
    elif diff <= 21:
        return 2
    else:
        return 3


def haines_alto_B(t700_c, td700_c):
    """
    Haines alto:
    B = T700 - Td700
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


def get_level(ds, varname, level):
    return ds[varname].sel(isobaricInhPa=level)


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

    levels = [int(x) for x in ds["isobaricInhPa"].values]

    needed = [950, 850, 700, 500]

    for level in needed:
        if level not in levels:
            raise ValueError(
                f"No encuentro el nivel {level} hPa en el GRIB. "
                f"Niveles disponibles: {levels}"
            )

    cv_geoms = load_cv_geometry()

    t950 = get_level(ds, "t", 950)
    t850 = get_level(ds, "t", 850)
    t700 = get_level(ds, "t", 700)
    t500 = get_level(ds, "t", 500)

    r850 = get_level(ds, "r", 850)
    r700 = get_level(ds, "r", 700)

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    features = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            lon_f = float(lon)
            lat_f = float(lat)

            if not point_inside_any(lon_f, lat_f, cv_geoms):
                continue

            t950_c = float(t950.values[i, j]) - 273.15
            t850_c = float(t850.values[i, j]) - 273.15
            t700_c = float(t700.values[i, j]) - 273.15
            t500_c = float(t500.values[i, j]) - 273.15

            rh850 = float(r850.values[i, j])
            rh700 = float(r700.values[i, j])

            values = [t950_c, t850_c, t700_c, t500_c, rh850, rh700]

            if any(np.isnan(v) for v in values):
                continue

            td850_c = dewpoint_from_temp_rh(t850_c, rh850)
            td700_c = dewpoint_from_temp_rh(t700_c, rh700)

            bajo_A = haines_bajo_A(t950_c, t850_c)
            bajo_B = haines_bajo_B(t850_c, td850_c)
            haines_bajo = bajo_A + bajo_B

            medio_A = haines_medio_A(t850_c, t700_c)
            medio_B = haines_medio_B(t850_c, td850_c)
            haines_medio = medio_A + medio_B

            alto_A = haines_alto_A(t700_c, t500_c)
            alto_B = haines_alto_B(t700_c, td700_c)
            haines_alto = alto_A + alto_B

            # Valor principal por defecto para mantener compatible el visor actual
            haines_principal = haines_alto

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon_f, lat_f]
                },
                "properties": {
                    "haines": int(haines_principal),
                    "nivel": haines_level(haines_principal),
                    "color": haines_color(haines_principal),

                    "haines_bajo": int(haines_bajo),
                    "haines_bajo_nivel": haines_level(haines_bajo),
                    "haines_bajo_color": haines_color(haines_bajo),
                    "bajo_A_estabilidad": int(bajo_A),
                    "bajo_B_sequedad": int(bajo_B),

                    "haines_medio": int(haines_medio),
                    "haines_medio_nivel": haines_level(haines_medio),
                    "haines_medio_color": haines_color(haines_medio),
                    "medio_A_estabilidad": int(medio_A),
                    "medio_B_sequedad": int(medio_B),

                    "haines_alto": int(haines_alto),
                    "haines_alto_nivel": haines_level(haines_alto),
                    "haines_alto_color": haines_color(haines_alto),
                    "alto_A_estabilidad": int(alto_A),
                    "alto_B_sequedad": int(alto_B),

                    "T950_C": round(t950_c, 1),
                    "T850_C": round(t850_c, 1),
                    "T700_C": round(t700_c, 1),
                    "T500_C": round(t500_c, 1),
                    "HR850": round(rh850, 0),
                    "HR700": round(rh700, 0),
                    "Td850_C": round(td850_c, 1),
                    "Td700_C": round(td700_c, 1),
                    "depresion_850": round(t850_c - td850_c, 1),
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
        "tipo": "Haines bajo, medio y alto",
        "archivo": str(GRIB_FILE),
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "valid_time": valid_time,
        "puntos": len(features),
        "variables": ["t", "r"],
        "niveles": [950, 850, 700, 500],
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
