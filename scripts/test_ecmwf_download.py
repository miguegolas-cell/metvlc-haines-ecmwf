
from pathlib import Path
from ecmwf.opendata import Client

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

target = DATA_DIR / "test_ecmwf_haines.grib2"

client = Client(source="ecmwf")

client.retrieve(
    date=0,
    time=0,
    step=0,
    type="fc",
    levtype="pl",
    levelist=[850, 700, 500],
    param=["t", "r"],
    target=str(target),
)

print(f"Archivo descargado: {target}")
print(f"Tamaño: {target.stat().st_size} bytes")
