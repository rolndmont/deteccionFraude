"""
Fase 1 — Generador de datos sintéticos
Fraude con tarjeta: Clonación vs Card-Not-Present (CNP)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# SEMILLA para reproducibilidad
# ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# PARÁMETROS GENERALES
# ─────────────────────────────────────────────
N_USUARIOS       = 250    # número de usuarios en la simulación
N_TARJETAS       = 300    # algunas tarjetas están comprometidas
N_DISPOSITIVOS   = 40    # dispositivos registrados
N_IPS            = 550    # direcciones IP
N_LEGITIMAS      = 900   # transacciones legítimas
N_CLONACION      = 40    # transacciones por clonación
N_CNP            = 40    # transacciones por CNP

FECHA_INICIO = datetime(2024, 1, 1)
FECHA_FIN    = datetime(2024, 6, 30)


# ─────────────────────────────────────────────
# GENERADORES DE ENTIDADES BASE
# ─────────────────────────────────────────────
def generar_entidades():
    """Genera los catálogos de entidades del ecosistema de pagos."""

    usuarios     = [f"USR_{i:03d}" for i in range(1, N_USUARIOS + 1)]
    tarjetas     = [f"TRJ_{i:03d}" for i in range(1, N_TARJETAS + 1)]
    dispositivos = [f"DEV_{i:03d}" for i in range(1, N_DISPOSITIVOS + 1)]
    ips          = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
                    for _ in range(N_IPS)]

    # Cada usuario tiene dispositivos e IPs habituales
    perfil_usuario = {}
    for u in usuarios:
        perfil_usuario[u] = {
            "tarjetas"    : random.sample(tarjetas, k=random.randint(1, 2)),
            "dispositivos": random.sample(dispositivos, k=random.randint(1, 3)),
            "ips"         : random.sample(ips, k=random.randint(1, 2)),
            # Ubicación habitual del usuario (ciudad simulada)
            "lat_base"    : round(random.uniform(19.0, 25.0), 4),
            "lon_base"    : round(random.uniform(-102.0, -98.0), 4),
        }

    return usuarios, tarjetas, dispositivos, ips, perfil_usuario


def fecha_aleatoria(inicio, fin):
    """Genera una fecha aleatoria entre dos fechas."""
    delta = fin - inicio
    segundos = random.randint(0, int(delta.total_seconds()))
    return inicio + timedelta(seconds=segundos)


def ubicacion_cercana(lat_base, lon_base, radio_km=50):
    """Genera una ubicación cercana a la base del usuario (radio en km)."""
    delta_grados = radio_km / 111.0
    lat = round(lat_base + random.uniform(-delta_grados, delta_grados), 4)
    lon = round(lon_base + random.uniform(-delta_grados, delta_grados), 4)
    return lat, lon


def ubicacion_lejana(lat_base, lon_base):
    """Genera una ubicación lejana — simula geo-velocity en clonación."""
    # Se desplaza entre 500 y 2000 km
    delta_grados = random.uniform(5.0, 18.0)
    lat = round(lat_base + random.choice([-1, 1]) * delta_grados, 4)
    lon = round(lon_base + random.choice([-1, 1]) * delta_grados, 4)
    return lat, lon


# ─────────────────────────────────────────────
# GENERADORES POR TIPO DE TRANSACCIÓN
# ─────────────────────────────────────────────
def generar_legitimas(usuarios, perfil_usuario, n):
    """
    Transacciones legítimas:
    - Canal presencial o remoto con probabilidad similar
    - Dispositivo e IP habituales del usuario
    - Ubicación cercana a la base del usuario
    """
    registros = []
    for i in range(n):
        usuario = random.choice(usuarios)
        perfil  = perfil_usuario[usuario]
        tarjeta = random.choice(perfil["tarjetas"])
        canal   = random.choice(["presencial", "remoto"])
        lat, lon = ubicacion_cercana(perfil["lat_base"], perfil["lon_base"])

        registros.append({
            "id_transaccion" : f"TXN_L_{i:04d}",
            "id_tarjeta"     : tarjeta,
            "id_usuario"     : usuario,
            "id_dispositivo" : random.choice(perfil["dispositivos"]),
            "direccion_ip"   : random.choice(perfil["ips"]),
            "latitud"        : lat,
            "longitud"       : lon,
            "canal"          : canal,
            "timestamp"      : fecha_aleatoria(FECHA_INICIO, FECHA_FIN),
            "monto"          : round(random.uniform(50, 5000), 2),
            "etiqueta"       : "legitima"
        })
    return registros


def generar_clonacion(tarjetas, dispositivos, ips, perfil_usuario, usuarios, n):
    """
    Fraude por clonación (card-present):
    - Misma tarjeta aparece en dos ubicaciones muy lejanas en poco tiempo
    - Canal siempre presencial
    - Dispositivo e IP NO habituales del usuario legítimo
    - Señal principal: geo-velocity
    """
    registros = []

    # Seleccionar tarjetas comprometidas para clonación
    tarjetas_comprometidas = random.sample(tarjetas, k=min(10, len(tarjetas)))

    for i in range(n):
        tarjeta = random.choice(tarjetas_comprometidas)

        # Encontrar el usuario legítimo de esa tarjeta
        usuario_legitimo = None
        for u, p in perfil_usuario.items():
            if tarjeta in p["tarjetas"]:
                usuario_legitimo = u
                break
        if usuario_legitimo is None:
            usuario_legitimo = random.choice(usuarios)

        perfil = perfil_usuario[usuario_legitimo]

        # Ubicación legítima (donde debería estar el usuario)
        lat_leg, lon_leg = ubicacion_cercana(
            perfil["lat_base"], perfil["lon_base"]
        )

        # Ubicación fraudulenta (muy lejos)
        lat_fraud, lon_fraud = ubicacion_lejana(
            perfil["lat_base"], perfil["lon_base"]
        )

        # Timestamp de la transacción fraudulenta muy cercano a una legítima
        ts_base   = fecha_aleatoria(FECHA_INICIO, FECHA_FIN)
        # El fraude ocurre entre 5 y 60 minutos después
        ts_fraude = ts_base + timedelta(minutes=random.randint(5, 60))

        registros.append({
            "id_transaccion" : f"TXN_C_{i:04d}",
            "id_tarjeta"     : tarjeta,
            "id_usuario"     : usuario_legitimo,
            # Dispositivo e IP desconocidos para el usuario legítimo
            "id_dispositivo" : random.choice(
                [d for d in dispositivos
                 if d not in perfil["dispositivos"]] or dispositivos
            ),
            "direccion_ip"   : random.choice(
                [ip for ip in ips
                 if ip not in perfil["ips"]] or ips
            ),
            "latitud"        : lat_fraud,
            "longitud"       : lon_fraud,
            "canal"          : "presencial",   # siempre presencial
            "timestamp"      : ts_fraude,
            "monto"          : round(random.uniform(500, 8000), 2),
            "etiqueta"       : "clonacion"
        })
    return registros


def generar_cnp(tarjetas, dispositivos, ips, n):
    """
    Fraude Card-Not-Present (CNP):
    - Misma IP o dispositivo usados con múltiples tarjetas distintas
    - Canal siempre remoto
    - No hay señal geoespacial relevante
    - Señal principal: infraestructura digital compartida
    """
    registros = []

    # IPs y dispositivos comprometidos — usados por el defraudador
    ips_comprometidas  = random.sample(ips, k=min(5, len(ips)))
    devs_comprometidos = random.sample(dispositivos, k=min(5, len(dispositivos)))

    # Tarjetas usadas en CNP (robadas, no clonadas físicamente)
    tarjetas_cnp = random.sample(tarjetas, k=min(15, len(tarjetas)))

    for i in range(n):
        # El defraudador usa una IP o dispositivo comprometido
        ip_fraude  = random.choice(ips_comprometidas)
        dev_fraude = random.choice(devs_comprometidos)
        tarjeta    = random.choice(tarjetas_cnp)

        # Ubicación aproximada (no es señal diagnóstica en CNP)
        lat = round(random.uniform(19.0, 25.0), 4)
        lon = round(random.uniform(-102.0, -98.0), 4)

        registros.append({
            "id_transaccion" : f"TXN_P_{i:04d}",
            "id_tarjeta"     : tarjeta,
            # Usuario desconocido o inventado
            "id_usuario"     : f"USR_FRAUD_{i:03d}",
            "id_dispositivo" : dev_fraude,
            "direccion_ip"   : ip_fraude,
            "latitud"        : lat,
            "longitud"       : lon,
            "canal"          : "remoto",   # siempre remoto
            "timestamp"      : fecha_aleatoria(FECHA_INICIO, FECHA_FIN),
            "monto"          : round(random.uniform(100, 3000), 2),
            "etiqueta"       : "cnp"
        })
    return registros


# ─────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  FASE 1 — Generador de datos sintéticos")
    print("  Fraude con tarjeta: Clonación vs CNP")
    print("=" * 55)

    # 1. Generar entidades base
    print("\n[1/4] Generando entidades del ecosistema de pagos...")
    usuarios, tarjetas, dispositivos, ips, perfil_usuario = generar_entidades()
    print(f"      Usuarios    : {len(usuarios)}")
    print(f"      Tarjetas    : {len(tarjetas)}")
    print(f"      Dispositivos: {len(dispositivos)}")
    print(f"      IPs         : {len(ips)}")

    # 2. Generar transacciones por tipo
    print("\n[2/4] Generando transacciones...")
    legitimas  = generar_legitimas(usuarios, perfil_usuario, N_LEGITIMAS)
    clonacion  = generar_clonacion(
        tarjetas, dispositivos, ips, perfil_usuario, usuarios, N_CLONACION
    )
    cnp        = generar_cnp(tarjetas, dispositivos, ips, N_CNP)

    print(f"      Legítimas : {len(legitimas)}")
    print(f"      Clonación : {len(clonacion)}")
    print(f"      CNP       : {len(cnp)}")

    # 3. Combinar y ordenar cronológicamente
    print("\n[3/4] Combinando y ordenando datos...")
    todos = legitimas + clonacion + cnp
    df    = pd.DataFrame(todos)
    df    = df.sort_values("timestamp").reset_index(drop=True)

    # 4. Guardar a CSV
    print("\n[4/4] Guardando dataset...")
    ruta = "transacciones_sinteticas.csv"
    df.to_csv(ruta, index=False)

    # ─── Resumen final ───
    print("\n" + "=" * 55)
    print("  RESUMEN DEL DATASET GENERADO")
    print("=" * 55)
    print(f"\n  Total de transacciones : {len(df)}")
    print(f"\n  Distribución de etiquetas:")
    print(df["etiqueta"].value_counts().to_string())
    print(f"\n  Período               : {df['timestamp'].min().date()} "
          f"→ {df['timestamp'].max().date()}")
    print(f"\n  Columnas              : {list(df.columns)}")
    print(f"\n  Archivo guardado en   : {ruta}")
    print("\n  Muestra de 5 registros:")
    print(df[["id_transaccion", "id_tarjeta", "canal",
              "etiqueta", "timestamp"]].head(5).to_string(index=False))
    print("\n" + "=" * 55)
