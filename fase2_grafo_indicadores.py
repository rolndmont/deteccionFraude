"""
Fase 2 — Construcción del grafo heterogéneo y cálculo de indicadores
Fraude con tarjeta: Clonación vs Card-Not-Present (CNP)

Indicadores implementados según Tablas 1 y 2 de la Metodología:

Clonación (I1-I5):
  I1: Canal presencial (POS/ATM)
  I2: Terminal/dispositivo nuevo para la tarjeta
  I3: Desplazamiento geográfico improbable (geo-velocity)
  I4: Ruptura del patrón histórico
  I5: Concentración de fraude en punto físico

CNP (J1-J5):
  J1: Canal remoto
  J2: IP asociada a múltiples tarjetas
  J3: Dispositivo usado con varias tarjetas
  J4: Identidad remota reutilizada
  J5: Alta tasa de fraude en infraestructura digital
"""

import pandas as pd
import numpy as np
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

# ─────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# (corresponden a los parámetros de la Metodología)
# ─────────────────────────────────────────────
VELOCIDAD_MAX_KMH = 900    # velocidad máxima humana realista (avión)
NC = 3    # factor normalización: tarjetas por IP
ND = 3    # factor normalización: tarjetas por dispositivo
NA = 3    # factor normalización: tarjetas por cuenta/usuario


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────
def distancia_km(lat1, lon1, lat2, lon2):
    """Calcula distancia en km entre dos coordenadas (fórmula Haversine)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def velocidad_kmh(dist_km, ts1, ts2):
    """Calcula velocidad en km/h entre dos transacciones."""
    delta_h = abs((ts2 - ts1).total_seconds()) / 3600.0
    if delta_h < 1e-6:
        return float('inf')
    return dist_km / delta_h


# ─────────────────────────────────────────────
# PARTE A — CONSTRUCCIÓN DEL GRAFO HETEROGÉNEO
# ─────────────────────────────────────────────
def construir_grafo(df):
    """
    Construye GH = (V, E, φ, ψ) a partir del DataFrame de transacciones.

    Tipos de vértices (φ):
      - tarjeta, usuario, dispositivo, ip, transaccion

    Tipos de aristas (ψ):
      - usa_tarjeta   : usuario → tarjeta
      - usa_dispositivo: usuario → dispositivo
      - usa_ip        : usuario → ip
      - realiza       : tarjeta → transaccion
      - desde_dispositivo: transaccion → dispositivo
      - desde_ip      : transaccion → ip
    """
    G = nx.MultiDiGraph()

    for _, row in df.iterrows():
        txn_id = row["id_transaccion"]
        usr_id = row["id_usuario"]
        trj_id = row["id_tarjeta"]
        dev_id = row["id_dispositivo"]
        ip_id  = f"IP_{row['direccion_ip'].replace('.', '_')}"

        # ── Agregar vértices con su tipo (φ) ──
        for nodo, tipo in [
            (txn_id, "transaccion"),
            (usr_id, "usuario"),
            (trj_id, "tarjeta"),
            (dev_id, "dispositivo"),
            (ip_id,  "ip"),
        ]:
            if not G.has_node(nodo):
                G.add_node(nodo, tipo=tipo)

        # Atributos extra en el vértice transacción
        G.nodes[txn_id].update({
            "canal"    : row["canal"],
            "timestamp": row["timestamp"],
            "monto"    : row["monto"],
            "latitud"  : row["latitud"],
            "longitud" : row["longitud"],
            "etiqueta" : row["etiqueta"],
        })

        # ── Agregar aristas con su tipo (ψ) ──
        ts = row["timestamp"]
        G.add_edge(usr_id, trj_id,    tipo="usa_tarjeta",       timestamp=ts)
        G.add_edge(usr_id, dev_id,    tipo="usa_dispositivo",   timestamp=ts)
        G.add_edge(usr_id, ip_id,     tipo="usa_ip",            timestamp=ts)
        G.add_edge(trj_id, txn_id,    tipo="realiza",           timestamp=ts,
                   canal=row["canal"], monto=row["monto"])
        G.add_edge(txn_id, dev_id,    tipo="desde_dispositivo", timestamp=ts)
        G.add_edge(txn_id, ip_id,     tipo="desde_ip",          timestamp=ts)

    return G


def estadisticas_grafo(G):
    """Imprime estadísticas básicas del grafo construido."""
    tipos_nodos = defaultdict(int)
    for _, data in G.nodes(data=True):
        tipos_nodos[data.get("tipo", "desconocido")] += 1

    tipos_aristas = defaultdict(int)
    for _, _, data in G.edges(data=True):
        tipos_aristas[data.get("tipo", "desconocido")] += 1

    print(f"\n  Vértices totales : {G.number_of_nodes()}")
    print(f"  Aristas totales  : {G.number_of_edges()}")
    print(f"\n  Tipos de vértices (φ):")
    for t, c in sorted(tipos_nodos.items()):
        print(f"    {t:<15} : {c}")
    print(f"\n  Tipos de aristas (ψ):")
    for t, c in sorted(tipos_aristas.items()):
        print(f"    {t:<20} : {c}")


# ─────────────────────────────────────────────
# PARTE B — CÁLCULO DE INDICADORES
# ─────────────────────────────────────────────

# ── Indicadores de Clonación (I1 - I5) ──

def calcular_I1(row):
    """I1: Canal presencial (POS/ATM) → señal de presencialidad física."""
    return 1 if row["canal"] == "presencial" else 0


def calcular_I2(row, df):
    """
    I2: Dispositivo nuevo para la tarjeta.
    Verifica si el dispositivo usado fue visto antes con esa tarjeta.
    """
    tarjeta  = row["id_tarjeta"]
    dev      = row["id_dispositivo"]
    ts_actual = row["timestamp"]

    # Transacciones anteriores de esa tarjeta
    hist = df[
        (df["id_tarjeta"] == tarjeta) &
        (df["timestamp"] < ts_actual)
    ]
    dispositivos_conocidos = set(hist["id_dispositivo"].tolist())
    return 1 if dev not in dispositivos_conocidos else 0


def calcular_I3(row, df):
    """
    I3: Desplazamiento geográfico improbable (geo-velocity).
    Calcula la velocidad necesaria para ir desde la última transacción
    hasta esta. Si supera VELOCIDAD_MAX_KMH → anomalía.
    """
    tarjeta   = row["id_tarjeta"]
    ts_actual = row["timestamp"]
    lat_act   = row["latitud"]
    lon_act   = row["longitud"]

    # Transacción más reciente anterior de la misma tarjeta
    hist = df[
        (df["id_tarjeta"] == tarjeta) &
        (df["timestamp"] < ts_actual)
    ].sort_values("timestamp")

    if hist.empty:
        return 0

    ultima = hist.iloc[-1]
    dist   = distancia_km(
        ultima["latitud"], ultima["longitud"], lat_act, lon_act
    )
    vel    = velocidad_kmh(dist, ultima["timestamp"], ts_actual)

    return 1 if vel > VELOCIDAD_MAX_KMH else 0


def calcular_I4(row, df):
    """
    I4: Ruptura del patrón histórico.
    Verifica si el canal de esta transacción difiere del canal
    más frecuente de la tarjeta en su historial.
    """
    tarjeta   = row["id_tarjeta"]
    ts_actual = row["timestamp"]

    hist = df[
        (df["id_tarjeta"] == tarjeta) &
        (df["timestamp"] < ts_actual)
    ]

    if hist.empty:
        return 0

    canal_frecuente = hist["canal"].mode()
    if canal_frecuente.empty:
        return 0

    return 1 if row["canal"] != canal_frecuente.iloc[0] else 0


def calcular_I5(row, df):
    """
    I5: Concentración de fraude en punto físico.
    Proporción de transacciones fraudulentas en el mismo dispositivo.
    Solo aplicable a transacciones etiquetadas (para validación).
    """
    dev = row["id_dispositivo"]

    txns_en_dev = df[df["id_dispositivo"] == dev]
    total       = len(txns_en_dev)

    if total == 0:
        return 0

    fraudes = len(txns_en_dev[txns_en_dev["etiqueta"] == "clonacion"])
    return round(fraudes / total, 4)


# ── Indicadores CNP (J1 - J5) ──

def calcular_J1(row):
    """J1: Canal remoto → señal de no presencialidad."""
    return 1 if row["canal"] == "remoto" else 0


def calcular_J2(row, df):
    """
    J2: IP asociada a múltiples tarjetas.
    J2 = número de tarjetas distintas que usan esa IP / NC
    Normalizado y acotado a [0, 1].
    """
    ip = row["direccion_ip"]
    tarjetas_en_ip = df[df["direccion_ip"] == ip]["id_tarjeta"].nunique()
    return min(round(tarjetas_en_ip / NC, 4), 1.0)


def calcular_J3(row, df):
    """
    J3: Dispositivo usado con varias tarjetas.
    J3 = número de tarjetas distintas en ese dispositivo / ND
    Normalizado y acotado a [0, 1].
    """
    dev = row["id_dispositivo"]
    tarjetas_en_dev = df[df["id_dispositivo"] == dev]["id_tarjeta"].nunique()
    return min(round(tarjetas_en_dev / ND, 4), 1.0)


def calcular_J4(row, df):
    """
    J4: Identidad remota reutilizada.
    J4 = número de tarjetas distintas asociadas al mismo usuario / NA
    Normalizado y acotado a [0, 1].
    """
    usuario = row["id_usuario"]
    tarjetas_usuario = df[df["id_usuario"] == usuario]["id_tarjeta"].nunique()
    return min(round(tarjetas_usuario / NA, 4), 1.0)


def calcular_J5(row, df):
    """
    J5: Alta tasa de fraude en infraestructura digital compartida.
    Proporción de transacciones CNP en el mismo dispositivo o IP.
    """
    ip  = row["direccion_ip"]
    dev = row["id_dispositivo"]

    infraestructura = df[
        (df["direccion_ip"] == ip) | (df["id_dispositivo"] == dev)
    ]
    total = len(infraestructura)

    if total == 0:
        return 0

    fraudes_cnp = len(infraestructura[infraestructura["etiqueta"] == "cnp"])
    return round(fraudes_cnp / total, 4)


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE CÁLCULO
# ─────────────────────────────────────────────
def calcular_todos_indicadores(df):
    """Calcula los 10 indicadores para cada transacción del dataset."""

    print("\n  Calculando indicadores estructurales...")
    resultados = []

    for idx, row in df.iterrows():
        indicadores = {
            "id_transaccion": row["id_transaccion"],
            "etiqueta"      : row["etiqueta"],
            # Indicadores de Clonación
            "I1": calcular_I1(row),
            "I2": calcular_I2(row, df),
            "I3": calcular_I3(row, df),
            "I4": calcular_I4(row, df),
            "I5": calcular_I5(row, df),
            # Indicadores CNP
            "J1": calcular_J1(row),
            "J2": calcular_J2(row, df),
            "J3": calcular_J3(row, df),
            "J4": calcular_J4(row, df),
            "J5": calcular_J5(row, df),
        }
        resultados.append(indicadores)

        # Mostrar progreso cada 100 transacciones
        if (idx + 1) % 100 == 0:
            print(f"    Procesadas: {idx + 1}/{len(df)}")

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  FASE 2 — Grafo heterogéneo e Indicadores")
    print("=" * 55)

    # ── Cargar dataset de Fase 1 ──
    print("\n[1/4] Cargando dataset de Fase 1...")
    df = pd.read_csv("transacciones_sinteticas.csv",
                     parse_dates=["timestamp"])
    print(f"      {len(df)} transacciones cargadas.")

    # ── Construir grafo ──
    print("\n[2/4] Construyendo grafo heterogéneo GH = (V, E, φ, ψ)...")
    G = construir_grafo(df)
    estadisticas_grafo(G)

    # ── Calcular indicadores ──
    print("\n[3/4] Calculando indicadores estructurales (I1-I5, J1-J5)...")
    df_ind = calcular_todos_indicadores(df)
    print(f"      Indicadores calculados para {len(df_ind)} transacciones.")

    # ── Guardar resultados ──
    print("\n[4/4] Guardando resultados...")
    ruta = "indicadores_calculados.csv"
    df_ind.to_csv(ruta, index=False)

    # ── Resumen por tipo de fraude ──
    print("\n" + "=" * 55)
    print("  RESUMEN DE INDICADORES POR TIPO DE TRANSACCIÓN")
    print("=" * 55)

    cols_ind = ["I1","I2","I3","I4","I5","J1","J2","J3","J4","J5"]
    resumen  = df_ind.groupby("etiqueta")[cols_ind].mean().round(3)
    print(f"\n  Promedio de indicadores por etiqueta:\n")
    print(resumen.to_string())

    print("\n" + "─" * 55)
    print("  INTERPRETACIÓN ESPERADA:")
    print("  · Clonación → I1, I2, I3, I4 altos; J1 bajo")
    print("  · CNP       → J1, J2, J3, J4 altos; I1 bajo")
    print("  · Legítima  → todos los indicadores bajos")
    print("─" * 55)
    print(f"\n  Archivo guardado en: {ruta}")
    print("=" * 55)
