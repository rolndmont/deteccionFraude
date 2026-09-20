"""
Implementación completa del Algoritmo 1
Fraude con tarjeta: Clonación vs Card-Not-Present (CNP)

Sigue exactamente la sección 3.6 de la Metodología:
  1. Construir GH = (V, E, φ, ψ)
  2. Para cada ti: extraer subgrafo Gi de profundidad k=2
  3. Calcular I=[I1..I5] y J=[J1..J5] SOBRE Gi
  4. Calcular S_CLON y S_CNP
  5. Aplicar regla de decisión con margen δ
  6. Registrar resultado y explicación estructural
  7. Extensión: entrenar perceptrón con los indicadores
"""

import pandas as pd
import numpy as np
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score


# ─────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# ─────────────────────────────────────────────
K             = 2       # profundidad del subgrafo local
DELTA         = 0.15    # margen de decisión δ
VELOCIDAD_MAX = 900     # km/h — umbral geo-velocity
NC            = 3       # normalización: tarjetas por IP
ND            = 3       # normalización: tarjetas por dispositivo
NA            = 3       # normalización: tarjetas por usuario

# Pesos manuales α (Clonación) — deben sumar 1
ALPHA = {"I1": 0.35, "I2": 0.25, "I3": 0.20, "I4": 0.10, "I5": 0.10}

# Pesos manuales β (CNP) — deben sumar 1
BETA  = {"J1": 0.35, "J2": 0.25, "J3": 0.20, "J4": 0.10, "J5": 0.10}


# ─────────────────────────────────────────────
# PASO 0 — UTILIDADES
# ─────────────────────────────────────────────
def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia entre dos coordenadas (Haversine)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def velocidad_kmh(dist_km, ts1, ts2):
    """Velocidad en km/h entre dos eventos."""
    delta_h = abs((ts2 - ts1).total_seconds()) / 3600.0
    return float('inf') if delta_h < 1e-6 else dist_km / delta_h


# ─────────────────────────────────────────────
# PASO 1 — CONSTRUIR EL GRAFO HETEROGÉNEO GH
# ─────────────────────────────────────────────
def construir_grafo(df):
    """
    Construye GH = (V, E, φ, ψ) desde el DataFrame.

    Tipos de vértices (φ):
      transaccion, tarjeta, usuario, dispositivo, ip

    Tipos de aristas (ψ):
      realiza, usa_tarjeta, usa_dispositivo, usa_ip,
      desde_dispositivo, desde_ip
    """
    G = nx.MultiDiGraph()

    for _, row in df.iterrows():
        txn = row["id_transaccion"]
        usr = row["id_usuario"]
        trj = row["id_tarjeta"]
        dev = row["id_dispositivo"]
        ip  = f"IP_{row['direccion_ip'].replace('.','_')}"
        ts  = row["timestamp"]

        # ── Vértices con tipo φ ──
        nodos = [
            (txn, {"tipo": "transaccion",
                   "canal": row["canal"],
                   "timestamp": ts,
                   "monto": row["monto"],
                   "latitud": row["latitud"],
                   "longitud": row["longitud"],
                   "etiqueta": row["etiqueta"]}),
            (usr, {"tipo": "usuario"}),
            (trj, {"tipo": "tarjeta"}),
            (dev, {"tipo": "dispositivo"}),
            (ip,  {"tipo": "ip"}),
        ]
        for nodo_id, attrs in nodos:
            if not G.has_node(nodo_id):
                G.add_node(nodo_id, **attrs)

        # ── Aristas con tipo ψ ──
        aristas = [
            (trj, txn, {"tipo": "realiza",           "timestamp": ts}),
            (usr, trj, {"tipo": "usa_tarjeta",        "timestamp": ts}),
            (usr, dev, {"tipo": "usa_dispositivo",    "timestamp": ts}),
            (usr, ip,  {"tipo": "usa_ip",             "timestamp": ts}),
            (txn, dev, {"tipo": "desde_dispositivo",  "timestamp": ts}),
            (txn, ip,  {"tipo": "desde_ip",           "timestamp": ts}),
        ]
        for u, v, attrs in aristas:
            G.add_edge(u, v, **attrs)

    return G


# ─────────────────────────────────────────────
# PASO 2 — EXTRAER SUBGRAFO LOCAL Gi (k=2)
# ─────────────────────────────────────────────
def extraer_subgrafo(G, txn_id, k=2):
    """
    Extrae el subgrafo local de profundidad k desde el nodo txn_id.

    Usa BFS sobre el grafo no dirigido subyacente para capturar
    vecinos en todas las direcciones (entrada y salida de aristas).

    Retorna un subgrafo de NetworkX con todos los nodos y aristas
    dentro de distancia k desde txn_id.
    """
    if txn_id not in G:
        return nx.MultiDiGraph()

    # BFS hasta profundidad k sobre versión no dirigida
    G_undir = G.to_undirected()
    nodos_vecinos = nx.single_source_shortest_path_length(
        G_undir, txn_id, cutoff=k
    )
    nodos_en_subgrafo = set(nodos_vecinos.keys())

    # Subgrafo inducido: conserva solo nodos y aristas dentro del vecindario
    subgrafo = G.subgraph(nodos_en_subgrafo).copy()
    return subgrafo


def inspeccionar_subgrafo(subgrafo, txn_id):
    """Muestra la composición del subgrafo extraído."""
    tipos = defaultdict(list)
    for nodo, data in subgrafo.nodes(data=True):
        tipos[data.get("tipo", "?")].append(nodo)

    print(f"\n    Subgrafo de {txn_id} (k={K}):")
    for tipo, nodos in tipos.items():
        print(f"      {tipo:<15}: {len(nodos)} nodos")
    print(f"      {'aristas':<15}: {subgrafo.number_of_edges()}")


# ─────────────────────────────────────────────
# PASO 3A — INDICADORES DE CLONACIÓN (I1-I5)
# Calculados SOBRE el subgrafo Gi
# ─────────────────────────────────────────────
def calcular_I1(txn_id, subgrafo):
    """
    I1: Canal presencial.
    Verifica el atributo 'canal' del nodo transacción en el subgrafo.
    """
    data = subgrafo.nodes.get(txn_id, {})
    return 1 if data.get("canal") == "presencial" else 0


def calcular_I2(txn_id, subgrafo):
    """
    I2: Dispositivo nuevo para la tarjeta.
    Dentro del subgrafo, verifica si el dispositivo de esta transacción
    apareció antes con la misma tarjeta.
    """
    # Encontrar tarjeta de esta transacción en el subgrafo
    tarjeta = None
    for u, v, data in subgrafo.edges(data=True):
        if v == txn_id and data.get("tipo") == "realiza":
            tarjeta = u
            break

    if tarjeta is None:
        return 0

    # Dispositivo de esta transacción
    dev_actual = None
    ts_actual  = subgrafo.nodes[txn_id].get("timestamp")
    for u, v, data in subgrafo.edges(data=True):
        if u == txn_id and data.get("tipo") == "desde_dispositivo":
            dev_actual = v
            break

    if dev_actual is None:
        return 0

    # Transacciones anteriores de esa tarjeta dentro del subgrafo
    txns_previas = [
        n for n, d in subgrafo.nodes(data=True)
        if d.get("tipo") == "transaccion"
        and n != txn_id
        and d.get("timestamp") is not None
        and d.get("timestamp") < ts_actual
    ]

    # Dispositivos vistos en esas transacciones previas
    devs_conocidos = set()
    for txn_prev in txns_previas:
        for u, v, data in subgrafo.edges(data=True):
            if u == txn_prev and data.get("tipo") == "desde_dispositivo":
                devs_conocidos.add(v)

    return 1 if dev_actual not in devs_conocidos else 0


def calcular_I3(txn_id, subgrafo):
    """
    I3: Geo-velocity — desplazamiento geográfico improbable.
    Compara la ubicación de esta transacción con la más reciente
    de la misma tarjeta dentro del subgrafo.
    """
    ts_actual  = subgrafo.nodes[txn_id].get("timestamp")
    lat_actual = subgrafo.nodes[txn_id].get("latitud")
    lon_actual = subgrafo.nodes[txn_id].get("longitud")

    if any(v is None for v in [ts_actual, lat_actual, lon_actual]):
        return 0

    # Tarjeta de esta transacción
    tarjeta = None
    for u, v, data in subgrafo.edges(data=True):
        if v == txn_id and data.get("tipo") == "realiza":
            tarjeta = u
            break

    if tarjeta is None:
        return 0

    # Transacciones previas de esa tarjeta en el subgrafo
    txns_previas = []
    for u, v, data in subgrafo.edges(data=True):
        if u == tarjeta and data.get("tipo") == "realiza":
            ts_prev = subgrafo.nodes[v].get("timestamp")
            if ts_prev is not None and ts_prev < ts_actual:
                txns_previas.append(v)

    if not txns_previas:
        return 0

    # Ordenar y tomar la más reciente
    txns_previas.sort(
        key=lambda x: subgrafo.nodes[x].get("timestamp")
    )
    ultima = txns_previas[-1]

    lat_prev = subgrafo.nodes[ultima].get("latitud")
    lon_prev = subgrafo.nodes[ultima].get("longitud")
    ts_prev  = subgrafo.nodes[ultima].get("timestamp")

    if any(v is None for v in [lat_prev, lon_prev, ts_prev]):
        return 0

    dist = distancia_km(lat_prev, lon_prev, lat_actual, lon_actual)
    vel  = velocidad_kmh(dist, ts_prev, ts_actual)

    return 1 if vel > VELOCIDAD_MAX else 0


def calcular_I4(txn_id, subgrafo):
    """
    I4: Ruptura del patrón histórico.
    Compara el canal de esta transacción con el canal más frecuente
    de la tarjeta en el historial visible dentro del subgrafo.
    """
    canal_actual = subgrafo.nodes[txn_id].get("canal")
    ts_actual    = subgrafo.nodes[txn_id].get("timestamp")

    tarjeta = None
    for u, v, data in subgrafo.edges(data=True):
        if v == txn_id and data.get("tipo") == "realiza":
            tarjeta = u
            break

    if tarjeta is None:
        return 0

    # Canales históricos de la tarjeta en el subgrafo
    canales_hist = []
    for u, v, data in subgrafo.edges(data=True):
        if u == tarjeta and data.get("tipo") == "realiza" and v != txn_id:
            ts_v = subgrafo.nodes[v].get("timestamp")
            if ts_v is not None and ts_v < ts_actual:
                canal_v = subgrafo.nodes[v].get("canal")
                if canal_v:
                    canales_hist.append(canal_v)

    if not canales_hist:
        return 0

    canal_frecuente = max(set(canales_hist), key=canales_hist.count)
    return 1 if canal_actual != canal_frecuente else 0


def calcular_I5(txn_id, subgrafo):
    """
    I5: Concentración de fraude en punto físico.
    Proporción de transacciones de clonación en el dispositivo
    visible dentro del subgrafo.
    """
    # Dispositivo de esta transacción
    dev = None
    for u, v, data in subgrafo.edges(data=True):
        if u == txn_id and data.get("tipo") == "desde_dispositivo":
            dev = v
            break

    if dev is None:
        return 0

    # Transacciones en ese dispositivo dentro del subgrafo
    txns_en_dev = []
    for u, v, data in subgrafo.edges(data=True):
        if v == dev and data.get("tipo") == "desde_dispositivo":
            txns_en_dev.append(u)

    if not txns_en_dev:
        return 0

    fraudes = sum(
        1 for t in txns_en_dev
        if subgrafo.nodes[t].get("etiqueta") == "clonacion"
    )
    return round(fraudes / len(txns_en_dev), 4)


# ─────────────────────────────────────────────
# PASO 3B — INDICADORES CNP (J1-J5)
# Calculados SOBRE el subgrafo Gi
# ─────────────────────────────────────────────
def calcular_J1(txn_id, subgrafo):
    """J1: Canal remoto."""
    data = subgrafo.nodes.get(txn_id, {})
    return 1 if data.get("canal") == "remoto" else 0


def calcular_J2(txn_id, subgrafo):
    """
    J2: IP asociada a múltiples tarjetas.
    Dentro del subgrafo cuenta cuántas tarjetas distintas
    usaron la misma IP que esta transacción.
    """
    ip_nodo = None
    for u, v, data in subgrafo.edges(data=True):
        if u == txn_id and data.get("tipo") == "desde_ip":
            ip_nodo = v
            break

    if ip_nodo is None:
        return 0

    # Tarjetas que llegan a transacciones que usan esa IP
    tarjetas_en_ip = set()
    for u, v, data in subgrafo.edges(data=True):
        if v == ip_nodo and data.get("tipo") == "desde_ip":
            txn_vecina = u
            for u2, v2, data2 in subgrafo.edges(data=True):
                if v2 == txn_vecina and data2.get("tipo") == "realiza":
                    tarjetas_en_ip.add(u2)

    return min(round(len(tarjetas_en_ip) / NC, 4), 1.0)


def calcular_J3(txn_id, subgrafo):
    """
    J3: Dispositivo usado con varias tarjetas.
    Dentro del subgrafo cuenta tarjetas distintas
    asociadas al dispositivo de esta transacción.
    """
    dev = None
    for u, v, data in subgrafo.edges(data=True):
        if u == txn_id and data.get("tipo") == "desde_dispositivo":
            dev = v
            break

    if dev is None:
        return 0

    tarjetas_en_dev = set()
    for u, v, data in subgrafo.edges(data=True):
        if v == dev and data.get("tipo") == "desde_dispositivo":
            txn_vecina = u
            for u2, v2, data2 in subgrafo.edges(data=True):
                if v2 == txn_vecina and data2.get("tipo") == "realiza":
                    tarjetas_en_dev.add(u2)

    return min(round(len(tarjetas_en_dev) / ND, 4), 1.0)


def calcular_J4(txn_id, subgrafo):
    """
    J4: Identidad remota reutilizada.
    Dentro del subgrafo cuenta tarjetas distintas
    asociadas al usuario de esta transacción.
    """
    # Encontrar usuario via: usuario → tarjeta → transaccion
    tarjeta = None
    for u, v, data in subgrafo.edges(data=True):
        if v == txn_id and data.get("tipo") == "realiza":
            tarjeta = u
            break

    if tarjeta is None:
        return 0

    usuario = None
    for u, v, data in subgrafo.edges(data=True):
        if v == tarjeta and data.get("tipo") == "usa_tarjeta":
            usuario = u
            break

    if usuario is None:
        return 0

    # Tarjetas del usuario en el subgrafo
    tarjetas_usuario = set()
    for u, v, data in subgrafo.edges(data=True):
        if u == usuario and data.get("tipo") == "usa_tarjeta":
            tarjetas_usuario.add(v)

    return min(round(len(tarjetas_usuario) / NA, 4), 1.0)


def calcular_J5(txn_id, subgrafo):
    """
    J5: Alta tasa de fraude en infraestructura digital.
    Proporción de transacciones CNP en el dispositivo
    o IP visibles dentro del subgrafo.
    """
    dev = None
    ip  = None
    for u, v, data in subgrafo.edges(data=True):
        if u == txn_id and data.get("tipo") == "desde_dispositivo":
            dev = v
        if u == txn_id and data.get("tipo") == "desde_ip":
            ip = v

    txns_infra = set()
    for u, v, data in subgrafo.edges(data=True):
        if (v == dev and data.get("tipo") == "desde_dispositivo") or \
           (v == ip  and data.get("tipo") == "desde_ip"):
            txns_infra.add(u)

    if not txns_infra:
        return 0

    fraudes_cnp = sum(
        1 for t in txns_infra
        if subgrafo.nodes[t].get("etiqueta") == "cnp"
    )
    return round(fraudes_cnp / len(txns_infra), 4)


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL — ALGORITMO 1 COMPLETO
# ─────────────────────────────────────────────
def ejecutar_algoritmo1(G, df, k=K):
    """
    Implementación completa del Algoritmo 1 (Sección 3.6).

    Para cada ti en Ts:
      1. Gi ← extraer_subgrafo(GH, ti, k)
      2. I  ← calcular_indicadores_clonacion(Gi, ti)
      3. J  ← calcular_indicadores_CNP(Gi, ti)
      4. S_CLON ← Σ αm * Im
      5. S_CNP  ← Σ βm * Jm
      6. Aplicar regla de decisión con δ
      7. Registrar resultado y explicación
    """
    resultados = []
    total      = len(df)

    print(f"\n  Procesando {total} transacciones con k={k}...")

    for idx, row in df.iterrows():
        txn_id = row["id_transaccion"]

        # ── Paso 1: Extraer subgrafo Gi ──
        Gi = extraer_subgrafo(G, txn_id, k=k)

        # ── Pasos 2-3: Calcular indicadores sobre Gi ──
        I1 = calcular_I1(txn_id, Gi)
        I2 = calcular_I2(txn_id, Gi)
        I3 = calcular_I3(txn_id, Gi)
        I4 = calcular_I4(txn_id, Gi)
        I5 = calcular_I5(txn_id, Gi)

        J1 = calcular_J1(txn_id, Gi)
        J2 = calcular_J2(txn_id, Gi)
        J3 = calcular_J3(txn_id, Gi)
        J4 = calcular_J4(txn_id, Gi)
        J5 = calcular_J5(txn_id, Gi)

        # ── Pasos 4-5: Calcular puntajes ──
        S_CLON = round(
            ALPHA["I1"]*I1 + ALPHA["I2"]*I2 + ALPHA["I3"]*I3 +
            ALPHA["I4"]*I4 + ALPHA["I5"]*I5, 4
        )
        S_CNP  = round(
            BETA["J1"]*J1 + BETA["J2"]*J2 + BETA["J3"]*J3 +
            BETA["J4"]*J4 + BETA["J5"]*J5, 4
        )

        # ── Paso 6: Regla de decisión ──
        diff_clon = S_CLON - S_CNP
        diff_cnp  = S_CNP  - S_CLON

        if diff_clon > DELTA:
            prediccion = "clonacion"
        elif diff_cnp > DELTA:
            prediccion = "cnp"
        else:
            prediccion = "indeterminado"

        # ── Paso 7: Explicación estructural ──
        contribs = {
            "α1·I1": round(ALPHA["I1"]*I1, 3),
            "α2·I2": round(ALPHA["I2"]*I2, 3),
            "α3·I3": round(ALPHA["I3"]*I3, 3),
            "α4·I4": round(ALPHA["I4"]*I4, 3),
            "α5·I5": round(ALPHA["I5"]*I5, 3),
            "β1·J1": round(BETA["J1"]*J1, 3),
            "β2·J2": round(BETA["J2"]*J2, 3),
            "β3·J3": round(BETA["J3"]*J3, 3),
            "β4·J4": round(BETA["J4"]*J4, 3),
            "β5·J5": round(BETA["J5"]*J5, 3),
        }
        top3 = sorted(contribs.items(),
                      key=lambda x: x[1], reverse=True)[:3]
        explicacion = " | ".join([f"{k}={v}" for k, v in top3])

        # ── Registrar resultado ──
        resultados.append({
            "id_transaccion": txn_id,
            "etiqueta"      : row["etiqueta"],
            "nodos_subgrafo": Gi.number_of_nodes(),
            "aristas_subgrafo": Gi.number_of_edges(),
            "I1": I1, "I2": I2, "I3": I3, "I4": I4, "I5": I5,
            "J1": J1, "J2": J2, "J3": J3, "J4": J4, "J5": J5,
            "S_CLON"    : S_CLON,
            "S_CNP"     : S_CNP,
            "prediccion": prediccion,
            "explicacion": explicacion,
        })

        if (idx + 1) % 100 == 0:
            print(f"    Procesadas: {idx+1}/{total}")

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────
# EXTENSIÓN — PERCEPTRÓN DE UNA CAPA (Sección 3.7)
# ─────────────────────────────────────────────
def entrenar_perceptron(df_res):
    """
    Entrena perceptrón con los indicadores calculados
    correctamente sobre subgrafos k=2.

    Entrada : x = [I1..I5, J1..J5] ∈ R¹⁰  (del subgrafo)
    Salida  : p = softmax(Wx+b) → clase
    """
    cols = ["I1","I2","I3","I4","I5","J1","J2","J3","J4","J5"]
    X    = df_res[cols].values
    y    = df_res["etiqueta"].values

    # División cronológica 70/15/15
    n       = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train, y_train = X[:n_train],          y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

    print(f"\n  División cronológica del dataset:")
    print(f"    Entrenamiento : {len(X_train)} (70%)")
    print(f"    Validación    : {len(X_val)}  (15%)")
    print(f"    Prueba        : {len(X_test)}  (15%)")

    modelo = SGDClassifier(
        loss         = "log_loss",
        max_iter     = 200,
        random_state = 42,
        class_weight = "balanced",
        learning_rate= "constant",
        eta0         = 0.01
    )
    modelo.fit(X_train, y_train)

    # Validación
    acc_val = accuracy_score(y_val, modelo.predict(X_val))
    print(f"\n  Exactitud en validación : {round(acc_val*100,2)}%")

    # Prueba
    y_pred = modelo.predict(X_test)
    print(f"\n  Reporte en prueba:\n")
    print(classification_report(y_test, y_pred,
                                labels=sorted(set(y_test)),
                                zero_division=0))
    print(f"  Exactitud en prueba : {round(accuracy_score(y_test,y_pred)*100,2)}%")

    # Pesos aprendidos
    cols_ind = ["I1","I2","I3","I4","I5","J1","J2","J3","J4","J5"]
    print(f"\n  Pesos aprendidos W:\n")
    print(f"  {'Indicador':<10}", end="")
    for c in modelo.classes_:
        print(f"  {c:<14}", end="")
    print()
    print("  " + "─" * (10 + len(modelo.classes_)*16))
    for j, ind in enumerate(cols_ind):
        print(f"  {ind:<10}", end="")
        for k in range(len(modelo.classes_)):
            print(f"  {modelo.coef_[k][j]:>+.4f}        ", end="")
        print()

    return modelo


# ─────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  ALGORITMO 1 COMPLETO — Subgrafo k=2 + Perceptrón")
    print("  Implementación de la Sección 3.6 de la Metodología")
    print("=" * 60)

    # ── Cargar datos ──
    print("\n[1/5] Cargando dataset...")
    df = pd.read_csv(
        "transacciones_sinteticas.csv",
        parse_dates=["timestamp"]
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"      {len(df)} transacciones cargadas.")

    # ── Construir GH ──
    print("\n[2/5] Construyendo grafo heterogéneo GH = (V, E, φ, ψ)...")
    G = construir_grafo(df)
    print(f"      Vértices : {G.number_of_nodes()}")
    print(f"      Aristas  : {G.number_of_edges()}")

    # ── Inspeccionar un subgrafo de ejemplo ──
    print("\n[3/5] Ejemplo de subgrafo k=2 para una transacción:")
    ejemplo_txn = df[df["etiqueta"]=="clonacion"]["id_transaccion"].iloc[0]
    Gi_ejemplo  = extraer_subgrafo(G, ejemplo_txn, k=K)
    inspeccionar_subgrafo(Gi_ejemplo, ejemplo_txn)

    # ── Ejecutar Algoritmo 1 completo ──
    print("\n[4/5] Ejecutando Algoritmo 1 (subgrafo k=2 por transacción)...")
    df_res = ejecutar_algoritmo1(G, df, k=K)

    # Resumen de resultados
    print("\n" + "─" * 60)
    print("  RESULTADOS — Regla de decisión con pesos manuales")
    print("─" * 60)

    total  = len(df_res)
    indet  = len(df_res[df_res["prediccion"]=="indeterminado"])
    clasif = df_res[df_res["prediccion"]!="indeterminado"]

    print(f"\n  Total          : {total}")
    print(f"  Indeterminadas : {indet} ({round(indet/total*100,1)}%)")
    print(f"  Clasificadas   : {len(clasif)} ({round(len(clasif)/total*100,1)}%)")

    if len(clasif) > 0:
        print(f"\n  Reporte (transacciones clasificadas):\n")
        print(classification_report(
            clasif["etiqueta"], clasif["prediccion"],
            labels=sorted(clasif["etiqueta"].unique()),
            zero_division=0
        ))

    # Ejemplos de explicación
    print("─" * 60)
    print("  EJEMPLOS DE EXPLICACIÓN ESTRUCTURAL")
    print("─" * 60)
    fraudes = df_res[df_res["etiqueta"]!="legitima"].head(4)
    for _, row in fraudes.iterrows():
        print(f"\n  {row['id_transaccion']} | "
              f"Real: {row['etiqueta']} | "
              f"Predicho: {row['prediccion']}")
        print(f"  Subgrafo: {row['nodos_subgrafo']} nodos, "
              f"{row['aristas_subgrafo']} aristas")
        print(f"  SCLON={row['S_CLON']}  SCNP={row['S_CNP']}  δ={DELTA}")
        print(f"  Top contribuciones: {row['explicacion']}")

    # ── Perceptrón ──
    print("\n" + "─" * 60)
    print("  EXTENSIÓN — Perceptrón de una capa (Sección 3.7)")
    print("─" * 60)
    print("\n[5/5] Entrenando perceptrón sobre indicadores del subgrafo...")
    modelo = entrenar_perceptron(df_res)

    # ── Guardar ──
    ruta = "resultados_algoritmo1_completo.csv"
    df_res.to_csv(ruta, index=False)
    print(f"\n  Resultados guardados en: {ruta}")
    print("=" * 60)
