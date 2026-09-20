"""
Fase 3 — Regla de decisión y Perceptrón de una capa
Fraude con tarjeta: Clonación vs Card-Not-Present (CNP)

Implementa:
  Parte A: Regla de decisión con pesos manuales (experto)
  Parte B: Perceptrón de una capa con pesos aprendidos
  Parte C: Comparación de ambos enfoques
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# ─────────────────────────────────────────────

# Pesos manuales para indicadores de Clonación (α1..α5)
# Deben sumar 1.0
ALPHA = {
    "I1": 0.35,   # Canal presencial — señal más fuerte
    "I2": 0.25,   # Dispositivo nuevo
    "I3": 0.20,   # Geo-velocity
    "I4": 0.10,   # Ruptura de patrón
    "I5": 0.10,   # Concentración de fraude físico
}

# Pesos manuales para indicadores CNP (β1..β5)
# Deben sumar 1.0
BETA = {
    "J1": 0.35,   # Canal remoto — señal más fuerte
    "J2": 0.25,   # IP compartida
    "J3": 0.20,   # Dispositivo compartido
    "J4": 0.10,   # Identidad reutilizada
    "J5": 0.10,   # Concentración de fraude digital
}

# Margen de decisión δ
# Si la diferencia entre puntajes es menor a δ → Indeterminado
DELTA = 0.15


# ─────────────────────────────────────────────
# PARTE A — REGLA DE DECISIÓN CON PESOS MANUALES
# ─────────────────────────────────────────────
def calcular_puntajes(df_ind):
    """
    Calcula SCLON y SCNP para cada transacción.

    SCLON(ti) = Σ αj * Ij    (j=1..5)
    SCNP(ti)  = Σ βj * Jj    (j=1..5)
    """
    df = df_ind.copy()

    df["SCLON"] = (
        ALPHA["I1"] * df["I1"] +
        ALPHA["I2"] * df["I2"] +
        ALPHA["I3"] * df["I3"] +
        ALPHA["I4"] * df["I4"] +
        ALPHA["I5"] * df["I5"]
    ).round(4)

    df["SCNP"] = (
        BETA["J1"] * df["J1"] +
        BETA["J2"] * df["J2"] +
        BETA["J3"] * df["J3"] +
        BETA["J4"] * df["J4"] +
        BETA["J5"] * df["J5"]
    ).round(4)

    return df


def aplicar_regla_decision(df):
    """
    Aplica la regla de decisión con margen δ:

    y(ti) = Clonación     si SCLON - SCNP > δ
            CNP           si SCNP - SCLON > δ
            Indeterminado en otro caso
    """
    def clasificar(row):
        diff_clon = row["SCLON"] - row["SCNP"]
        diff_cnp  = row["SCNP"]  - row["SCLON"]

        if diff_clon > DELTA:
            return "clonacion"
        elif diff_cnp > DELTA:
            return "cnp"
        else:
            return "indeterminado"

    df["prediccion_manual"] = df.apply(clasificar, axis=1)
    return df


def evaluar_modelo_manual(df):
    """Evalúa el modelo de pesos manuales excluyendo Indeterminados."""

    total        = len(df)
    indet        = len(df[df["prediccion_manual"] == "indeterminado"])
    clasificados = df[df["prediccion_manual"] != "indeterminado"].copy()

    print(f"\n  Total transacciones   : {total}")
    print(f"  Indeterminadas        : {indet} "
          f"({round(indet/total*100, 1)}%) → pasan a revisión manual")
    print(f"  Clasificadas          : {len(clasificados)} "
          f"({round(len(clasificados)/total*100, 1)}%)")

    if len(clasificados) == 0:
        print("  Sin transacciones clasificadas.")
        return

    y_real = clasificados["etiqueta"]
    y_pred = clasificados["prediccion_manual"]

    # Solo evaluar clases que aparecen en ambos
    clases = sorted(set(y_real.unique()) | set(y_pred.unique()))

    print(f"\n  Reporte de clasificación (transacciones clasificadas):\n")
    print(classification_report(y_real, y_pred,
                                labels=clases,
                                zero_division=0))

    print(f"  Exactitud general     : "
          f"{round(accuracy_score(y_real, y_pred) * 100, 2)}%")


def imprimir_explicacion(df, n=5):
    """
    Muestra la explicación estructural de las primeras n
    transacciones fraudulentas, ordenada por contribución de indicadores.
    """
    print(f"\n  Ejemplos de explicación estructural "
          f"(primeras {n} transacciones fraudulentas):\n")

    fraudes = df[df["etiqueta"] != "legitima"].head(n)

    for _, row in fraudes.iterrows():
        print(f"  ID: {row['id_transaccion']} | "
              f"Real: {row['etiqueta']} | "
              f"Predicho: {row['prediccion_manual']}")
        print(f"  SCLON={row['SCLON']}  SCNP={row['SCNP']}  δ={DELTA}")

        # Contribuciones ordenadas de mayor a menor
        contribs = {
            "α1·I1": round(ALPHA["I1"] * row["I1"], 3),
            "α2·I2": round(ALPHA["I2"] * row["I2"], 3),
            "α3·I3": round(ALPHA["I3"] * row["I3"], 3),
            "α4·I4": round(ALPHA["I4"] * row["I4"], 3),
            "α5·I5": round(ALPHA["I5"] * row["I5"], 3),
            "β1·J1": round(BETA["J1"] * row["J1"], 3),
            "β2·J2": round(BETA["J2"] * row["J2"], 3),
            "β3·J3": round(BETA["J3"] * row["J3"], 3),
            "β4·J4": round(BETA["J4"] * row["J4"], 3),
            "β5·J5": round(BETA["J5"] * row["J5"], 3),
        }
        ordenadas = sorted(contribs.items(),
                           key=lambda x: x[1], reverse=True)
        top3 = "  |  ".join([f"{k}={v}" for k, v in ordenadas[:3]])
        print(f"  Top contribuciones → {top3}")
        print()


# ─────────────────────────────────────────────
# PARTE B — PERCEPTRÓN DE UNA CAPA
# ─────────────────────────────────────────────
def entrenar_perceptron(df_ind):
    """
    Entrena un perceptrón de una capa con los 10 indicadores como entrada.

    Arquitectura:
      Entrada  : x = [I1, I2, I3, I4, I5, J1, J2, J3, J4, J5] ∈ R¹⁰
      Salida   : p = softmax(Wx + b) → probabilidad de cada clase
      Clases   : {clonacion, cnp, legitima}
    """
    cols_ind = ["I1", "I2", "I3", "I4", "I5",
                "J1", "J2", "J3", "J4", "J5"]

    X = df_ind[cols_ind].values
    y = df_ind["etiqueta"].values

    # División cronológica: 70% entrenamiento, 15% validación, 15% prueba
    n       = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val   = X[n_train:n_train + n_val]
    y_val   = y[n_train:n_train + n_val]
    X_test  = X[n_train + n_val:]
    y_test  = y[n_train + n_val:]

    print(f"\n  División del dataset:")
    print(f"    Entrenamiento : {len(X_train)} transacciones (70%)")
    print(f"    Validación    : {len(X_val)}  transacciones (15%)")
    print(f"    Prueba        : {len(X_test)}  transacciones (15%)")

    # Perceptrón de una capa con entropía cruzada y SGD
    # loss='log_loss' equivale a softmax + entropía cruzada multiclase
    modelo = SGDClassifier(
        loss        = "log_loss",
        max_iter    = 200,
        tol         = 1e-3,
        random_state= 42,
        class_weight= "balanced",    # maneja el desbalance de clases
        learning_rate= "constant",
        eta0        = 0.01
    )

    modelo.fit(X_train, y_train)

    # Evaluación en validación
    y_pred_val = modelo.predict(X_val)
    print(f"\n  Exactitud en validación : "
          f"{round(accuracy_score(y_val, y_pred_val) * 100, 2)}%")

    # Evaluación en prueba
    y_pred_test = modelo.predict(X_test)

    print(f"\n  Reporte de clasificación (conjunto de prueba):\n")
    clases = sorted(set(y_test) | set(y_pred_test))
    print(classification_report(y_test, y_pred_test,
                                labels=clases,
                                zero_division=0))
    print(f"  Exactitud en prueba     : "
          f"{round(accuracy_score(y_test, y_pred_test) * 100, 2)}%")

    return modelo, X_test, y_test, y_pred_test


def mostrar_pesos_perceptron(modelo):
    """
    Muestra los pesos aprendidos W por el perceptrón.
    Wk,j = contribución del indicador j a la clase k.
    """
    cols_ind = ["I1", "I2", "I3", "I4", "I5",
                "J1", "J2", "J3", "J4", "J5"]
    clases   = modelo.classes_

    print(f"\n  Pesos aprendidos W ∈ R^(clases × indicadores):\n")
    print(f"  {'Indicador':<12}", end="")
    for c in clases:
        print(f"  {c:<14}", end="")
    print()
    print("  " + "─" * (12 + len(clases) * 16))

    coef = modelo.coef_
    for j, ind in enumerate(cols_ind):
        print(f"  {ind:<12}", end="")
        for k in range(len(clases)):
            print(f"  {coef[k][j]:>+.4f}        ", end="")
        print()


# ─────────────────────────────────────────────
# PARTE C — COMPARACIÓN DE AMBOS ENFOQUES
# ─────────────────────────────────────────────
def comparar_enfoques(df, y_test, y_pred_perceptron):
    """Compara las métricas de ambos enfoques."""

    # Métricas del modelo manual (sobre transacciones clasificadas)
    clasificadas = df[df["prediccion_manual"] != "indeterminado"]
    acc_manual   = accuracy_score(
        clasificadas["etiqueta"],
        clasificadas["prediccion_manual"]
    ) if len(clasificadas) > 0 else 0

    # Métricas del perceptrón
    acc_perceptron = accuracy_score(y_test, y_pred_perceptron)

    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │         COMPARACIÓN DE ENFOQUES             │")
    print("  ├──────────────────────┬──────────────────────┤")
    print("  │ Pesos manuales       │ Perceptrón           │")
    print("  ├──────────────────────┼──────────────────────┤")
    print(f"  │ Exactitud: {round(acc_manual*100,1):>5}%    │"
          f" Exactitud: {round(acc_perceptron*100,1):>5}%    │")
    print(f"  │ Interpreta.: Alta   │ Interpreta.: Media   │")
    print(f"  │ Entrena.  : No      │ Entrena.  : Sí       │")
    print(f"  │ Adaptable : No      │ Adaptable : Sí       │")
    print("  └──────────────────────┴──────────────────────┘")

    print("""
  Conclusión:
  · Los pesos manuales ofrecen máxima interpretabilidad y no
    requieren datos etiquetados, pero dependen del experto.
  · El perceptrón ajusta los pesos automáticamente y se adapta
    a los datos, manteniendo interpretabilidad por sus pesos W.
  · Ambos sirven como base antes de escalar a arquitecturas
    GNN más complejas sobre el grafo heterogéneo completo.
    """)


# ─────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  FASE 3 — Regla de decisión y Perceptrón")
    print("=" * 55)

    # ── Cargar indicadores de Fase 2 ──
    print("\n[1/5] Cargando indicadores de Fase 2...")
    df_ind = pd.read_csv(
        "indicadores_calculados.csv"
    )
    print(f"      {len(df_ind)} transacciones cargadas.")

    # ── Parte A: Puntajes y regla de decisión ──
    print("\n" + "─" * 55)
    print("  PARTE A — Regla de decisión con pesos manuales")
    print("─" * 55)

    print(f"\n[2/5] Calculando puntajes SCLON y SCNP...")
    print(f"\n  Pesos α (Clonación): {ALPHA}")
    print(f"  Pesos β (CNP)      : {BETA}")
    print(f"  Margen δ           : {DELTA}")

    df_res = calcular_puntajes(df_ind)
    df_res = aplicar_regla_decision(df_res)

    print(f"\n[3/5] Evaluando modelo manual...")
    evaluar_modelo_manual(df_res)
    imprimir_explicacion(df_res, n=4)

    # ── Parte B: Perceptrón ──
    print("─" * 55)
    print("  PARTE B — Perceptrón de una capa")
    print("─" * 55)

    print(f"\n[4/5] Entrenando perceptrón...")
    modelo, X_test, y_test, y_pred = entrenar_perceptron(df_ind)
    mostrar_pesos_perceptron(modelo)

    # ── Parte C: Comparación ──
    print("\n" + "─" * 55)
    print("  PARTE C — Comparación de enfoques")
    print("─" * 55)

    print(f"\n[5/5] Comparando ambos enfoques...")
    comparar_enfoques(df_res, y_test, y_pred)

    # ── Guardar resultados finales ──
    ruta = "resultados_clasificacion.csv"
    df_res.to_csv(ruta, index=False)
    print(f"  Resultados guardados en: {ruta}")
    print("=" * 55)
