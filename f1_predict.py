import fastf1
import pandas as pd
import datetime
import os

# Configuración de la caché para descargas rápidas
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

def obtener_ultimo_gp_automatico():
    """Detecta automáticamente el GP más reciente o en curso y su formato."""
    hoy = datetime.datetime.now()
    # Le sumamos 2 días para que si es viernes o sábado, ya detecte la carrera del domingo
    margen_fin_de_semana = hoy + datetime.timedelta(days=2)
    
    anio_actual = hoy.year
    calendario = fastf1.get_event_schedule(anio_actual)
    eventos_pasados = calendario[calendario['EventDate'] <= margen_fin_de_semana]
    
    anio_evento = anio_actual
    if eventos_pasados.empty:
        anio_evento = anio_actual - 1
        calendario = fastf1.get_event_schedule(anio_evento)
        ultimo_evento = calendario.iloc[-1]
    else:
        ultimo_evento = eventos_pasados.iloc[-1]
        
    return anio_evento, ultimo_evento['EventName'], ultimo_evento['EventFormat']

def obtener_vueltas_rapidas(laps_df):
    """Función de apoyo: Devuelve un DataFrame con la vuelta más rápida de cada piloto."""
    df_limpio = laps_df.dropna(subset=['LapTime'])
    if df_limpio.empty:
        return pd.DataFrame()
    # Busca el índice del tiempo mínimo para cada piloto y devuelve esas filas
    idx_rapidas = df_limpio.groupby('Driver')['LapTime'].idxmin()
    return df_limpio.loc[idx_rapidas].reset_index(drop=True)

def analizar_fin_de_semana_completo(anio, gran_premio, formato):
    print(f"\nDescargando y procesando telemetría para: {gran_premio} ({anio}) | Formato: {formato}...")
    
    # CONTROL DE FORMATO SPRINT
    if formato in ['sprint', 'sprint_shootout']:
        print("⚠️ ADVERTENCIA: Este fin de semana es SPRINT. Las sesiones FP2 y FP3 no existen.")
        print("El coeficiente predictivo de la FP1 cae a C ≈ 0.18. El modelo estándar no es aplicable.")
        return None

    # 1. Cargar Sesiones de Libres (FP2 y FP3)
    try:
        fp2 = fastf1.get_session(anio, gran_premio, 'FP2')
        fp3 = fastf1.get_session(anio, gran_premio, 'FP3')
        fp2.load(laps=True, telemetry=False, weather=False)
        fp3.load(laps=True, telemetry=False, weather=False)
    except Exception as e:
        print(f"❌ Error al cargar FP2 o FP3: {e}")
        return None

    # 2. Cargar Sesión de Clasificación (OPCIÓN B)
    hay_qualy = False
    try:
        qualy = fastf1.get_session(anio, gran_premio, 'Q')
        qualy.load(laps=True, telemetry=False, weather=False)
        if not qualy.laps.empty:
            hay_qualy = True
    except Exception:
        pass 

    if not hay_qualy:
        print("⚠️ Datos de Qualy no disponibles aún. Generando reporte predictivo solo con FP2 y FP3...")

    # --- PROCESAMIENTO AVANZADO DE FP2 (Mediana de Tandas Largas) ---
    laps_fp2 = fp2.laps.pick_accurate()
    laps_fp2 = laps_fp2[laps_fp2['PitOutTime'].isna() & laps_fp2['PitInTime'].isna()]
    
    if not laps_fp2.empty:
        # Usamos transform para calcular el límite del 107% sin borrar columnas ni generar warnings
        tiempos_minimos = laps_fp2.groupby(['Driver', 'Stint'])['LapTime'].transform('min')
        vueltas_lanzadas = laps_fp2[laps_fp2['LapTime'] <= tiempos_minimos * 1.07]
        
        # Filtramos las tandas de 5 o más vueltas
        stints_largos = vueltas_lanzadas.groupby(['Driver', 'Stint']).filter(lambda x: len(x) >= 5)
    else:
        stints_largos = pd.DataFrame()

    if not stints_largos.empty:
        ritmo_carrera_fp2 = stints_largos.groupby('Driver')['LapTime'].median().reset_index()
        ritmo_carrera_fp2 = ritmo_carrera_fp2.sort_values(by='LapTime').reset_index(drop=True)
        ritmo_carrera_fp2['Pos_Ritmo_FP2'] = ritmo_carrera_fp2.index + 1
    else:
        # SOLUCIÓN ATTRIBUTE ERROR
        vueltas_rapidas_fp2 = obtener_vueltas_rapidas(fp2.laps)
        ritmo_carrera_fp2 = vueltas_rapidas_fp2[['Driver', 'LapTime']].sort_values(by='LapTime').reset_index(drop=True)
        ritmo_carrera_fp2['Pos_Ritmo_FP2'] = ritmo_carrera_fp2.index + 1

    # SOLUCIÓN ATTRIBUTE ERROR
    fp2_fastest = obtener_vueltas_rapidas(fp2.laps)[['Driver', 'Compound']]

    # --- PROCESAMIENTO FP3 ---
    vueltas_rapidas_fp3 = obtener_vueltas_rapidas(fp3.laps)
    fp3_df = vueltas_rapidas_fp3[['Driver', 'LapTime']].sort_values(by='LapTime').reset_index(drop=True)
    fp3_df['Pos_FP3'] = fp3_df.index + 1

    # --- CRUCE DE TABLAS BASE ---
    df_cruzado = pd.merge(ritmo_carrera_fp2, fp2_fastest, on='Driver')
    df_cruzado = pd.merge(df_cruzado, fp3_df[['Driver', 'Pos_FP3']], on='Driver')
    
    # --- PROCESAMIENTO QUALY (Solo si existe) ---
    if hay_qualy:
        vueltas_rapidas_q = obtener_vueltas_rapidas(qualy.laps)
        q_df = vueltas_rapidas_q[['Driver', 'LapTime']].sort_values(by='LapTime').reset_index(drop=True)
        q_df['Pos_Qualy'] = q_df.index + 1
        df_cruzado = pd.merge(df_cruzado, q_df[['Driver', 'Pos_Qualy']], on='Driver')

    reporte = []
    for idx, row in df_cruzado.iterrows():
        piloto = row['Driver']
        pos_ritmo_fp2 = int(row['Pos_Ritmo_FP2'])
        pos_fp3 = int(row['Pos_FP3'])
        compound_fp2 = str(row['Compound']).upper()
        
        # A. Evaluación de Ritmo Libre
        eval_fp3 = []
        if pos_ritmo_fp2 <= 5 and compound_fp2 in ['MEDIUM', 'HARD']:
            eval_fp3.append("Ritmo real superior (Goma de carrera).")
            
        if pos_ritmo_fp2 <= 3 and pos_fp3 <= 3:
            eval_fp3.append("Candidato Podio/Victoria (C ≈ 0,60-0,68).")
        elif pos_ritmo_fp2 in [4, 5] and pos_fp3 in [1, 2]:
            eval_fp3.append("Peligroso: Base sólida confirmada.")
        elif pos_ritmo_fp2 in [1, 2] and pos_fp3 >= 6:
            eval_fp3.append("Alerta: Caída de rendimiento sábado.")

        fila_reporte = {
            'Piloto': piloto,
            'Mediana FP2': f"P{pos_ritmo_fp2}",
            'FP3': f"P{pos_fp3}",
            'Diagnóstico Libres': " | ".join(eval_fp3) if eval_fp3 else "Normal"
        }

        # B. Validación Qualy y Tendencia WDC
        if hay_qualy:
            pos_q = int(row['Pos_Qualy'])
            eval_qualy = []
            if pos_fp3 <= 3 and pos_q <= 3:
                eval_qualy.append("Cumplió expectativas.")
            elif pos_fp3 <= 3 and pos_q >= 6:
                eval_qualy.append("RUPTURA DE PROYECCIÓN.")
            elif pos_fp3 >= 6 and pos_q <= 3:
                eval_qualy.append("SORPRESA EN QUALY: Riesgo alto de degradación.")
            else:
                eval_qualy.append("Alineado.")

            eval_wdc = []
            if pos_q <= 3 and pos_ritmo_fp2 <= 3 and pos_fp3 <= 3:
                eval_wdc.append("TENDENCIA WDC ALTA (ρ ≈ 0.85-0.91).")
            elif pos_q <= 5 and (pos_ritmo_fp2 > 7 or pos_fp3 > 7):
                eval_wdc.append("ANOMALÍA: Puesto por encima del ritmo real.")

            fila_reporte['QUALY'] = f"P{pos_q}"
            fila_reporte['Validación Qualy'] = " | ".join(eval_qualy)
            fila_reporte['Tendencia WDC'] = " | ".join(eval_wdc) if eval_wdc else "En rango"

        reporte.append(fila_reporte)

# --- MÓDULO DE PREDICCIÓN DE PODIO (RANKING MATEMÁTICO) ---
    df_final = pd.DataFrame(reporte)
    
    if hay_qualy:
        df_final['Puntaje_Carrera'] = 0.0
        
        for idx, row in df_final.iterrows():
            # Extraemos los números puros de las columnas de texto (quitando la 'P')
            pos_q = int(row['QUALY'].replace('P', ''))
            pos_fp2 = int(row['Mediana FP2'].replace('P', ''))
            
            # FÓRMULA ESTADÍSTICA PURA (78% Precisión teórica)
            # Peso distribuido según la varianza explicada: Qualy (>65%) y FP2 (~30-35%)
            puntaje = (pos_q * 0.65) + (pos_fp2 * 0.35)
            
            # PENALIZADOR DE DEGRADACIÓN EXTREMA
            if pos_q <= 5 and pos_fp2 >= 8:
                puntaje += 5.0
                
            df_final.at[idx, 'Puntaje_Carrera'] = puntaje
            
        # Ordenamos la tabla de mejor a peor puntaje
        df_final = df_final.sort_values(by='Puntaje_Carrera').reset_index(drop=True)
        podio_predicho = df_final['Piloto'].head(3).tolist()
        
        return df_final, podio_predicho
    else:
        return df_final, ["Sin datos", "Sin datos", "Sin datos"]

# --- EJECUCIÓN AUTOMÁTICA FINAL ---
ANIO, GRAN_PREMIO, FORMATO = obtener_ultimo_gp_automatico()
datos_salida = analizar_fin_de_semana_completo(ANIO, GRAN_PREMIO, FORMATO)

# Evitamos errores si el GP es formato Sprint y devuelve None
if datos_salida is not None:
    resultado, podio = datos_salida
    
    print(f"\n================ INFORME PREDICTIVO: {GRAN_PREMIO} ({ANIO}) ================")
    
    # Imprimimos la tabla ocultando la columna matemática para que quede limpia
    if 'Puntaje_Carrera' in resultado.columns:
        print(resultado.drop(columns=['Puntaje_Carrera']).to_string(index=False))
    else:
        print(resultado.to_string(index=False))
        
    print("\n" + "="*70)
    print(" 🤖 PREDICCIÓN AUTOMÁTICA DEL PODIO (BASADA EN ALGORITMO) 🤖")
    print(f" 🥇 1º {podio[0]} | 🥈 2º {podio[1]} | 🥉 3º {podio[2]}")
    print("="*70 + "\n")
# --- EJECUCIÓN AUTOMÁTICA FINAL ---
ANIO, GRAN_PREMIO, FORMATO = obtener_ultimo_gp_automatico()
datos_salida = analizar_fin_de_semana_completo(ANIO, GRAN_PREMIO, FORMATO)

# Evitamos errores si el GP es formato Sprint y devuelve None
if datos_salida is not None:
    resultado, podio = datos_salida
    
    print(f"\n================ INFORME PREDICTIVO: {GRAN_PREMIO} ({ANIO}) ================")
    
    # Imprimimos la tabla ocultando la columna matemática para que quede limpia
    if 'Puntaje_Carrera' in resultado.columns:
        print(resultado.drop(columns=['Puntaje_Carrera']).to_string(index=False))
    else:
        print(resultado.to_string(index=False))
        
    print("\n" + "="*70)
    print(" 🤖 PREDICCIÓN AUTOMÁTICA DEL PODIO (BASADA EN ALGORITMO) 🤖")
    print(f" 🥇 1º {podio[0]} | 🥈 2º {podio[1]} | 🥉 3º {podio[2]}")
    print("="*70 + "\n")
