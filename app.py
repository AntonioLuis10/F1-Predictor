import streamlit as st
import fastf1
import pandas as pd
import datetime
import os

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="F1 Predictive Engine", page_icon="🏎️", layout="wide")

# Configuración de caché
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

# --- FUNCIONES BASE (Ocultas al usuario) ---
@st.cache_data(ttl=3600) # La web guardará los datos 1 hora para no recargar lento
def obtener_ultimo_gp_automatico():
    hoy = datetime.datetime.now()
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
    df_limpio = laps_df.dropna(subset=['LapTime'])
    if df_limpio.empty: return pd.DataFrame()
    idx_rapidas = df_limpio.groupby('Driver')['LapTime'].idxmin()
    return df_limpio.loc[idx_rapidas].reset_index(drop=True)

# --- MOTOR PREDICTIVO ---
def analizar_datos(anio, gran_premio, formato):
    if formato in ['sprint', 'sprint_shootout']:
        st.warning("⚠️ Fin de semana SPRINT. La FP1 cae a un coeficiente predictivo de C ≈ 0.18. El modelo estándar no es aplicable.")
        return None, None

    try:
        fp2 = fastf1.get_session(anio, gran_premio, 'FP2')
        fp3 = fastf1.get_session(anio, gran_premio, 'FP3')
        fp2.load(laps=True, telemetry=False, weather=False)
        fp3.load(laps=True, telemetry=False, weather=False)
        
        # Forzamos la lectura de las vueltas aquí. Si están vacías, saltará al except y no romperá la web.
        _ = fp2.laps
        _ = fp3.laps
    except Exception as e:
        st.warning("⚠️ La telemetría de las prácticas libres (FP2/FP3) aún no está procesada por los servidores oficiales. Vuelve a intentarlo en unos minutos.")
        return None, None

    hay_qualy = False
    try:
        qualy = fastf1.get_session(anio, gran_premio, 'Q')
        qualy.load(laps=True, telemetry=False, weather=False)
        if not qualy.laps.empty: hay_qualy = True
    except Exception:
        pass 

    if not hay_qualy:
        st.info("ℹ️ Datos de Qualy no disponibles aún. Mostrando proyecciones basadas en Libres.")

    laps_fp2 = fp2.laps.pick_accurate()
    laps_fp2 = laps_fp2[laps_fp2['PitOutTime'].isna() & laps_fp2['PitInTime'].isna()]
    
    if not laps_fp2.empty:
        tiempos_minimos = laps_fp2.groupby(['Driver', 'Stint'])['LapTime'].transform('min')
        vueltas_lanzadas = laps_fp2[laps_fp2['LapTime'] <= tiempos_minimos * 1.07]
        stints_largos = vueltas_lanzadas.groupby(['Driver', 'Stint']).filter(lambda x: len(x) >= 5)
    else:
        stints_largos = pd.DataFrame()

    if not stints_largos.empty:
        ritmo_carrera_fp2 = stints_largos.groupby('Driver')['LapTime'].median().reset_index()
        ritmo_carrera_fp2 = ritmo_carrera_fp2.sort_values(by='LapTime').reset_index(drop=True)
        ritmo_carrera_fp2['Pos_Ritmo_FP2'] = ritmo_carrera_fp2.index + 1
    else:
        vueltas_rapidas_fp2 = obtener_vueltas_rapidas(fp2.laps)
        ritmo_carrera_fp2 = vueltas_rapidas_fp2[['Driver', 'LapTime']].sort_values(by='LapTime').reset_index(drop=True)
        ritmo_carrera_fp2['Pos_Ritmo_FP2'] = ritmo_carrera_fp2.index + 1

    fp2_fastest = obtener_vueltas_rapidas(fp2.laps)[['Driver', 'Compound']]
    
    vueltas_rapidas_fp3 = obtener_vueltas_rapidas(fp3.laps)
    fp3_df = vueltas_rapidas_fp3[['Driver', 'LapTime']].sort_values(by='LapTime').reset_index(drop=True)
    fp3_df['Pos_FP3'] = fp3_df.index + 1

    df_cruzado = pd.merge(ritmo_carrera_fp2, fp2_fastest, on='Driver')
    df_cruzado = pd.merge(df_cruzado, fp3_df[['Driver', 'Pos_FP3']], on='Driver')
    
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
        
        eval_fp3 = []
        if pos_ritmo_fp2 <= 3 and pos_fp3 <= 3: eval_fp3.append("Candidato Podio/Victoria")
        elif pos_ritmo_fp2 in [4, 5] and pos_fp3 in [1, 2]: eval_fp3.append("Peligroso: Base sólida")
        elif pos_ritmo_fp2 in [1, 2] and pos_fp3 >= 6: eval_fp3.append("Alerta: Caída sábado")

        fila = {
            'Piloto': piloto,
            'Mediana FP2': f"P{pos_ritmo_fp2}",
            'FP3': f"P{pos_fp3}",
            'Diagnóstico Libres': " | ".join(eval_fp3) if eval_fp3 else "Normal"
        }

        if hay_qualy:
            pos_q = int(row['Pos_Qualy'])
            eval_q = "Cumplió" if pos_fp3 <= 3 and pos_q <= 3 else "Ruptura" if pos_fp3 <= 3 and pos_q >= 6 else "Alineado"
            fila['Qualy'] = f"P{pos_q}"
            fila['Validación'] = eval_q

        reporte.append(fila)

    df_final = pd.DataFrame(reporte)
    
    if hay_qualy:
        df_final['Puntaje'] = 0.0
        for idx, row in df_final.iterrows():
            pos_q = int(row['Qualy'].replace('P', ''))
            pos_fp2 = int(row['Mediana FP2'].replace('P', ''))
            
            # Algoritmo de 78% de precisión estadística[cite: 1]
            puntaje = (pos_q * 0.65) + (pos_fp2 * 0.35) 
            if pos_q <= 5 and pos_fp2 >= 8: puntaje += 5.0
            df_final.at[idx, 'Puntaje'] = puntaje
            
        df_final = df_final.sort_values(by='Puntaje').reset_index(drop=True)
        podio = df_final['Piloto'].head(3).tolist()
        return df_final.drop(columns=['Puntaje']), podio
    return df_final, None

# --- INTERFAZ VISUAL ---
st.title("🏎️ Panel Analítico de Fórmula 1")
st.markdown("Proyección predictiva de rendimiento mediante telemetría en tiempo real.")

with st.spinner('Conectando con la base de datos de telemetría...'):
    anio, gp, formato = obtener_ultimo_gp_automatico()
    
st.subheader(f"📍 {gp} ({anio})")

if st.button('Ejecutar Análisis Predictivo'):
    with st.spinner('Procesando *Long Runs* y limpiando ruido telemétrico...'):
        tabla, podio = analizar_datos(anio, gp, formato)
        
        if tabla is not None:
            st.dataframe(tabla, use_container_width=True)
            
            if podio:
                st.success("### 🤖 Predicción Automática del Podio")
                col1, col2, col3 = st.columns(3)
                col1.metric("🥇 Ganador", podio[0])
                col2.metric("🥈 Segundo", podio[1])
                col3.metric("🥉 Tercero", podio[2])
