import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
	page_title = "Klasifikasi Level Coding Siswa"
)

model = joblib.load("model_level_coding_siswa.joblib")

st.title("Klasifikasi Level Coding Siswa")
st.markdown("Analisis level coding siswa")

hours_coding_daily = st.slider("Hours Coding Daily", 1.0, 6.0, 3.5)
typing_speed = st.slider("Typing Daily", 20.0, 70.0, 40.0)
prefered_language = st.pills("Prefered Language", ["Python", "Java", "C++"], default = ["Java"])
import_usage = st.pills("Import Usage", ["Yes", "No"], default = ["Yes"])
oop_usage = st.pills("OOP Usage", ["Yes", "No"], default = ["No"])

if st.button("Level", type="primary") :
	data_baru = pd.DataFrame([[4.0, "Java", 48, "Yes", "No", "Advanced"]], columns=["hours_coding_daily","preferred_language","typing_speed","import_usage","oop_usage","level"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi {prediksi} dengan tingkat akurasi {presentase*100:.2f}%")
	st.balloons()
st.divider()
st.caption("Dibuat oleh Adzkia")