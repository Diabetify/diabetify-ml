import gpt

feature_aliases = {
    "age": "Usia",
    "smoking_status": "Status Merokok",
    "is_cholesterol": "Kolesterol Tinggi",
    "is_macrosomic_baby": "Riwayat Melahirkan Bayi Besar",
    "physical_activity_frequency": "Frekuensi Aktivitas Fisik Sedang",
    "is_bloodline": "Riwayat Keluarga dengan Diabetes",
    "brinkman_score": "Indeks brinkman",
    "BMI": "Indeks Massa Tubuh (BMI)",
    "is_hypertension": "Hipertensi"
}

feature_descriptions = {
    "age": "The user's age in years, represented as a whole number (e.g., 50).",
    
    "smoking_status": (
        "The user's smoking status: "
        "0 = never smoked, 1 = former smoker, 2 = current smoker."
    ),
    
    "is_cholesterol": (
        "Indicates whether the user has been diagnosed with high cholesterol: "
        "0 = no, 1 = yes."
    ),
    
    "is_macrosomic_baby": (
        "Indicates whether the user has given birth to a baby weighing more than 4 kg: "
        "0 = no, 1 = yes, 2 = not applicable (male or never pregnant)."
    ),
    
    "physical_activity_frequency": (
        "The number of days per week the user performs moderate-intensity physical activities. "
    ),
    
    "is_bloodline": (
        "Indicates whether the user's parent has died due to diabetes: "
        "0 = no, 1 = yes."
    ),
    
    "brinkman_score": (
        "Brinkman Index measures lifetime tobacco exposure: "
        "0 = never smoked, 1 = mild smoker, 2 = moderate smoker, 3 = heavy smoker."
    ),
    
    "BMI": (
        "The user's Body Mass Index (BMI), a continuous numeric value (e.g., 20.5), "
        "used to assess whether a person is underweight, normal, overweight, or obese."
    ),
    
    "is_hypertension": (
        "Indicates whether the user has been diagnosed with hypertension (high blood pressure): "
        "0 = no, 1 = yes."
    )
}

shap_and_input_value = {
            "id": 1,
            "created_at": "2025-06-03T15:58:39.855159+07:00",
            "updated_at": "2025-06-03T15:58:39.855159+07:00",
            "user_id": 1,
            "risk_score": 0.31683722138404846,
            "age": 25,
            "age_shap": -0.6137380591867203,
            "age_contribution": 0.6137380591867203,
            "age_impact": 0,
            "age_explanation": "",
            "bmi": 24.7,
            "bmi_shap": 0.07897499662042597,
            "bmi_contribution": 0.07897499662042597,
            "bmi_impact": 1,
            "bmi_explanation": "",
            "brinkman_score": 0,
            "brinkman_score_shap": -0.04491426921184273,
            "brinkman_score_contribution": 0.04491426921184273,
            "brinkman_score_impact": 0,
            "brinkman_score_explanation": "",
            "is_hypertension": False,
            "is_hypertension_shap": -0.06290869667908416,
            "is_hypertension_contribution": 0.06290869667908416,
            "is_hypertension_impact": 0,
            "is_hypertension_explanation": "",
            "is_cholesterol": False,
            "is_cholesterol_shap": -0.06290869667908416,
            "is_cholesterol_contribution": 0.06290869667908416,
            "is_cholesterol_impact": 0,
            "is_cholesterol_explanation": "",
            "is_bloodline": False,
            "is_bloodline_shap": -0.06290869667908416,
            "is_bloodline_contribution": 0.06290869667908416,
            "is_bloodline_impact": 0,
            "is_bloodline_explanation": "",
            "is_macrosomic_baby": 0,
            "is_macrosomic_baby_shap": -0.11950345400760777,
            "is_macrosomic_baby_contribution": 0.11950345400760777,
            "is_macrosomic_baby_impact": 0,
            "is_macrosomic_baby_explanation": "",
            "smoking_status": "2",
            "smoking_status_shap": 0.03380394802865151,
            "smoking_status_contribution": 0.03380394802865151,
            "smoking_status_impact": 1,
            "smoking_status_explanation": "",
            "physical_activity_frequency": 3,
            "physical_activity_frequency_shap": -0.04615657626566752,
            "physical_activity_frequency_contribution": 0.04615657626566752,
            "physical_activity_frequency_impact": 0,
            "physical_activity_frequency_explanation": ""
        }

with open("global.png", "rb") as image_file:
    image = image_file.read()

result = gpt.explain(image, shap_and_input_value, feature_aliases, feature_descriptions)