import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

SYMPTOMS = [
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever",
    "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
    "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain",
    "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance",
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell",
    "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine",
    "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation",
    "dischromic _patches", "watering_from_eyes", "increased_appetite",
    "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma",
    "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking",
    "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

DISEASES = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis",
    "Drug Reaction", "Peptic ulcer diseae", "AIDS", "Diabetes",
    "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine",
    "Cervical spondylosis", "Paralysis (brain hemorrhage)", "Jaundice",
    "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Alcoholic hepatitis", "Tuberculosis", "Common Cold", "Pneumonia",
    "Dimorphic hemmorhoids(piles)", "Heartattack", "Varicoseveins",
    "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia", "Osteoarthristis",
    "Arthritis", "(vertigo) Paroymsal  Positional Vertigo", "Acne",
    "Urinary tract infection", "Psoriasis", "Impetigo"
]

PROGNOSIS_MAP = {
    "Fungal infection": 0, "Allergy": 1, "GERD": 2,
    "Chronic cholestasis": 3, "Drug Reaction": 4, "Peptic ulcer diseae": 5,
    "AIDS": 6, "Diabetes ": 7, "Diabetes": 7, "Gastroenteritis": 8,
    "Bronchial Asthma": 9, "Hypertension ": 10, "Hypertension": 10,
    "Migraine": 11, "Cervical spondylosis": 12,
    "Paralysis (brain hemorrhage)": 13, "Jaundice": 14, "Malaria": 15,
    "Chicken pox": 16, "Dengue": 17, "Typhoid": 18, "hepatitis A": 19,
    "Hepatitis B": 20, "Hepatitis C": 21, "Hepatitis D": 22,
    "Hepatitis E": 23, "Alcoholic hepatitis": 24, "Tuberculosis": 25,
    "Common Cold": 26, "Pneumonia": 27, "Dimorphic hemmorhoids(piles)": 28,
    "Heart attack": 29, "Heartattack": 29, "Varicose veins": 30,
    "Varicoseveins": 30, "Hypothyroidism": 31, "Hyperthyroidism": 32,
    "Hypoglycemia": 33, "Osteoarthristis": 34, "Arthritis": 35,
    "(vertigo) Paroymsal  Positional Vertigo": 36, "Acne": 37,
    "Urinary tract infection": 38, "Psoriasis": 39, "Impetigo": 40
}


def load_dataset(csv_path, feature_cols):
    data = pd.read_csv(csv_path)
    data = data.replace({"prognosis": PROGNOSIS_MAP})
    features = data[feature_cols]
    labels = data["prognosis"].astype(int).values
    return features, labels


X_train, y_train = load_dataset("Training.csv", SYMPTOMS)
X_test, y_test = load_dataset("Testing.csv", SYMPTOMS)


def build_feature_vector(selected_symptoms):
    vector = np.zeros(len(SYMPTOMS), dtype=int)
    selected_set = {s.strip() for s in selected_symptoms if s and s != "Select Symptom"}
    for idx, sym in enumerate(SYMPTOMS):
        if sym in selected_set:
            vector[idx] = 1
    return vector.reshape(1, -1)


def run_prediction(model_name, model, output_widget):
    chosen = [s1_var.get(), s2_var.get(), s3_var.get(), s4_var.get(), s5_var.get()]
    active_symptoms = [s for s in chosen if s and s != "Select Symptom"]

    if not active_symptoms:
        messagebox.showwarning("Missing Input", "Please select at least one symptom.")
        return

    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    acc = accuracy_score(y_test, test_preds) * 100

    sample_vector = build_feature_vector(active_symptoms)
    pred_idx = int(model.predict(sample_vector)[0])
    predicted_disease = DISEASES[pred_idx] if 0 <= pred_idx < len(DISEASES) else "Not Found"

    patient = patient_var.get().strip() or "Anonymous"
    print(f"[{model_name}] Patient: {patient} | Accuracy: {acc:.2f}% | Prediction: {predicted_disease}")

    output_widget.delete("1.0", tk.END)
    output_widget.insert(tk.END, f"{predicted_disease}  (Acc: {acc:.1f}%)")


# ---------------- GUI Setup ----------------
BG_MAIN = "#0f172a"
BG_CARD = "#1e293b"
ACCENT = "#0ea5e9"
TEXT_LIGHT = "#f8fafc"
TEXT_MUTED = "#cbd5e1"
RESULT_BG = "#f1f5f9"

root = tk.Tk()
root.title("AI Disease Prediction System")
root.configure(bg=BG_MAIN)
root.resizable(True,True)
root.state("zoomed")

patient_var = tk.StringVar()
s1_var = tk.StringVar(value="Select Symptom")
s2_var = tk.StringVar(value="Select Symptom")
s3_var = tk.StringVar(value="Select Symptom")
s4_var = tk.StringVar(value="Select Symptom")
s5_var = tk.StringVar(value="Select Symptom")

symptom_options = ["Select Symptom"] + sorted(SYMPTOMS)

# Header
header_frame = tk.Frame(root, bg=BG_MAIN, padx=24, pady=16)
header_frame.grid(row=0, column=0, columnspan=4, sticky="ew")

tk.Label(
    header_frame,
    text="AI-Based Disease Prediction System",
    font=("Segoe UI", 22, "bold"),
    fg=TEXT_LIGHT,
    bg=BG_MAIN
).pack(anchor="w")

tk.Label(
    header_frame,
    text="Developed by Gunjan Mirchandani  |  ML Symptom Classifier",
    font=("Segoe UI", 11),
    fg=ACCENT,
    bg=BG_MAIN
).pack(anchor="w", pady=(2, 0))

# Input Card
input_frame = tk.LabelFrame(
    root,
    text=" Patient & Symptom Details ",
    font=("Segoe UI", 10, "bold"),
    fg=TEXT_LIGHT,
    bg=BG_CARD,
    padx=16,
    pady=12
)
input_frame.grid(row=1, column=0, columnspan=2, padx=(24, 12), pady=(0, 16), sticky="nsew")

tk.Label(input_frame, text="Patient Name", fg=TEXT_MUTED, bg=BG_CARD, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=6)
tk.Entry(input_frame, textvariable=patient_var, width=28, font=("Segoe UI", 10)).grid(row=0, column=1, padx=(10, 0), pady=6)

symptom_vars = [s1_var, s2_var, s3_var, s4_var, s5_var]
for i, var in enumerate(symptom_vars, start=1):
    tk.Label(input_frame, text=f"Symptom {i}", fg=TEXT_MUTED, bg=BG_CARD, font=("Segoe UI", 10)).grid(row=i, column=0, sticky="w", pady=5)
    menu = tk.OptionMenu(input_frame, var, *symptom_options)
    menu.config(width=24, font=("Segoe UI", 9), bg="#334155", fg=TEXT_LIGHT, highlightthickness=0)
    menu["menu"].config(font=("Segoe UI", 9))
    menu.grid(row=i, column=1, padx=(10, 0), pady=5, sticky="w")

# Model Actions Card
model_frame = tk.LabelFrame(
    root,
    text=" Select Classifier & View Result ",
    font=("Segoe UI", 10, "bold"),
    fg=TEXT_LIGHT,
    bg=BG_CARD,
    padx=16,
    pady=12
)
model_frame.grid(row=1, column=2, columnspan=2, padx=(12, 24), pady=(0, 16), sticky="nsew")

dt_out = tk.Text(model_frame, height=1, width=28, font=("Segoe UI", 10, "bold"), bg=RESULT_BG, fg="#0f172a", padx=6, pady=4)
rf_out = tk.Text(model_frame, height=1, width=28, font=("Segoe UI", 10, "bold"), bg=RESULT_BG, fg="#0f172a", padx=6, pady=4)
nb_out = tk.Text(model_frame, height=1, width=28, font=("Segoe UI", 10, "bold"), bg=RESULT_BG, fg="#0f172a", padx=6, pady=4)

models_config = [
    ("Decision Tree", tree.DecisionTreeClassifier(random_state=42), dt_out, 0),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), rf_out, 2),
    ("Naive Bayes", GaussianNB(), nb_out, 4),
]

for label_text, clf, out_box, row_idx in models_config:
    btn = tk.Button(
        model_frame,
        text=f"Predict ({label_text})",
        font=("Segoe UI", 9, "bold"),
        bg="#0284c7",
        fg="white",
        activebackground="#0369a1",
        activeforeground="white",
        relief="flat",
        padx=10,
        pady=5,
        command=lambda name=label_text, m=clf, box=out_box: run_prediction(name, m, box)
    )
    btn.grid(row=row_idx, column=0, sticky="w", pady=(6, 2))
    out_box.grid(row=row_idx + 1, column=0, sticky="ew", pady=(0, 8))

root.mainloop()