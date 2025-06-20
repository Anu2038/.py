# ATTEMPT 3
# -*- coding: utf-8 -*-
"""
Unified Diet Recommendation System for All Users
- Routes between Non-Special and Special Population
- Predicts Diet Plans using ML model for special cases
- Provides BMI-based recommendations for healthy individuals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import traceback

# ===============================
# SHARED FUNCTIONS
# ===============================

def calculate_bmi(weight, height):
    if height <= 0 or weight <= 0:
        raise ValueError("Height and weight must be positive values")
    return round(weight / (height ** 2), 2)

def get_bmi_category(bmi):
    if bmi < 16.0:
        return "Severely Underweight"
    elif 16.0 <= bmi <= 16.9:
        return "Moderately Underweight"
    elif 17.0 <= bmi <= 18.4:
        return "Mildly Underweight"
    elif 18.5 <= bmi <= 24.9:
        return "Normal Weight"
    elif 25.0 <= bmi <= 29.9:
        return "Overweight"
    elif 30.0 <= bmi <= 34.9:
        return "Obese Class I"
    elif 35.0 <= bmi <= 39.9:
        return "Obese Class II"
    else:
        return "Obese Class III"

def safe_encode(encoder, value, default_value=0):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"⚠️ Warning: '{value}' not seen during training. Using default encoding.")
        return default_value

# ===============================
# SPECIAL POPULATION CODE
# ===============================

HEALTH_CONDITIONS = [
    "None", "Prediabetes / Insulin Resistance", "ADHD", "Arthritis (Osteo/Rheumatoid)",
    "Type 2 Diabetes", "Irritable Bowel Syndrome (IBS)", "Osteoporosis", "Atherosclerosis",
    "COPD / Chronic Bronchitis", "Metabolic Syndrome", "Lactose Intolerance", "Hypertension",
    "Hypothyroidism", "GERD (Reflux)", "Alzheimer's / Dementia", "Obesity", "Overweight",
    "Coronary Artery Disease (CAD)", "Eating Disorder Recovery", "Hyperlipidemia",
    "Multiple Sclerosis (MS)", "Sarcopenia", "Constipation", "Asthma",
    "Depression / Anxiety", "Celiac Disease", "Lupus", "PCOS"
]

def train_special_model(csv_path):
    df = pd.read_csv("/content/special_population_diet_plans.csv")

    breakfast_enc = LabelEncoder()
    lunch_enc = LabelEncoder()
    dinner_enc = LabelEncoder()
    snack_enc = LabelEncoder()

    df["Breakfast"] = breakfast_enc.fit_transform(df["Breakfast"])
    df["Lunch"] = lunch_enc.fit_transform(df["Lunch"])
    df["Dinner"] = dinner_enc.fit_transform(df["Dinner"])
    df["Snack"] = snack_enc.fit_transform(df["Snack"])

    encoders = {}
    cat_cols = ["Name", "Gender", "Activity_Level", "BMI_Category", "Health_Condition", "Dietary_Preference"]
    for col in cat_cols:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])

    X = df.drop(["Name", "Breakfast", "Lunch", "Dinner", "Snack"], axis=1)
    y = df[["Breakfast", "Lunch", "Dinner", "Snack"]]

    scale_cols = [col for col in ["Age", "BMI", "Weight", "Height", "Total_Calories", "Protein", "Carbohydrate", "Fat"] if col in X.columns]
    scaler = MinMaxScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    rf.fit(X, y)

    return rf, scaler, breakfast_enc, lunch_enc, dinner_enc, snack_enc, encoders, X.columns, scale_cols

def predict_special_diet(user_data, model, scaler, encoders, feature_cols, scale_cols,
                         breakfast_enc, lunch_enc, dinner_enc, snack_enc):

    height_m = user_data['height'] / 100
    bmi = user_data['weight'] / (height_m ** 2)
    bmi_category = get_bmi_category(bmi)

    df_input = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
    df_input.at[0, "Age"] = user_data['age']
    df_input.at[0, "Weight"] = user_data['weight']
    df_input.at[0, "Height"] = user_data['height']
    df_input.at[0, "BMI"] = bmi

    if "Health_Condition" in feature_cols:
        df_input.at[0, "Health_Condition"] = safe_encode(encoders["Health_Condition"], user_data['health_condition'])
    if "BMI_Category" in feature_cols:
        df_input.at[0, "BMI_Category"] = safe_encode(encoders["BMI_Category"], bmi_category)

    df_input[scale_cols] = scaler.transform(df_input[scale_cols])
    prediction = model.predict(df_input)

    return {
        "Breakfast": breakfast_enc.inverse_transform([prediction[0][0]])[0],
        "Lunch": lunch_enc.inverse_transform([prediction[0][1]])[0],
        "Dinner": dinner_enc.inverse_transform([prediction[0][2]])[0],
        "Snack": snack_enc.inverse_transform([prediction[0][3]])[0],
        "BMI": round(bmi, 2),
        "Category": bmi_category
    }

# ===============================
# NON-SPECIAL POPULATION CODE
# ===============================

def determine_diet(bmi):
    if bmi < 18.5:
        return 'High Calorie Diet'
    elif 18.5 <= bmi < 25:
        return 'Balanced Diet'
    elif 25 <= bmi < 30:
        return 'Low Calorie Diet'
    else:
        return 'Very Low Calorie Diet'

def non_special_population_code(user):
    height_m = user['height'] / 100
    bmi = calculate_bmi(user['weight'], height_m)
    diet = determine_diet(bmi)

    print(f"\n✅ Processing {user['name']} through NON-SPECIAL POPULATION system...")
    print(f"👤 Name: {user['name']}")
    print(f"📊 Age: {user['age']} years")
    print(f"⚖️ Weight: {user['weight']} kg")
    print(f"📏 Height: {user['height']} cm")
    print(f"🧮 BMI: {bmi:.2f}")
    print(f"\n🍽️ Recommended Diet: {diet}")

    if diet == "High Calorie Diet":
        print("- Focus on nutrient-dense, higher-calorie foods")
        print("- Include healthy fats like nuts, avocados, olive oil")
    elif diet == "Balanced Diet":
        print("- Emphasize whole grains, lean protein, and vegetables")
        print("- Maintain current physical activity")
    elif diet == "Low Calorie Diet":
        print("- Reduce sugary snacks and portion sizes")
        print("- Increase vegetables and fiber")
    else:
        print("- Consult a doctor for very low-calorie planning")
        print("- Focus on protein and fiber-rich foods")

# ===============================
# ROUTING LOGIC
# ===============================

def get_user_input():
    print("\n📋 Please enter your details:")
    name = input("Enter your Name: ")

    try:
        age = int(input("Enter your Age: "))
        weight = float(input("Enter your Weight (kg): "))
        height = float(input("Enter your Height (cm): "))
    except ValueError:
        print("❌ Invalid input. Please enter valid numbers.")
        return None

    print("\n🩺 Do you have any health conditions?")
    print("1. Yes\n2. No")
    try:
        has_condition = int(input("Enter your choice (1 or 2): ")) == 1
    except:
        has_condition = False

    user_data = {
        'name': name,
        'age': age,
        'weight': weight,
        'height': height,
        'has_health_condition': has_condition
    }

    if has_condition:
        print("\nAvailable conditions:")
        for idx, cond in enumerate(HEALTH_CONDITIONS):
            print(f"{idx}. {cond}")
        try:
            cond_idx = int(input("Select your condition number: "))
            user_data['health_condition'] = HEALTH_CONDITIONS[cond_idx] if 0 <= cond_idx < len(HEALTH_CONDITIONS) else "None"
        except:
            user_data['health_condition'] = "None"

    return user_data

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    print("\n🏥 Welcome to the Unified Diet Recommendation System 🏥")
    user = get_user_input()

    if not user:
        print("❌ Failed to collect valid user data. Exiting.")
        exit()

    if user['has_health_condition']:
        try:
            model, scaler, b_enc, l_enc, d_enc, s_enc, encoders, f_cols, s_cols = train_special_model(
                "/content/special_population_diet_plans.csv"
            )
            result = predict_special_diet(user, model, scaler, encoders, f_cols, s_cols, b_enc, l_enc, d_enc, s_enc)

            print(f"\n🏥 Processed {user['name']} through SPECIAL POPULATION system")
            print(f"🧮 BMI: {result['BMI']:.2f} ({result['Category']})")
            print("\n🍽️ Personalized Diet Plan:")
            for meal, item in result.items():
                if meal not in ['BMI', 'Category']:
                    print(f"- {meal}: {item}")

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            traceback.print_exc()
    else:
        non_special_population_code(user)


    print("\n✅ Diet recommendation process completed!")
