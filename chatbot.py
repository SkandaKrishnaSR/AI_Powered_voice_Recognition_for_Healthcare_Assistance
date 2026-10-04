import json
import os
import csv
import pyttsx3
import logging
from datetime import datetime
from difflib import get_close_matches
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, List
from spellchecker import SpellChecker
import random
import re  # NEW: For advanced text processing

# ---------------- Global Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class HealthChatbot:
    """
    Health Chatbot: Responds to user inputs about symptoms, using JSON data and/or ML model.
    Features: TTS, session-based conversations, health tips, capabilities, ML predictions.
    """
    def __init__(
        self,
        json_files: Optional[List[str]] = None,
        ml_model_file: str = "backend/models/best_model.pkl",
        train_csv: str = "backend/data/Training.csv",
        test_csv: str = "backend/data/Testing.csv",
        log_file: str = "backend/logs/chat_history.csv",
        tts_enabled: bool = True,
    ):
        self.diseases = {}
        self.session_flow = {}
        self.tts_enabled = tts_enabled
        self.engine: Optional[pyttsx3.Engine] = None
        self.spell = SpellChecker()
        self.awaiting_symptom_input = {}  # Track sessions waiting for symptom input

        # ------------------ TTS engine ------------------
        if self.tts_enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 160)
                logging.info("✅ TTS engine initialized.")
            except Exception as e:
                logging.error(f"❌ TTS engine failed: {e}")
                self.engine = None

        # ------------------ Synonyms (ENHANCED) ------------------
        self.synonyms = {
            "stomach ache": "stomach_pain",           # Now maps to actual ML symptom
            "stomach pain": "stomach_pain",
            "tummy pain": "stomach_pain",
            "tummy ache": "stomach_pain",
            "abdominal pain": "stomach_pain",
            "belly ache": "stomach_pain",
            "belly pain": "stomach_pain",
            "gastric pain": "stomach_pain",
            "gut pain": "stomach_pain",
            "leg ache": "leg_pain",
            "knee ache": "knee_pain",
            "head ache": "headache",
            "headache": "headache",
            "pimples": "acne",
            "whiteheads": "acne",
            "cold": "common_cold",
            "runny nose": "common_cold",
            "sneezing": "common_cold",
            "fever": "fever",
            "high temperature": "fever",
            "body pain": "body_pain",
            "body ache": "body_pain",
            "muscle pain": "body_pain",
            "joint pain": "joint_pain",
            "chest pain": "chest_pain",
            "back pain": "back_pain",
            "nausea": "nausea",
            "vomiting": "vomiting",
            "diarrhea": "diarrhoea",
            "cough": "cough",
            "sore throat": "sore_throat",
            "fatigue": "fatigue",
            "tiredness": "fatigue",
            "itching": "itching",
            "rash": "skin_rash",
            "skin rash": "skin_rash"
        }

        # ------------------ Health Tips ------------------
        self.general_tips = [
            "Drink at least 8 glasses of water daily.",
            "Eat more fruits and vegetables for a balanced diet.",
            "Exercise at least 30 minutes every day.",
            "Get 7-8 hours of sleep every night.",
            "Wash your hands regularly to prevent infections.",
            "Avoid excessive sugar and junk food.",
            "Take short breaks if working on a computer for long periods.",
            "Practice meditation or deep breathing to reduce stress.",
            "Maintain a healthy body weight.",
            "Visit your doctor regularly for check-ups."
        ]

        # ------------------ Capabilities ------------------
        self.capabilities = [
            "Provide information about symptoms of common diseases",
            "Give general health tips and advice",
            "Predict possible disease based on symptoms using ML",
            "Answer questions about medicines and remedies",
            "Handle session-based conversations about your health",
            "Respond to greetings and thanks politely",
            "Provide personalized advice if your profile is set"
        ]

        # ------------------ CSV logging ------------------
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_message", "bot_response"])

        # ------------------ Load JSON datasets ------------------
        if json_files:
            for file in json_files:
                self.load_json_file(file)

        # ------------------ Load or Train ML model ------------------
        self.model = None
        self.top_features = []
        if os.path.exists(ml_model_file):
            try:
                self.model = joblib.load(ml_model_file)
                train_data = pd.read_csv(train_csv)
                self.top_features = [col for col in train_data.columns if col != "prognosis"]
                logging.info(f"✅ ML model loaded from {ml_model_file}")
            except Exception as e:
                logging.error(f"❌ Error loading ML model: {e}")
        else:
            self.train_ml(train_csv, test_csv, ml_model_file)

        # ---------------- NEW: Enhanced symptom keywords for ML fallback ----------------
        self.common_symptom_keywords = {
            "stomach_pain", "abdominal_pain", "vomiting", "nausea", "diarrhoea", "headache",
            "fever", "cough", "fatigue", "skin_rash", "itching", "joint_pain", "chest_pain",
            "back_pain", "muscle_pain", "body_pain", "sore_throat", "runny_nose", "chills"
        }

    # ---------------- Load JSON ----------------
    def load_json_file(self, file: str):
        try:
            logging.info(f"📂 Loading JSON: {file}")
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            diseases_data = data.get("diseases", data)
            for name, details in diseases_data.items():
                key = name.lower().replace(" ", "_")
                if key in self.diseases:
                    self.diseases[key].update(details)
                else:
                    self.diseases[key] = details
            logging.info(f"✅ Loaded {file}. Total diseases: {len(self.diseases)}")
        except Exception as e:
            logging.error(f"❌ Error loading JSON {file}: {e}")

    # ---------------- Normalize Input (ENHANCED) ----------------
    def normalize_input(self, text: str) -> str:
        text = text.lower().strip()
        # Remove extra punctuation and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply synonym replacement
        for k, v in self.synonyms.items():
            text = text.replace(k, v)
        return text.replace(" ", "_")

    # ---------------- Yes/No Interpretation ----------------
    def interpret_yes_no(self, text: str) -> Optional[str]:
        text = text.lower()
        yes_words = ["yes", "yeah", "yep", "i have", "sure", "correct", "ya", "yup", "ok"]
        no_words = ["no", "nope", "nah", "not really", "don't", "dont"]
        if any(word in text for word in yes_words):
            return "yes"
        elif any(word in text for word in no_words):
            return "no"
        return None

    # ---------------- Find Disease (ENHANCED) ----------------
    def find_disease(self, text: str) -> Optional[str]:
        text = self.normalize_input(text)
        words = text.split("_")
        
        # Spell correction on each word
        corrected_words = []
        for w in words:
            if w:
                correction = self.spell.correction(w)
                corrected_words.append(correction if correction else w)
        text_corrected = "_".join(corrected_words)
        
        # Exact match
        if text_corrected in self.diseases:
            return text_corrected
        
        # Partial match in disease name
        for disease in self.diseases.keys():
            if text_corrected in disease or disease in text_corrected:
                return disease
        
        # Fuzzy match with higher tolerance for short terms
        cutoff = 0.6 if len(text_corrected) <= 6 else 0.75
        matches = get_close_matches(text_corrected, self.diseases.keys(), n=3, cutoff=cutoff)
        return matches[0] if matches else None

    # ---------------- NEW: Extract symptoms more intelligently ----------------
    def extract_symptoms_from_text(self, text: str) -> List[str]:
        normalized = self.normalize_input(text)
        words = normalized.split("_")
        
        found_symptoms = set()
        
        # Direct match with top_features (ML symptoms)
        for word in words:
            if word in self.top_features:
                found_symptoms.add(word)
            else:
                # Try spell correction
                corrected = self.spell.correction(word)
                if corrected and corrected in self.top_features:
                    found_symptoms.add(corrected)
        
        # Also check common symptom keywords
        for sym in self.common_symptom_keywords:
            if sym in normalized:
                found_symptoms.add(sym)
        
        return list(found_symptoms)

    # ---------------- ML Training ----------------
    def train_ml(self, train_file: str, test_file: str, model_file: str):
        logging.info("🔄 Training ML model...")
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        X_train = train_data.drop("prognosis", axis=1)
        y_train = train_data["prognosis"]
        X_test = test_data.drop("prognosis", axis=1)
        y_test = test_data["prognosis"]
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test)
        self.model = rf
        self.top_features = X_train.columns.tolist()
        logging.info(f"✅ RandomForest trained with accuracy {accuracy:.2f}")
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        joblib.dump(self.model, model_file)
        logging.info(f"💾 Model saved at {model_file}")

    # ---------------- ML Prediction ----------------
    def predict_disease_ml(self, user_symptoms: List[str]) -> Optional[str]:
        if not self.model or not self.top_features:
            return None
        input_data = {feat: 0 for feat in self.top_features}
        for sym in user_symptoms:
            if sym in input_data:
                input_data[sym] = 1
        df = pd.DataFrame([input_data])
        return self.model.predict(df)[0]

    # ---------------- Logging ----------------
    def log_conversation(self, user_message: str, bot_response: str):
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_message, bot_response])

    # ---------------- Speak ----------------
    def speak(self, text: str):
        if self.tts_enabled and self.engine and text.strip():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"❌ TTS error: {e}")

    # ---------------- Get Random Health Tip ----------------
    def get_health_tip(self) -> str:
        return random.choice(self.general_tips)

    # ---------------- Show Capabilities ----------------
    def show_capabilities(self) -> str:
        response = "Here’s what I can do:\n"
        for idx, cap in enumerate(self.capabilities, 1):
            response += f"{idx}. {cap}\n"
        return response.strip()

    # ---------------- Respond (ENHANCED NLP) ----------------
    def respond(self, user_input: str, session_id="default") -> str:
        text = self.normalize_input(user_input)
        original_input = user_input.lower()
        greetings = ["hi", "hello", "hey", "good_morning", "good_evening"]
        thanks = ["thanks", "thank_you", "ok", "okay"]

        # ---------------- Symptom Help ----------------
        if "symptom help" in user_input.lower():
            self.awaiting_symptom_input[session_id] = True
            response = "Sure! Type the symptom you want help with, e.g., 'fever', 'cold', 'body pain'."
            self.log_conversation(user_input, response)
            self.speak(response)
            return response

        # ---------------- Awaiting Symptom Input ----------------
        if self.awaiting_symptom_input.get(session_id):
            self.awaiting_symptom_input[session_id] = False
            disease_name = self.find_disease(text)
            if disease_name and disease_name in self.diseases:
                symptoms = self.diseases[disease_name].get("symptoms", [])
                response = f"The symptoms of {disease_name.replace('_',' ')} are: {', '.join(symptoms)}."
            else:
                response = f"Sorry, I do not have information about {text.replace('_', ' ')}."
            self.log_conversation(user_input, response)
            self.speak(response)
            return response

        # ---------------- Capabilities Trigger ----------------
        if "what can you do" in user_input.lower() or "capabilities" in user_input.lower():
            response = self.show_capabilities()
            self.log_conversation(user_input, response)
            self.speak(response)
            return response

        # ---------------- Health Tips Trigger ----------------
        if "health_tip" in text or "tip" in text or "advice" in text:
            tip = self.get_health_tip()
            self.log_conversation(user_input, tip)
            self.speak(tip)
            return tip

        # ---------------- ENHANCED: Symptoms Query (Handles both disease and symptom queries) ----------------
        if "symptoms_of" in text or "symptoms of" in user_input.lower() or "symptoms" in user_input.lower():
            query = None
            if "symptoms of" in user_input.lower():
                query = user_input.lower().split("symptoms of")[-1].strip()
            elif "symptoms" in user_input.lower():
                parts = user_input.lower().split("symptoms")
                if len(parts) > 1:
                    query = parts[-1].strip()
                elif "of" in parts[0]:
                    query = parts[0].split("of")[-1].strip()

            if query:
                # Try to match as a DISEASE first
                normalized_query = self.normalize_input(query)
                disease_name = self.find_disease(normalized_query)

                if disease_name and disease_name in self.diseases:
                    symptoms_list = self.diseases[disease_name].get("symptoms", [])
                    if symptoms_list:
                        response = (
                            f"The common symptoms of **{disease_name.replace('_', ' ')}** are:\n\n"
                            + "\n• ".join(symptoms_list) + "\n\n"
                            "⚠️ Always consult a doctor for accurate diagnosis."
                        )
                    else:
                        response = f"I have information about {disease_name.replace('_', ' ')}, but no detailed symptoms list."
                else:
                    # It's likely a SYMPTOM → use ML to predict possible diseases
                    symptom_list = self.extract_symptoms_from_text(query)
                    if symptom_list:
                        predicted = self.predict_disease_ml(symptom_list)
                        symptom_name = query.strip()
                        response = (
                            f"\"{symptom_name.title()}\" is a **symptom**, not a disease.\n\n"
                            f"It is commonly associated with conditions such as:\n"
                            f"• **{predicted.replace('_', ' ')}**\n\n"
                            f"Other possible causes include injury, strain, arthritis, or poor circulation.\n"
                            f"⚠️ If the pain persists or worsens, please consult a doctor immediately."
                        )
                    else:
                        response = (
                            f"I'm sorry, I couldn't find specific information about \"{query}\".\n"
                            "Please try describing your symptoms more clearly, e.g., 'I have leg pain and swelling'."
                        )
            else:
                response = "Please specify which disease or symptom you'd like to know about."

            self.log_conversation(user_input, response)
            self.speak(response)
            return response

        # ---------------- Greetings & Thanks ----------------
        if any(g in text for g in greetings):
            response = "Hello! How can I assist you with your health today?"
            self.log_conversation(user_input, response)
            self.speak(response)
            return response
        if any(t in text for t in thanks):
            response = "You're welcome! Stay healthy 💙."
            self.log_conversation(user_input, response)
            self.speak(response)
            return response

        # ---------------- Session & Flow ----------------
        if session_id in self.session_flow:
            disease, q_index, ask_medicine = self.session_flow[session_id]
            flow = self.diseases[disease].get("flow", [])
            if ask_medicine:
                answer = self.interpret_yes_no(text)
                if answer == "yes":
                    meds = "\n".join(self.diseases[disease].get("medicines", []))
                    disclaimer = self.diseases[disease].get("disclaimer", "")
                    response = f"{meds}\n{disclaimer}".strip()
                    del self.session_flow[session_id]
                elif answer == "no":
                    response = "Alright! Stay healthy 💙."
                    del self.session_flow[session_id]
                else:
                    response = "Please answer with 'yes' or 'no'."
                self.log_conversation(user_input, response)
                self.speak(response)
                return response
            if q_index < len(flow):
                answer = self.interpret_yes_no(text)
                if answer:
                    reply = flow[q_index].get(answer)
                    if reply:
                        if reply.startswith("Next:"):
                            response = reply.replace("Next:", "").strip()
                            self.session_flow[session_id] = (disease, q_index + 1, False)
                        else:
                            response = f"{reply}\n{self.diseases[disease].get('medicine_prompt','Do you want medicine advice?')}"
                            self.session_flow[session_id] = (disease, q_index + 1, True)
                    else:
                        response = flow[q_index].get("question", "")
                        self.session_flow[session_id] = (disease, q_index + 1, False)
                else:
                    response = flow[q_index].get("question", "Please answer with 'yes' or 'no'.")
                self.log_conversation(user_input, response)
                self.speak(response)
                return response

        # ---------------- New Session or ML Fallback (ENHANCED) ----------------
        disease = self.find_disease(text)
        if disease:
            flow = self.diseases[disease].get("flow", [])
            if flow:
                self.session_flow[session_id] = (disease, 0, False)
                response = flow[0].get("question", "Can you tell me more about your symptoms?")
            else:
                ask_medicine = True if self.diseases[disease].get("medicines") else False
                self.session_flow[session_id] = (disease, 0, ask_medicine)
                advice = self.diseases[disease].get("remedies", "Here is some advice for you.")
                response = advice
                if ask_medicine:
                    response += f"\n{self.diseases[disease].get('medicine_prompt','Do you want medicine advice?')}"
        else:
            # ENHANCED ML fallback with better symptom extraction
            user_symptom_list = self.extract_symptoms_from_text(user_input)
            
            if not user_symptom_list:
                # Final friendly fallback
                response = (
                    "I'm sorry, I couldn't clearly understand your symptoms. "
                    "Could you please describe them more specifically? "
                    "For example: 'I have stomach pain and nausea' or 'fever and cough'."
                )
            else:
                predicted_disease = self.predict_disease_ml(user_symptom_list)
                symptom_names = [s.replace("_", " ") for s in user_symptom_list]
                response = (
                    f"Based on your symptoms ({', '.join(symptom_names)}), "
                    f"the predicted condition could be: **{predicted_disease.replace('_', ' ')}**.\n\n"
                    "⚠️ This is an AI prediction only — please consult a doctor for proper diagnosis."
                )

        self.log_conversation(user_input, response)
        self.speak(response)
        return response
