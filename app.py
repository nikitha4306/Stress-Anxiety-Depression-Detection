from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import numpy as np
import random
from flask_session import Session

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load models
models = {
    "Stress": joblib.load("models/best_stress_model.joblib"),
    "Anxiety": joblib.load("models/best_anxiety_model.joblib"),
    "Depression": joblib.load("models/best_depression_model.joblib")
}

# DASS-21 Questions
questions = {
    1: "I found it hard to wind down.",
    2: "I was aware of dryness of my mouth.",
    3: "I couldn’t seem to experience any positive feeling at all.",
    4: "I experienced breathing difficulty.",
    5: "I found it difficult to work up the initiative to do things.",
    6: "I tended to over-react to situations.",
    7: "I experienced trembling.",
    8: "I felt that I was using a lot of nervous energy.",
    9: "I was worried about situations in which I might panic.",
    10: "I felt that I had nothing to look forward to.",
    11: "I found myself getting agitated.",
    12: "I found it difficult to relax.",
    13: "I felt down-hearted and blue.",
    14: "I was intolerant of anything that kept me from getting on.",
    15: "I felt I was close to panic.",
    16: "I was unable to become enthusiastic about anything.",
    17: "I felt I wasn’t worth much as a person.",
    18: "I felt that I was rather touchy.",
    19: "I was aware of the beating of my heart.",
    20: "I felt scared without any good reason.",
    21: "I felt that life was meaningless."
}

level_mapping = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Extremely Severe"
}

recommendations = {
    "Normal": "Your responses do not indicate significant symptoms in this area. Continue regular self‑care and healthy coping habits.",
    "Mild": "You report some symptoms. Monitor how you feel, use relaxation and self‑care, and try to maintain sleep, food, and routine.",
    "Moderate": "Your symptoms are noticeable. Consider talking to a trusted person, counselor, or student support service if available.",
    "Severe": "Your symptoms are high. It is advisable to consult a mental health professional as soon as you can.",
    "Extremely Severe": "Your symptoms are very high. Please seek professional help immediately or contact a helpline if you feel unsafe."
}

MAX_QUESTIONS = 10  # always ask 10 items

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    # reset session state
    session["answers"] = {}
    session["sequence"] = random.sample(list(questions.keys()), MAX_QUESTIONS)
    session["current_index"] = 0
    return redirect(url_for("question"))

@app.route("/question", methods=["GET", "POST"])
def question():
    answers = session.get("answers", {})
    sequence = session.get("sequence", [])
    current_index = session.get("current_index", 0)

    # Save previous answer
    if request.method == "POST":
        qid = int(request.form.get("qid"))
        score = int(request.form.get("score"))
        answers[qid] = score
        session["answers"] = answers
        current_index += 1
        session["current_index"] = current_index

    # If all 10 questions answered -> go to result
    if current_index >= MAX_QUESTIONS or not sequence:
        return redirect(url_for("result"))

    # Get next question id from fixed sequence
    qid = sequence[current_index]
    question_text = questions[qid]
    progress = int(current_index / MAX_QUESTIONS * 100)

    return render_template(
        "question.html",
        qid=qid,
        question=question_text,
        progress=progress
    )

@app.route("/result")
def result():
    answers = session.get("answers", {})
    if not answers:
        return redirect(url_for("index"))

    # build full 21-item vector, missing items as 0
    full_vector = [answers.get(i, 0) for i in range(1, 22)]
    features = np.array(full_vector + [sum(full_vector)]).reshape(1, -1)

    final_results = {}
    for key, model in models.items():
        try:
            pred = int(model.predict(features)[0])
            level = level_mapping.get(pred, "Unknown")
        except Exception:
            avg = sum(full_vector) / max(len(full_vector), 1)
            if avg <= 4:
                level = "Normal"
            elif avg <= 8:
                level = "Mild"
            elif avg <= 12:
                level = "Moderate"
            elif avg <= 16:
                level = "Severe"
            else:
                level = "Extremely Severe"

        tip = recommendations.get(level, "For any concern, please consider talking to a professional.")
        final_results[key] = {"level": level, "tip": tip}

    return render_template("result.html", results=final_results)

if __name__ == "__main__":
    app.run(debug=True)
