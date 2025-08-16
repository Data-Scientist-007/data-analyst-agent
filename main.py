from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
from bs4 import BeautifulSoup
import re
from io import StringIO
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Safe parser for gross values
def parse_gross(gross_str):
    if pd.isna(gross_str):
        return None
    number = re.sub(r"[^\d.]", "", gross_str)
    try:
        return float(number)
    except:
        return None

# Scrape and parse the Wikipedia table
def scrape_wikipedia_table(url):
    res = requests.get(url)
    tables = pd.read_html(StringIO(res.text))
    df = tables[0]
    return df

# Web UI (GET): Upload form
@app.route("/", methods=["GET"])
def home():
    return '''
    <h2>Test Data Analyst Agent</h2>
    <form action="/api/" method="post" enctype="multipart/form-data">
        <label>Select questions.txt:</label><br>
        <input type="file" name="questions.txt" required><br><br>
        <input type="submit" value="Submit Task">
    </form>
    '''

# API + HTML UI output (POST)
@app.route("/api/", methods=["POST"])
def analyze():
    try:
        qfile = request.files.get("questions.txt")
        if not qfile:
            return "questions.txt file is required", 400

        questions = qfile.read().decode("utf-8").strip().split("\n")
        url = next((line.strip() for line in questions if "http" in line), None)
        if not url:
            return "URL not found in question file", 400

        df = scrape_wikipedia_table(url)
        df.columns = [col.strip() for col in df.columns]

        # Process Gross
        if "Worldwide gross" in df.columns:
            df["Gross"] = df["Worldwide gross"].apply(parse_gross)
        elif "Gross" in df.columns:
            df["Gross"] = df["Gross"].apply(parse_gross)
        else:
            return "Gross column not found", 500

        if "Title" in df.columns:
            df["Title"] = df["Title"].astype(str)
        if "Year" not in df.columns:
            df["Year"] = df["Title"].str.extract(r"\((\d{4})\)").astype(float)

        # Q1
        q1 = df[(df["Gross"] >= 2_000_000_000) & (df["Year"] < 2000)]
        answer1 = len(q1)

        # Q2
        q2 = df[df["Gross"] > 1_500_000_000].sort_values("Year")
        answer2 = q2.iloc[0]["Title"] if not q2.empty else "Not found"

        # Q3
        if "Rank" in df.columns and "Peak" in df.columns:
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
            df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")
            corr = df[["Rank", "Peak"]].dropna().corr().iloc[0, 1]
        else:
            corr = None
        answer3 = round(corr, 6) if corr is not None else None

        # Q4: Plot
        fig, ax = plt.subplots()
        scatter_df = df[["Rank", "Peak"]].dropna()
        data_uri = ""
        if not scatter_df.empty:
            X = scatter_df["Rank"].values.reshape(-1, 1)
            y = scatter_df["Peak"].values
            ax.scatter(X, y, label="Data")

            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            ax.plot(X, y_pred, linestyle="dotted", color="red", label="Regression")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Peak")
            ax.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            data_uri = "data:image/png;base64," + img_base64

        # If request was from browser, render HTML
        if "curl" not in request.headers.get("User-Agent", "").lower():
            return render_template_string("""
            <h2>Results</h2>
            <ul>
                <li><b>1. How many $2B movies before 2000?</b> {{ a1 }}</li>
                <li><b>2. Earliest film > $1.5B?</b> {{ a2 }}</li>
                <li><b>3. Correlation between Rank and Peak?</b> {{ a3 }}</li>
                <li><b>4. Scatterplot:</b><br><img src="{{ img }}" width="600"></li>
            </ul>
            <br><a href="/">⬅️ Back</a>
            """, a1=answer1, a2=answer2, a3=answer3, img=data_uri)

        # Otherwise return raw JSON (for curl)
        return jsonify([answer1, answer2, answer3, data_uri])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
