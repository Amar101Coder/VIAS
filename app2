from flask import Flask, send_file

# 1️⃣ Create the Flask app first
app = Flask(__name__)

# 2️⃣ Define routes AFTER app is created
@app.route("/")
def index():
    return send_file("index.html")

# 3️⃣ Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
