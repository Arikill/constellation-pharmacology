# -*- coding: utf-8 -*-

from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def board():
    return render_template("board.html")

if __name__ == "__main__":
    app.run(host="localhost", port="8001", debug=True)