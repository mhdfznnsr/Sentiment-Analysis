from flask import Flask,request,jsonify,render_template,redirect,url_for
import pickle
from sklearn.feature_extraction.text import CountVectorizer

with open('E:/Projects/Sentiment and Category/models/Sentiment_Prediction.pkl','rb') as f:
    model = pickle.load(f)

with open('E:/Projects/Sentiment and Category/models/trained_vectorizer.pkl','rb') as f:
    vec = pickle.load(f)


app = Flask(__name__)

@app.route('/')
def home():
    title_name = "Flask Page"
    return render_template('index.html',title_name = title_name)

@app.route("/",methods = ['POST','GET'])
def prediction():
    review = [request.form['review_1']]
    review_vec = vec.transform(review)
    prediction = model.predict(review_vec)
    return render_template("index.html",answer = prediction[0])

if __name__ == "__main__":
    app.run(debug=True)