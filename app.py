from flask import Flask, render_template, url_for, request, redirect  # import Flask class
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle

app = Flask(__name__, template_folder='../templates', static_folder='../static') # Create a Flask instance
# name or module. For single application it'd be __name__

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app) ##initialises the database

model = pickle.load(open('model.pkl', 'rb'))
vector1 = pickle.load(open('model1.pkl','rb'))

class reviews(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    movie_review = db.Column(db.String(300), nullable = False)
    prediction = db.Column(db.String(30))

    def __repr__(self):
        return '<Task %r>' % self.id
        
# Decorator takes another function and extends the behavior of the latter function without explicitly modifying it.
@app.route('/', methods=['POST','GET']) # decorator to tell Flask what URL should trigger the function below
def home(): # function that will return results for the webpage
    global pred
    if request.method == 'POST':
        task_content = request.form['review']
        prediction = model.predict(vector1.transform([task_content]).toarray())
        pred = int(prediction[0])
        new_task = reviews(movie_review=task_content, prediction = pred)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/result.html')
        except:
            "There was an issue adding your task"

    else:
        data()
        #tasks = reviews.query.order_by(reviews.id).all()
        return render_template('home.html')

@app.route('/home.html')
def back():
   return render_template('home.html') 

@app.route('/data.html')
def data():
    tasks = reviews.query.order_by(reviews.id).all()
    return render_template('data.html', tasks = tasks)


@app.route('/result.html')
def result():
    return render_template('result.html', value = pred)


if __name__ == '__main__': 
    app.run(debug = True) # starts development serve