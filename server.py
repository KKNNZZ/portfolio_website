from flask import Flask, render_template, url_for, request, redirect, jsonify
import csv

app = Flask(__name__)
# print(__name__)


@app.route("/")
def my_home():
    return render_template("index.html")


@app.route("/<string:page_name>")
def html_page(page_name):
    return render_template(page_name)


def write_to_csv(data):
    with open('database.csv', newline='', mode='a') as database2:
        email = data["email"]
        subject = data["subject"]
        message = data["message"]
        csv_writer = csv.writer(database2, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([email,subject,message])


@app.route('/submit_form', methods=['POST', 'GET'])
def submit_form():
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            write_to_csv(data)
            return redirect('/thankyou.html')
        except:
            return 'did not save to database'
    else:
        return 'Something went wrong, try again.'


from password_checker import pwned_api_check

@app.route("/work2.html", methods=["GET", "POST"])
def password():
    count = 0
    suggestions = []
    if request.method == "POST":
        password = request.form.get("password")
        count, suggestions = pwned_api_check(password)

    return render_template("work2.html", count=count, suggestions=suggestions)


import img_classification_models
#import the trained models
@app.route('/upload', methods=['POST'])
def upload():
    if 'imageFile' in request.files:
        try:
            image = request.files['imageFile']
            image.save('uploaded_image.jpg')
            result_mnv2 = img_classification_models.mobilnet_v2('uploaded_image.jpg')
            result_cnn = img_classification_models.simple_CNN('uploaded_image.jpg')
            result_rn50 = img_classification_models.resnet_50('uploaded_image.jpg')
            if result_mnv2 and result_cnn and result_rn50:
                return jsonify({
                    "mnv2": result_mnv2,
                    "cnn": result_cnn,
                    "rn50": result_rn50
                    
                })
            else:
                return 'Classification results are invalid.'
        except Exception as e:
            return f'An error occurred: {str(e)}'

    return 'No image file received.'


if __name__ == '__main__':
    app.run()
