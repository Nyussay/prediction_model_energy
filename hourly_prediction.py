import csv

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/prediction/hourly')
def get_hourly_prediction():
    start_time = ''
    end_time = ''
    with open('21_01_2019.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            date = row[0]
            time = row[1]
            amount = int(row[3])
            if amount > 20:
                if not start_time:
                    start_time = time
                    end_time = time
                else:
                    end_time = time
            elif start_time:
                break

    prediction_result = {
        'start_time': start_time,
        'end_time': end_time,
    }

    return jsonify(prediction_result)


if __name__ == '__main__':
    app.run()
