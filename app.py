from flask import Flask, request, jsonify, send_from_directory, render_template_string, url_for
import os
from real import InferenceServer

app = Flask(__name__)
UPLOAD_FOLDER = 'C:\\Users\\ludal\\PycharmProjects\\pythonProject\\test'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
inference_server = InferenceServer('test/')
inference_results = {}
gesture_mapping = {
    0: 'push',
    1: 'down',
    2: 'push',
    3: 'push',
    4: 'down'
}
@app.route('/')
def home():
    return 'Welcome to my Flask app!'


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request'
        file = request.files['file']
        if file and file.filename != '':
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_index = inference_server.inference(filepath, verbose=True)
            result_index = result_index[0] if isinstance(result_index, list) else result_index
            result = gesture_mapping.get(result_index, 'Unknown gesture')  # 将索引转换为手势名称


            image_filename = f'{result}.png' if os.path.exists(
                os.path.join(app.static_folder, 'images', f'{result}.png')) else '.png'
            image_url = url_for('static', filename=f'images/{image_filename}')

            explanations = {
                'push': 'pushing and pulling (PP)',
                'left': 'waving left and right (WLR)',
                'down': 'waving up and down (WUD)',
                'stretch': 'stretching each finger (SF)',
                'circle': 'circling clockwise (CC)'
            }
            #

            # Store the results of reasoning
            inference_results[filename] = {
                'result': result,
                'explanation': explanations.get(result, 'Unknown gesture'),
                'image_url': image_url
            }
            # Returns an HTML page containing an image
            return jsonify({'message': 'File uploaded successfully', 'filename': filename})

        else:
            return 'No selected file'
    else:
        # For GET requests
        return render_template_string('''
                   <h1>Uploaded Files and Inference Results</h1>
                   <ul>
                       {% for filename, details in inference_results.items() %}
                           <li>{{ filename }} - Result: {{ details['result'] }}
                               <p>{{ details['explanation'] }}</p>
                               <img src="{{ details['image_url'] }}" alt="result image">
                           </li>
                       {% endfor %}
                   </ul>
                   
               ''', inference_results=inference_results)

@app.route('/uploads')
def list_uploads():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string('''
        <h1>Uploaded Files</h1>
        <ul>
            {% for file in files %}
                <li><a href="{{ url_for('get_file', filename=file) }}">{{ file }}</a></li>
            {% endfor %}
        </ul>
    ''', files=files)
    return 'Here will be listed the uploaded files.'

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    with app.test_request_context():
        print(app.url_map)
    app.run(host='0.0.0.0', port=8000)

