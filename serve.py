import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import torch
import random
import logging

# Initialize the Flask app
app = Flask(__name__)
model = None
accelerator = None

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# initialize the model
def load_model(model_path, accelerator):
    model_path = os.path.join('/opt/ml/model', 'best_model.pth')
    model = torch.load(model_path, map_location=torch.device(accelerator))
    model.eval()
    return model


@app.route('/ping', methods=['GET'])
def ping():
    # Check if the model is loaded correctly
    health = 'healthy' if model else 'unhealthy'
    status = 200 if model else 404
    return jsonify(status=status, health=health)

@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        # Get the input data
        input_data = request.get_json()
        logging.info(f'Received input data: {input_data}')
        
        # Convert input data to DataFrame
        data = pd.DataFrame(input_data)
        logging.info(f'Converted input data to DataFrame: {data}')
        
        # Make predictions - Replace with your actual prediction logic
        predictions = model.predict(data, return_y=True, trainer_kwargs=dict(accelerator=accelerator))
        
        # Example: Convert TensorFlow tensors to NumPy arrays and flatten
        new_prediction_output = predictions.output[0].cpu().numpy().flatten()
        new_prediction_y = predictions.y[0].cpu().numpy().flatten()
        
        # Create a DataFrame from predictions
        new_predicted_df = pd.DataFrame({
            'prediction_output': new_prediction_output,
            'prediction_y': new_prediction_y
        })
        
        # Convert DataFrame to JSON
        new_predicted_json = new_predicted_df.to_json(orient='records')
        
        logging.info(f'Generated predictions: {new_predicted_json}')
        
        # Return the predictions as JSON
        return new_predicted_json
    except Exception as e:
        logging.error(f'Error during prediction: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using accelerator: {accelerator}')
    set_seed(42)
    model = load_model('/opt/ml/model', accelerator)
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)
