# Text Toxicity Analysis 📝

This project provides a web application to analyze the toxicity of text inputs using a pre-trained BERT model. The application classifies text into multiple categories of toxicity including `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

## Features

- **Text Analysis**: Classifies text input into multiple categories of toxicity.
- **Interactive Web Interface**: Uses Streamlit for a user-friendly interface.
- **Model Training**: Includes a Jupyter notebook for training the BERT model on a custom dataset.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Steps

1. **Clone the repository**:

    ```sh
    git clone https://github.com/kvba1/Text-Toxicity-Analysis
    cd text-toxicity-analysis
    ```

2. **Install dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model**:

    Place your trained model (`toxic.pt`) in the `./model` directory.

4. **Run the Streamlit app**:

    ```sh
    streamlit run app.py
    ```

## Usage

1. **Enter Text**: Type or paste the text you want to analyze into the input box.
2. **Analyze**: Click the "Analyze" button to classify the text.
3. **View Results**: The app displays the classification results and maintains a history of analyzed texts.

## Training the Model

To train the model, you can use the provided Jupyter notebook `train.ipynb`:

1. **Open the notebook**:

    ```sh
    jupyter notebook train.ipynb
    ```

2. **Follow the instructions**: The notebook contains detailed steps for training the BERT model on a toxicity dataset.

## File Structure

- `app.py`: The main application file for the Streamlit app.
- `train.ipynb`: Jupyter notebook for training the BERT model.
- `requirements.txt`: List of Python dependencies.
- `model/toxic.pt`: Pre-trained model weights (not included, needs to be downloaded separately).
- `README.md`: Project documentation.

## Example

### Input

    I am friendly

### Output

    | toxic | severe_toxic | obscene | threat | insult | identity_hate |
    |-------|--------------|---------|--------|--------|---------------|
    |   0   |       0      |    0    |   0    |    0   |       0       |

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Streamlit](https://streamlit.io/)
- [Jupyter](https://jupyter.org/)

