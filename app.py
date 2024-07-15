import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

@st.cache_resource
def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=6
    )
    model.load_state_dict(torch.load('./model/toxic.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', 
                                              do_lower_case=True, 
                                              strip_accents=True, 
                                              clean_text=True)
    return tokenizer

def get_embeddings(tokenizer, input: str):
    input_encoded = tokenizer(
        input,
        max_length=128,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt'
    )
    return input_encoded

st.set_page_config(page_title="Text Toxicity Analysis", page_icon="📝")
model = get_model()
tokenizer = get_tokenizer()

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Text Toxicity Analysis 📝")

user_input = st.text_area("Enter text for toxicity analysis:")

if st.button("Analyze"):
    if user_input:
        with st.spinner('Analyzing...'):
            input_embeddings = get_embeddings(tokenizer, user_input)
            with torch.no_grad():
                output = model(**input_embeddings)
            result = torch.sigmoid(output.logits)
            mapped_result = (result > 0.7).int().numpy().tolist()[0] 
            st.session_state.history.append({"text": user_input, "result": mapped_result})
            df = pd.DataFrame([mapped_result], columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
            st.dataframe(df)

    else:
        st.error("Please enter some text.")

if st.session_state.history:
    st.write("Search history:")
    for record in st.session_state.history:
        st.write(f"Text: {record['text']}, Toxicity: {record['result']}")
