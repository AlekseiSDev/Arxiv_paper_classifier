import streamlit as st
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ArxivClassifierModel():

    def __init__(self):
        self.model = self.__load_model()

        model_name_global = "allenai/scibert_scivocab_uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_global)
        with open('./models/scibert/decode_dict.pkl', 'rb') as f:
            self.decode_dict = pickle.load(f)

    def make_predict(self, text):
        # tokenizer_ = AutoTokenizer.from_pretrained(model_name_global)
        tokens = self.tokenizer(text, return_tensors="pt")

        outs = self.model(tokens.input_ids)

        probs = outs["logits"].softmax(dim=-1).tolist()[0]
        topic_probs = {}
        for i, p in enumerate(probs):
            if p > 0.1:
                topic_probs[self.decode_dict[i]] = p
        return topic_probs

    #  allow_output_mutation=True
    @st.cache(suppress_st_warning=True)
    def __load_model(self):
        st.write("Loading big model")
        return AutoModelForSequenceClassification.from_pretrained("models/scibert/")