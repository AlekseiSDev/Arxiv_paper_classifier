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



class ArxivClassifierModelsPipeline():

    def __init__(self):
        self.model_topic_clf = self.__load_topic_clf()
        self.model_maintopic_clf = self.__load_maintopic_clf()

        topic_clf_default_model = "allenai/scibert_scivocab_uncased"
        self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_clf_default_model)

        maintopic_clf_default_model = "Wi/arxiv-topics-distilbert-base-cased"
        self.maintopic_tokenizer = AutoTokenizer.from_pretrained(maintopic_clf_default_model)

        with open('models/scibert/decode_dict_topic.pkl', 'rb') as f:
            self.decode_dict_topic = pickle.load(f)

        with open('models/maintopic_clf/decode_dict_maintopic.pkl', 'rb') as f:
            self.decode_dict_maintopic = pickle.load(f)

    def make_predict(self, text):
        tokens_topic = self.topic_tokenizer(text, return_tensors="pt")
        topic_outs = self.model_topic_clf(tokens_topic.input_ids)
        probs_topic = topic_outs["logits"].softmax(dim=-1).tolist()[0]
        topic_probs = {}
        for i, p in enumerate(probs_topic):
            if p > 0.1:
                topic_probs[self.decode_dict_topic[i]] = p

        tokens_maintopic = self.maintopic_tokenizer(text, return_tensors="pt")
        maintopic_outs = self.model_maintopic_clf(tokens_maintopic.input_ids)
        probs_maintopic = maintopic_outs["logits"].softmax(dim=-1).tolist()[0]
        maintopic_probs = {}
        for i, p in enumerate(probs_maintopic):
            if p > 0.1:
                maintopic_probs[self.decode_dict_maintopic[i]] = p
        


        return topic_probs, maintopic_probs

    @st.cache(suppress_st_warning=True)
    def __load_topic_clf(self):
        st.write("Loading model")
        return AutoModelForSequenceClassification.from_pretrained("models/scibert/")

    @st.cache(suppress_st_warning=True)
    def __load_maintopic_clf(self):
        st.write("Loading second model")
        return AutoModelForSequenceClassification.from_pretrained("models/maintopic_clf/")