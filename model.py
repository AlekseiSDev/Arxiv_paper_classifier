import streamlit as st
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

        with open('models/maintopic_clf/main_topic_dict.pkl', 'rb') as f:
            self.main_topic_dict = pickle.load(f)

        with open('models/scibert/topic_dict.pkl', 'rb') as f:
            self.topic_dict = pickle.load(f)

    def make_predict(self, text):
        tokens_topic = self.topic_tokenizer(text, return_tensors="pt")
        topic_outs = self.model_topic_clf(tokens_topic.input_ids)
        probs_topic = topic_outs["logits"].softmax(dim=-1).tolist()[0]
        topic_probs = {}
        for i, p in enumerate(probs_topic):
            if p > 0.1:
                if self.decode_dict_topic[i] in self.topic_dict:
                    topic_probs[self.topic_dict[self.decode_dict_topic[i]]] = p
                else:
                    topic_probs[self.decode_dict_topic[i]] = p

        tokens_maintopic = self.maintopic_tokenizer(text, return_tensors="pt")
        maintopic_outs = self.model_maintopic_clf(tokens_maintopic.input_ids)
        probs_maintopic = maintopic_outs["logits"].softmax(dim=-1).tolist()[0]
        maintopic_probs = self.decode_dict_maintopic[0]
        
        return topic_probs, self.main_topic_dict[maintopic_probs]

    @st.cache(suppress_st_warning=True)
    def __load_topic_clf(self):
        st.write("Loading model")
        return AutoModelForSequenceClassification.from_pretrained("models/scibert/")

    @st.cache(suppress_st_warning=True)
    def __load_maintopic_clf(self):
        st.write("Loading second model")
        return AutoModelForSequenceClassification.from_pretrained("models/maintopic_clf/")