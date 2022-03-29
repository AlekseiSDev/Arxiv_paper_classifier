import streamlit as st
import transformers
import pickle
import seaborn as sns
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.markdown("# Hello, friend!")
st.markdown(" This magic application going to help you with understanding of science paper topic! Cool? Yeah! ")
# st.markdown("<img width=200px src='https://rozetked.me/images/uploads/dwoilp3BVjlE.jpg'>", unsafe_allow_html=True)

st.write("Loading tokenizer and dict")
model_name_global = "allenai/scibert_scivocab_uncased"
tokenizer_ = AutoTokenizer.from_pretrained(model_name_global)
with open('./models/scibert/decode_dict.pkl', 'rb') as f:
    decode_dict = pickle.load(f)

with st.form(key="my_form"):
    st.markdown("### 🎈 Do you want a little magic?  ")
    st.markdown(" Write your article title and abstract to textboxes bellow and I'll gues topic of your paper!  ")
    # ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    ce, c2, c3 = st.columns([0.07, 5, 0.07])
    # with c1:
    #     ModelType = st.radio(
    #         "Choose your model",
    #         ["DistilBERT (Default)", "Flair"],
    #         help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
    #     )
    #
    #     if ModelType == "Default (DistilBERT)":
    #         # kw_model = KeyBERT(model=roberta)
    #
    #         @st.cache(allow_output_mutation=True)
    #         def load_model():
    #             return KeyBERT(model=roberta)
    #
    #
    #         kw_model = load_model()
    #
    #     else:
    #         @st.cache(allow_output_mutation=True)
    #         def load_model():
    #             return KeyBERT("distilbert-base-nli-mean-tokens")
    #
    #
    #         kw_model = load_model()

    with c2:
        doc_title = st.text_area(
            "Paste your abstract title below (max 100 words)",
            height=210,
        )

        doc_abstract = st.text_area(
            "Paste your abstract text below (max 100500 words)",
            height=410,
        )

        MAX_WORDS_TITLE, MAX_WORDS_ABSTRACT = 50, 500
        import re

        len_title = len(re.findall(r"\w+", doc_title))
        len_abstract = len(re.findall(r"\w+", doc_abstract))
        if len_title > MAX_WORDS_TITLE:
            st.warning(
                "⚠️ Your title contains "
                + str(len_title)
                + " words."
                + " Only the first 50 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc_title = doc_title[:MAX_WORDS_TITLE]

        if len_abstract > MAX_WORDS_ABSTRACT:
            st.warning(
                "⚠️ Your abstract contains "
                + str(len_abstract)
                + " words."
                + " Only the first 50 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc_abstract = doc_abstract[:MAX_WORDS_ABSTRACT]

        submit_button = st.form_submit_button(label="✨ Let's play, try it!")

if not submit_button:
    st.stop()


#  allow_output_mutation=True
@st.cache(suppress_st_warning=True)
def load_model():
    st.write("Loading big model")
    return AutoModelForSequenceClassification.from_pretrained("models/scibert/")


def make_predict(tokens, decode_dict):
    # tokenizer_ = AutoTokenizer.from_pretrained(model_name_global)
    # tokens = tokenizer_(title + abstract, return_tensors="pt")

    model_ = load_model()
    outs = model_(tokens.input_ids)

    probs = outs["logits"].softmax(dim=-1).tolist()[0]
    topic_probs = {}
    for i, p in enumerate(probs):
        if p > 0.1:
            topic_probs[decode_dict[i]] = p
    return topic_probs


model_local = "models/scibert/"

title = doc_title
abstract = doc_abstract
tokens = tokenizer_(title + abstract, return_tensors="pt")

predicts = make_predict(model_name_global, model_local, tokens, decode_dict, title, abstract)

st.markdown("## 🎈 Yor article probably about:  ")
st.header("")

df = (
    DataFrame(predicts.items(), columns=["Topic", "Prob"])
        .sort_values(by="Prob", ascending=False)
        .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Prob",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Prob": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
