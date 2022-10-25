import streamlit as st
from pandas import DataFrame
import seaborn as sns
from model import ArxivClassifierModelsPipeline

st.markdown("# Hello, friend!")
st.markdown(" This magic application going to help you with understanding of science paper topic! Cool? Yeah! ")

model = ArxivClassifierModelsPipeline()

with st.form(key="my_form"):
    st.markdown("### 🎈 Do you want a little magic?  ")
    st.markdown(" Write your article title and abstract to textboxes bellow and I'll gues topic of your paper!  ")
    ce, c2, c3 = st.columns([0.07, 5, 0.07])

    with c2:
        doc_title = st.text_area(
            "Paste your paper's title below (max 100 words)",
            height=210,
        )

        doc_abstract = st.text_area(
            "Paste your paper's abstract text below (max 100500 words)",
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
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc_abstract = doc_abstract[:MAX_WORDS_ABSTRACT]

        submit_button = st.form_submit_button(label="✨ Let's play, try it!")

if not submit_button:
    st.stop()


title = doc_title
abstract = doc_abstract
# try:
#     tokens = tokenizer_(title + abstract, return_tensors="pt")
# except ValueError:
#     st.error("Word parsing into tokens went wrong! Is input valid? If yes, pls contact author alekseystepin13@gmail.com")

preds_topic, preds_maintopic = model.make_predict(title + abstract)

st.markdown("## 🎈 Yor article probably about:  ")
st.header("")

df = (
    DataFrame(preds_topic.items(), columns=["Topic", "Probability"])
        .sort_values(by="Probability", ascending=False)
        .reset_index(drop=True)
)
df.index += 1


# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Probability",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Probability": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.markdown("#### We suppose your research about:  ")
    st.markdown(f"### {preds_maintopic}! ")
    st.markdown(f"Wow, we're impressed, are you addicted to {preds_maintopic.lower()}?! Coool! ")
    st.markdown("##### More detailed, it's about topic:  ")
    st.table(df)
