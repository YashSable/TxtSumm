import streamlit as st
from transformers import pipeline, AutoModel, AutoTokenizer
from summarizer import Summarizer
from rouge_score import rouge_scorer
import sacrebleu

st.set_page_config(layout="wide")

@st.cache_data
def bert_summary(text):
    bert_model = Summarizer(model='bert-base-uncased')
    summary = bert_model(text)
    return summary

@st.cache_data
def pegasus_summary(text):
    summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail", tokenizer="google/pegasus-cnn_dailymail")
    summary = summarizer(text)[0]['summary_text']
    return summary

@st.cache_data
def t5_summary(text):
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def compute_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

def compute_bleu(reference, summary):
    # Tokenize reference and summary
    reference_tokens = [reference.split()]
    summary_tokens = summary.split()

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(summary_tokens, reference_tokens)
    return bleu.score

def display_evaluation_scores(reference, summary):
    rouge_scores = compute_rouge(reference, summary)
    bleu_score = compute_bleu(reference, summary)
    
    st.write("ROUGE-1 F1 Score:", rouge_scores['rouge1'].fmeasure)
    st.write("ROUGE-L F1 Score:", rouge_scores['rougeL'].fmeasure)
    st.write("BLEU Score:", bleu_score)

choice = st.sidebar.selectbox("Select your model", ["BERT", "PEGASUS", "T5"])

if choice == "BERT":
    st.subheader("Summarizing Text Using BERT")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = bert_summary(input_text)
                st.success(result)
                display_evaluation_scores(input_text, result)

elif choice == "PEGASUS":
    st.subheader("Summarizing Text Using PEGASUS")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = pegasus_summary(input_text)
                st.success(result)
            display_evaluation_scores(input_text, result)

elif choice == "T5":
    st.subheader("Summarizing Text Using T5")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = t5_summary(input_text)
                st.success(result)
            display_evaluation_scores(input_text, result)
