#
#
# def make_predict(model_name_global, model_local, decode_dict, title, abstract):
#     model_name_global="allenai/scibert_scivocab_uncased"
#     model_local="scibert_trainer/checkpoint-2000/"
#
#     tokenizer_ = AutoTokenizer.from_pretrained(model_name_global)
#     tokens = tokenizer_(title + abstract, return_tensors="pt")
#     model_ = AutoModelForSequenceClassification.from_pretrained(model_local)
#     outs = model_(tokens.input_ids)
#
#     probs = outs["logits"].softmax(dim=-1).tolist()[0]
#     topic_probs = {}
#     for i, p in enumerate(probs):
#         if p > 0.1:
#             topic_probs[decode_dict[i]] = p
#     return topic_probs