def summarize_article(article_text, tokenizer, model):
    inputs = tokenizer.encode(
        "summarize: " + article_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
