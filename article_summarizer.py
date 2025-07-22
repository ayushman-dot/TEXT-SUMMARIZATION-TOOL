# article_summarizer.py

from transformers import pipeline

def summarize_article(text, max_length=130, min_length=30):
    """
    Summarizes the input text using a transformer-based model.
    
    Args:
        text (str): The article or long text to be summarized.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
    
    Returns:
        str: A concise summary of the input text.
    """
    summarizer = pipeline("summarization")

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_text = """
    Artificial Intelligence (AI) has significantly evolved over the last decade. With the rise of deep learning,
    machine learning models have achieved human-level performance on various tasks such as image recognition, speech 
    synthesis, and language translation. AI is now being applied in healthcare, finance, education, and transportation. 
    Despite its growing popularity, there are still major concerns about ethical implications, data privacy, and the 
    potential for job displacement. Policymakers, researchers, and the global tech community are working together 
    to establish guidelines to ensure AI is used for the benefit of society.
    """

    print("\nOriginal Text (Length: {} characters):\n".format(len(input_text)))
    print(input_text)

    summary = summarize_article(input_text)
    print("\nSummarized Text (Length: {} characters):\n".format(len(summary)))
    print(summary)
