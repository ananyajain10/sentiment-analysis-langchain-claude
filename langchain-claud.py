import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os
key_api = os.environ.get('ANTHROPIC_API_KEY')

def get_sentiment_analysis(text):
    chat = ChatAnthropic(temperature=0, api_key=key_api , model_name="claude-3-opus-20240229")

    system = (
        "You are an advanced natural language processing system tasked with analyzing a user's input text {user_input} about how their day went. Your goal is to provide the following information in the {model_output} format: Sentiment analysis: Determine the overall sentiment of the text, classifying it as positive, negative, or neutral. Provide a sentiment score ranging from -1 (highly negative) to 1 (highly positive). Emotion detection: Identify the specific emotions expressed in the text, such as happiness, sadness, anger, fear, surprise, etc. For each emotion detected, provide a score or intensity level ranging from 0 (not present) to 1 (highly present). Emotion fluctuation: Analyze how the emotions fluctuate throughout the text by breaking it down into smaller segments (e.g., sentences or paragraphs). Provide a list or array of emotion scores for each segment, allowing visualization of the emotional journey. Mindset analysis: Based on the sentiment analysis and emotion detection results, determine the user's overall mindset or emotional state. This could be a single label (e.g., positive, negative, neutral, mixed) or a more detailed description. Recommendations: Provide personalized recommendations or tips to help the user improve their mindset or emotional well-being, based on the analysis results."
    )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{text}")])
    chain = prompt | chat

    result = chain.invoke({
        "user_input": text,
        "model_output": """
{
    "sentiment_analysis": {
        "score": 0.9,
        "label": "positive"
    },
    "emotion_detection": {
        "joy": 0.8,
        "contentment": 0.7,
        "excitement": 0.9,
        "relaxation": 0.8
    },
    "emotion_fluctuation": [
        {"emotion": "Joy", "score": 0.8},
        {"emotion": "Excitement", "score": 0.9},
        {"emotion": "Contentment", "score": 0.7},
        {"emotion": "Relaxation", "score": 0.8}
    ],
    "mindset_analysis": "Your mindset seems to be overwhelmingly positive and optimistic. You're experiencing high levels of joy, excitement, and contentment, likely due to spending quality time with friends and engaging in relaxing activities like gardening.",
    "recommendations": "Keep nurturing activities that bring you joy and relaxation, like spending time with friends and gardening. It's essential to prioritize activities that uplift your mood and contribute to your overall well-being."
}
""",
        "text": text
    })

    return result

def main():
    st.title("Sentiment Analysis App")
    text = st.text_area("Enter your text here:")

    if st.button("Analyze"):
        result = get_sentiment_analysis(text)
        st.write(result)

if __name__ == "__main__":
    main()