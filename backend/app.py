import streamlit as st
import time
from travel_agent import query_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

llm = ChatGroq(
    temperature=0.7, 
    model_name="llama-3.3-70b-versatile")

#  Page Configuration
st.set_page_config(
    page_title="Tripe",
    page_icon="✈️",
    layout="wide"
)

#  Sidebar UI
with st.sidebar:
    st.title("🚃 Travel Agent")
    st.markdown("घूमने जाना है? आओ, घुमा देंगे।😃")
    
    # Input field
    user_query = st.text_area(
        "Where do you want to go?", 
        placeholder="e.g., Plan a 4-day trip to Kyoto for a first-timer..."
    )
    
    # Generate Button
    generate_btn = st.button("Plan My Trip", type="primary")
    
    st.markdown("---")
    st.markdown("### ⚙️ Preferences")
    budget = st.selectbox("Budget Level", ["Budget Friendly", "Moderate", "Luxury"])
    travelers = st.slider("Number of Travelers", 1, 10, 2)

# Main Content UI
def build_enhanced_prompt(user_query, preferences):
    prompt = f"""
    You are an expert prompt engineer for a travel AI.

    User original query:
    {user_query}

    User preferences:
    - Budget: {preferences['budget']}
    - Number of travelers: {preferences['travelers']}

    Task:
    Rewrite the user query into a detailed, clear, and structured travel planning prompt.
    Expand it into 4–6 lines.
    Include budget sensitivity and group size considerations.
    Do NOT give the travel plan.
    ONLY return the enhanced prompt text.
    """

    enhanced_prompt = llm.invoke(prompt)
    return enhanced_prompt.content.strip()


if generate_btn and user_query:
    with st.spinner("Drafting your perfect itinerary..."):

        user_preferences = {
            "budget": budget,
            "travelers": travelers
        }
        enhanced_query = build_enhanced_prompt(user_query, user_preferences)
        result = query_agent(enhanced_query)

        st.markdown(result , width = "stretch", unsafe_allow_html=True)

        st.success("Itinerary generated successfully!")

elif not user_query:

    st.markdown(
        """
        <div style="text-align:center; padding: 30px 10px;">
            <h1>🌍 Tripe </h1>
            <h3><b>Plan smarter. Travel better. Explore more✨.</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="font-size:18px; line-height:1.8;">
        <b>🚀 I can help with you this:</b><br><br>

        🗺️ <b>Day-wise Travel Itinerary</b><br>
        💰 <b>Smart Budget Estimation</b><br>
        🍜 <b>Food & Local Cuisine Recommendations</b><br>
        🚌 <b>Travel Routes, Tips & Hacks</b><br>
        🌦️ <b>Best Time & Season to Visit</b><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("👉 Start by entering your travel destination in the sidebar.")