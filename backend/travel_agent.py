import os
from dotenv import load_dotenv
import re
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") 

action_re = re.compile(r'^Action:\s*(\w+):\s*(.*)$')

class TravelAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        # load Model
        self.llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY") )
        self.messages = [{"role": "system", "content": system_prompt}]

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.llm.invoke(self.messages)
        self.messages.append({"role": "assistant", "content": result.content})
        return result.content

# Search Tool
search = GoogleSerperAPIWrapper(serpapi_api_key=os.getenv("SERPER_API_KEY"))

@tool
def search_destination(query) :
    """Find tourist spots, food, and TripAdvisor reviews for a place."""
    if "review" in query.lower() or "rating" in query.lower():
        refined_query = f"{query} site:tripadvisor.com"
    else:
        refined_query = query
        
    return search.run(refined_query)

@tool
def weather_tool(query):
    """Get current weather info."""
    return search.run(f"current weather in {query}")

known_actions = {
    "search_destination": search_destination,
    "weather_tool": weather_tool
}

# Query
prompt = """
You are a Professional Travel Expert with experience in itinerary design, local insights, and travel planning.
You work in a loop of Thought, Action, PAUSE, Observation.

PROCESS:
1. **Thought**: Plan what needs to be searched.
2. **Action**: Use 'search_destination' ONLY if you don't have enough information.
3. **PAUSE**: Always return PAUSE after an Action.
4. **Observation**: Review the search results.
5. **Answer**: Once you have enough details, stop searching and provide the final itinerary.

Do NOT show Thought, Action, Observation.
Give ONLY final answer in clean markdown format with:
- Day-wise itinerary (Day 1, Day 2, etc.)
- Morning / Afternoon / Evening activity flow
- Bullet points for clarity
- Practical travel tips at the end

**ITINERARY STYLE GUIDELINES**:
- Balance sightseeing, rest, and travel time
- Mention famous landmarks with brief descriptions
- Suggest ideal areas to stay (not exact bookings)
- Include food experiences (local cuisine or famous eateries)
- Keep plans realistic for first-time travelers
- Avoid overcrowding each day

**
Use case	            Emoji
Tourist spot / place	📍
City / Area	            🏙️
Beach	                🏖️
Mountain / Hill     	⛰️
Temple / Monument      	🏛️
Nature / Park	        🌿

**CRITICAL RULES**:
- Do NOT repeat the same search query more than once.
- If the 'Observation' contains enough places/details, proceed to the Final Answer immediately.
- Your final response MUST start with the word 'Answer:' followed by the itinerary.
- Keep the tone friendly, helpful, and professional.

Query:
    {query}

    STRICT RULES:
    - Return ONLY valid JSON
    - Do not include explanations
    - Use this exact schema:

    {{
      "duration": "string",
      "budget": "string",
      "season": "string",
      "itinerary": "string"
    }}


Answer only regarding tour and travel. If the query is unrelated, say 'Please ask regarding tour and travel only.'
""".strip()
    
def query_agent(question,max_turns=5):
    i = 0
    bot = TravelAgent(prompt)
    next_step = question

    previous_actions = set()

    while i < max_turns:
        i += 1
        result = bot(next_step)
        
        # Check for Action
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        
        if actions:
            action, action_input = actions[0].groups()
            print(f"--- Running {action}: {action_input} ---")
            observation = known_actions[action].invoke(action_input)
            next_step = f"Observation: {observation}"
        else:
            return result

        return result