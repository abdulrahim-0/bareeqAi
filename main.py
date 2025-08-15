from unittest.mock import Base
from fastapi import FastAPI
from agent import Agent
from pydantic import BaseModel
import logging
import uvicorn
import os

logger = logging.getLogger(__name__)
logger.info("API is starting up")
logger.info(uvicorn.Config.asgi_version)

app = FastAPI()
agent = Agent()
documents = [
    {"question": "How often should I work out per week?", "answer": "3 to 5 times per week is optimal for general fitness."},
    {"question": "What is the best way to build muscle?", "answer": "Focus on compound exercises like squats, deadlifts, bench press, and progressive overload."},
    {"question": "Should I do cardio before or after weight training?", "answer": "It depends on your goals, but usually after weight training to preserve strength."},
    {"question": "How many reps should I do for muscle growth?", "answer": "Generally 8-12 reps per set for hypertrophy."},
    {"question": "Is protein powder necessary?", "answer": "Not necessary if you get enough protein from food, but it can help reach your protein target."},
    {"question": "How much water should I drink daily?", "answer": "About 2 to 3 liters per day, depending on activity level."},
    {"question": "Can I lose fat and gain muscle at the same time?", "answer": "Yes, especially for beginners or those returning after a break, with proper nutrition and training."},
    {"question": "What is a good warm-up routine?", "answer": "5-10 minutes of light cardio plus dynamic stretches targeting the muscles you’ll train."},
    {"question": "Should I take rest days?", "answer": "Yes, rest days are crucial for recovery and muscle growth."},
    {"question": "How important is sleep for fitness?", "answer": "Very important; aim for 7-9 hours per night for recovery and performance."},
    {"question": "Is it better to train in the morning or evening?", "answer": "It depends on your schedule and energy levels; consistency matters most."},
    {"question": "Can I build abs without cardio?", "answer": "You can strengthen abs, but reducing fat through cardio and diet helps them show."},
    {"question": "How long should each workout last?", "answer": "Typically 45-75 minutes, depending on intensity and volume."},
    {"question": "Should I stretch after workouts?", "answer": "Yes, static stretching after workouts helps flexibility and recovery."},
    {"question": "What are compound exercises?", "answer": "Exercises that work multiple muscle groups at once, like squats or bench press."},
    {"question": "Is lifting heavy weights dangerous?", "answer": "Not if done with proper form and progressive loading."},
    {"question": "How many sets should I do per exercise?", "answer": "3-5 sets per exercise is common for strength and hypertrophy."},
    {"question": "Can I do strength and cardio on the same day?", "answer": "Yes, but consider the order based on your goals."},
    {"question": "Should I eat before or after a workout?", "answer": "Eating a small meal with carbs and protein before is fine; post-workout protein helps recovery."},
    {"question": "Is bodyweight training effective?", "answer": "Yes, especially for beginners and for building functional strength."},
    {"question": "Can I gain muscle on a vegan diet?", "answer": "Yes, with proper protein intake from plant-based sources."},
    {"question": "How important is core strength?", "answer": "Very; it stabilizes your body and improves performance in most exercises."},
    {"question": "Should I track calories to lose weight?", "answer": "Yes, tracking helps create a calorie deficit for fat loss."},
    {"question": "Are supplements necessary?", "answer": "Not necessary for most people, but some like protein or creatine can help."},
    {"question": "How fast can I expect results?", "answer": "Beginners often see changes in 4-6 weeks; consistency is key."},
    {"question": "What is progressive overload?", "answer": "Gradually increasing weight, reps, or intensity over time to build strength/muscle."},
    {"question": "How do I prevent injuries?", "answer": "Proper form, warming up, rest, and not overloading too fast."},
    {"question": "Is HIIT better than steady-state cardio?", "answer": "Both have benefits; HIIT is time-efficient and improves cardiovascular fitness quickly."},
    {"question": "Should I lift weights every day?", "answer": "No, muscles need recovery; usually 3-5 times per week is ideal."},
    {"question": "How do I improve flexibility?", "answer": "Regular stretching, yoga, and mobility exercises help increase flexibility over time."}
]

agent.add_documents(documents)

class ChatInput(BaseModel):
    messages: list[str]
    thread_id: str

class QueryInput(BaseModel):
    query: str
    thread_id: str

@app.post("/chat")
async def chat(input: ChatInput):
    config = {"configurable": {"thread_id": input.thread_id}}
    logger.info(input.messages)
    return {"empty":""}
    #response = await agent.retrieve_documents({"messages": input.messages}, config=config)
    #return response["messages"][-1].content

@app.post("/search")
async def search(input: QueryInput):
    config = {"configurable": {"thread_id": input.thread_id}}
    logger.info(input.query)
    documents = agent.get_answer(input.query)
    return {"query": input.query, "documents": documents['answer']}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)), 
        reload=True
    )


