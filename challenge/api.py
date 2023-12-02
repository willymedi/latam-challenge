import fastapi
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from challenge.service import ApiService

api_service = None

app = fastapi.FastAPI()

def startup_event():
    global api_service
    api_service = ApiService()
    api_service.initialize_model()
    print("Ejecutando código durante el evento de inicio")


app.add_event_handler("startup", startup_event)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }



async def get_api_service():
    global api_service
    if api_service is None:
        api_service = ApiService()
        await api_service.initialize_model()
    return api_service

@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    service = await get_api_service()
    body = await request.json()
    flights = body["flights"]
    flight = flights[0]
    response = service.predict(flight)
    return {"prediction": response}