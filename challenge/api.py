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


@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    global api_service
    if api_service is None:
        api_service = ApiService()
        api_service.initialize_model()
    body = await request.json()
    flights = body["flights"]
    flight = flights[0]
    response = api_service.predict(flight)
    return JSONResponse(content=response, status_code=200)