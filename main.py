from fastapi import FastAPI
import datetime
import getimage
from pydantic import BaseModel
# Create an instance of the FastAPI class
app = FastAPI()

# Define a route and its corresponding function


class Item(BaseModel):
    start: str
    end: str
    coordinates: str


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2019, 12, 31)
betsiboka_coords_wgs84 = (-7.612938, 33.155029, -7.255824, 34.005625)


@app.get("/getimages")
def read_root():
    return {"data": getimage.getImagesByCoordinatesAndImages(betsiboka_coords_wgs84, start, end)[0].tolist()}


@app.post("/getimages/")
def read_root(item: Item):
    start = item.start.split(",")
    end = item.end.split(",")
    coordinates = item.coordinates.split(",")
    start = datetime.datetime(int(start[0]), int(start[1]), int(start[2]))
    end = datetime.datetime(int(end[0]), int(end[1]), int(end[2]))
    coordinates = (float(coordinates[0]), float(coordinates[1]),
                   float(coordinates[2]), float(coordinates[3]))
    return {"data": getimage.getImagesByCoordinatesAndImages(coordinates, start, end)[0].tolist()}


# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
