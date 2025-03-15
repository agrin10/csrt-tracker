from pydantic import BaseModel, ConfigDict


class CoordinatesXYAH(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    center_x: float
    center_y: float
    aspect_ratio: float
    height: float


class CoordinatesXYXY(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start_x: float
    start_y: float
    end_x: float
    end_y: float
