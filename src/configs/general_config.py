from pydantic import BaseModel, Field


class GeneralConfig(BaseModel):
    project_name: str = Field()
