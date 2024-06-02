import warnings
warnings.filterwarnings('ignore')

from crewai import Agent,Task,Crew
import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool,FileReadTool	
from IPython.display import Markdown


open_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAPI_MODEL_NAME"]="gpt-3.5-turbo"

summary_agent=Agent(
    role="Summarizer ",
    goal="generar un resumen del texto",
    backstory="""Eres un estudiante de Marketing universitario
            y te han asignado la tarea de resumir un artículo
            sobre el tema: {topic}.
            Debes leer el artículo y resumirlo en un documento 
            que capture las ideas principales y los puntos clave
            del artículo. Tu resumen debe ser claro, conciso y
            fácil de entender para un público general pero sin dejar
            de usar el vocavulario adecuado.""",
            allow_delegation=True, #no delega en otro agente el trabajo
            verbose=True
    )


docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://www.elconfidencial.com/economia/2024-03-13/regionalizacion-comercio-mundial-europa-eeuu_3847511/"
)


proveedor_articulos= Task(
   description=(
       "{customer} te contacto por una tarea importante: \n "
       "{inquiry} \n \n "
       "Asegurate de usar solamente el documento de la herramienta provista"
       " "
       "You must strive to provide a complete "
       "and accurate response to the custeomr's inquiry. "
   ),
   expected_output=(
	    "Un resumen del artículo que capture las ideas principales y los puntos clave del artículo"
            "usando lenguaje específico del tema"
    ),
   tools=[docs_scrape_tool],
   agent=summary_agent
)
    
    

crew = Crew(
  agents=[summary_agent],
  tasks=[proveedor_articulos],
  verbose=2,
  memory=True
)


inputs = {
    "customer": "Jose Carlos",
    "topic":"Regionalización del comercio mundial",
    "inquiry": "Necesito un resumen del artículo que sea de al menos 800 palabras"
}


result = crew.kickoff(inputs=inputs)
Markdown(result)