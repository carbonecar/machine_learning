import warnings
warnings.filterwarnings('ignore')
from crewai import Agent,Task,Crew
import os 
from utils import get_openai_api_key
from IPython.display import Markdown
open_api_key=get_openai_api_key()
os.environ["OPENAPI_MODEL_NAME"]="gpt-3.5-turbo"





reclutador=Agent(
    role="Reclutador",
    goal="Reclutar candidatos para una vacante de {vacante}",
    backstory="""Estás trabajando en reclutar candidatos para una vacante de {vacante}.
            Debes realizar una serie de tareas para encontrar candidatos
            que cumplan con los requisitos de la vacante y que tengan
            el perfil adecuado para el puesto.""",
            allow_delegation=True, #no delega en otro agente el trabajo
            verbose=True
    )


publicadorEmpelo=Agent(
    role="Publicador de Empleo",
    goal="""Publicar una vacante de empleo para {vacante} en un portal de empleo""",
    backstory="""Estás trabajando en publicar una vacante de empleo en un portal de empleo.
                 Debes redactar un anuncio de trabajo que sea atractivo y que
                 atraiga a candidatos calificados para la vacante.""",
    allow_delegation=False,
    verbose=True)

editor=Agent(
    role="Editor",
    goal="""Editar un anuncio de trabajo para que sea claro y conciso""",
    backstory="""Eres un editor que recibe un anuncio de trabajo del Publicador de Empleo.
                 Tu objetivo es revisar el anuncio de trabajo para asegurarte de que sea claro,
                 conciso y esté libre de errores gramaticales o de redacción.""",
    allow_delegation=False,
    verbose=True)

