import warnings
warnings.filterwarnings('ignore')
from crewai import Agent,Task,Crew
import os 
from utils import get_openai_api_key
from IPython.display import Markdown
open_api_key=get_openai_api_key()
os.environ["OPENAPI_MODEL_NAME"]="gpt-3.5-turbo"

# planner 
# writer
# editor

planner=Agent(
    role="Content Planner",
    goal="plan engagin and factually accurate contet on {topic}",
    backstory="""you're working on planning a blog article
            about the topic: {topic}
            You collect information that helps the
            audience learn something
            and make informed decisions.
            You work is teh basis for
            the Content Writer to write an article on this topic""",
            allow_delegation=True, #no delega en otro agente el trabajo
            verbose=True
    )

writer=Agent(
    role="Content Writer",
    goal="""Write insightful and factually accurate
        opinion piece about the topic: {topic}""",
    backstory="""You're working on a writing 
                 a new opinion piece about the topic: {topic}.
                 You base your writting on the work fo 
                 the Content Planner, who provies an aoutline
                 and relevant context about the topic. 
                 You follow the main abjetives and 
                 direction of the outline, 
                 as provide by the Content Planner. 
                 you also provide objective and impartial insights
                 and back them up with information
                 provide by the Content Planner.
                 you acknowledge in your opinion piece 
                 when your statements are opinions
                 as opposed to objetive statements.""",
    allow_delegation=False,
    verbose=True)

editor=Agent(
    role="Editor",
    goal="""Edit a given blog post to align with
        the writing style of the organization. """,
    backstory="""your are an editor who receives a blog post 
                 from the Content Writer.
                 your goal is to review the blog post
                 to ensure that it follows journalistic best practices,
                 provides balanced viewpoints
                 when providing opinions or assertions, 
                 and also avoids major controversial topics
                 or opinions when possible.""",
    allow_delegation=False,
    verbose=True)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)


crew=Crew(agents=[planner,writer,editor],
          tasks=[write,edit],
          verbose=True)

# by default operates sequentially
result=crew.kickoff(inputs={"topic": "Artificial Intelligence"})


Markdown(result)
