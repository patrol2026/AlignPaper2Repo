
import sys
from os.path import dirname, abspath, join
sys.path.append(dirname(abspath(__file__)))
from reproduction_utils.os_utils import load_text_file
from info_loader import InfoLoader
from bs4 import Tag
from reproduction_utils.nlp_utils import get_word_similarity
from reproduction_utils.layout_utils import get_textual_representation, get_prompt_desc_for_view, is_layout_view
from utils.config import Config
from utils.cmd_args import CmdArgs
from reproduction_utils.llm_helper import language_query
from reproduction_utils.logger_utils import get_logger
import json

class E:
    """
    This represents a UI element on the screen.
    """
    
    def __init__(self, desc:str=None, color:str=None, checked:bool=None, location:str = None) -> None:
        """
        Initialize the Element object

        :param desc: the description of the UI element's icon or its label, or the input value for an input field; 
        :param color: the described color of the UI element;
        :param checked: if a checkbox/radio button is checked or not;
        :param location: the described location of the element on the screen;
        """
        self.desc = desc
        self.color = color
        self.checked = checked
        self.location = location

    def populate_info(self, view: Tag):
        self.widget_view = view 
        
    def __str__(self) -> str:
        return f"<UI element, {self.desc}>"

class S:
    """
    This represents a UI state of the mobile app at a time point.
    """
    
    def __init__(self, is_crash:bool=False, keyboard:str = None) -> None:
       """
       Initialize the Screen object 

       :param is_crash: if the app has crashed.
       """
       self.is_crash = is_crash
       self.keyboard = keyboard
       self.id = "Unknown"
    
    def populate_info(self, info_loader:InfoLoader, retrieve_prev_info=False):
        self.layout = info_loader.get_layout(retrieve_prev_info)
        self.id = self.layout.id
        self.app_pkg = info_loader.app_pkg


    def validate(self) -> bool:
        """
        Populate the information of the screen and validate whether the screen satisfy the description (e.g., if the screen displays a crash)
        """
        result = True
        # *** Crash Dialog Recognizer ***
        if self.is_crash:
            crash_widget = E(desc="app has stopped")
            result = result and in_screen(crash_widget, self)
        
        if self.keyboard is not None:
            result = result and (self.keyboard == self.keyboard_status())

        return result

    def keyboard_status(self)->bool:
        return "on" if self.layout.keyboard_on() else "off"

    def __str__(self) -> str:
        return f"<UI State {self.id}>"
    
    def __eq__(self, value: object) -> bool:
        return self.layout.get_layout_hash(self.app_pkg) == value.layout.get_layout_hash(self.app_pkg)

class D:
    """
    This represents the state of the mobile device at one time point.
    """

    def __init__(self, log: str = None, volumn: int = None, audio: str = None) -> None:
        self.log = log
        self.audio = audio

    def populate_info(self, info_loader:InfoLoader, retrieve_prev_info):
        self.device_info = info_loader.get_device_info(retrieve_prev_info)  
        self.volume = self.device_info.volume
        self.id = self.device_info.id

    def validate(self) -> bool:
        """
        Populate the information of the device and validate whether the device status satisfy the description (e.g., if the device generated a stacktrace)
        """
        # *** Device Log Recognizer *** 
        check_result = [True]
        if self.log is not None:
            check_result.append(self.log in self.device_info.log)
        if self.audio is not None:
            check_result.append(self.audio == self.device_info.audio)
        
        return all(check_result)
    
    def __str__(self) -> str:
        return f"<Device State, {self.id}>"


def load_prompts(example_ids):
    prompt_text = ""

    for i in example_ids:
        prompt_text += "\n\n"
        prompt_text += load_text_file(join(dirname(abspath(__file__)), "prompts", i+".txt"))
     
    return prompt_text

# ** widget recognizer **
def in_screen(element:E, screen:S)->bool:
    """
    Search through all widgets in the specified screen to see if there is a match. If yes, populate the info into the element. Otherwise, return False.
    """
    logger = get_logger("in-screen-function", CmdArgs.logger_level)
    logger.debug(f"Checking whether the {str(element)} is in {str(screen)}")
    all_views = screen.layout.iterate_views(app_only=True)

    if element.desc is not None:
        if Config.use_llm_for_widget_recognition:
            logger.debug("Matching the widget using LLM")
            # get a list of widget description
            widget_descs = "\n".join(
                [
                  f"{i}. {get_prompt_desc_for_view(widget)}"  for i, widget in enumerate(all_views)
                  if not is_layout_view(widget)
                ]
            )
            
            # construct the prompt for the widget matching
            system_msg = load_prompts(
                [
                    'system',
                    'consider_type',
                    'label_desc',
                    'label_exact_match',
                ]
            )
            usr_msg = f"Target widget description: {element.desc}\n"
            if len(widget_descs)>0:
                usr_msg+=f"Widgets on the UI: \n{widget_descs}"
            else:
                usr_msg+=f"No widgets on the UI."
            
            # query llm and parse results
            logger.debug("Prompt: \n"+usr_msg)
            response, model_info = language_query(usr_msg, system_msg, Config.llm_model, Config.llm_seed, Config.llm_temperature)
            logger.debug(response)
            response = json.loads(response)
            
            if response['id'] is not None:
                if int(response['confidence']) <= 5:
                    all_views = []
                    logger.debug(f'Confidence too low. Ignore the chose widget.')
                else:
                    all_views = [all_views[int(response['id'])]]
                    logger.debug(f"LLM matcher found a matched widget: {get_textual_representation(all_views[0])}")
            else:
                logger.debug(f"LLM matcher didn't find a matched widget.")
                all_views = []
            logger.debug(f"Reason: {response['reason']}")
            logger.debug(f"Confidence level: {response['confidence']}")
        else:
            # filter view by checking status
            most_similar_view = None
            most_similar_sim = 0
            for widget in all_views:
                text_representations = get_textual_representation(widget)
                if element.desc == "app has stopped": # for crash widget, directly check if keyword "has stopped" is in the screen
                    similarity = max([0] + [1 if "has stopped" in x else 0 for x in text_representations])
                else:
                    similarity = max([0] + [get_word_similarity(element.desc, x) for x in text_representations])
                if similarity >= Config.text_sim_thred and similarity > most_similar_sim:
                    most_similar_view = widget
                    most_similar_sim = similarity
            all_views.clear()
            if most_similar_view is not None:
                all_views = [most_similar_view]


    if element.color is not None:
        # filter view by color
        pass        

    if element.checked is not None and len(all_views)>0:
        # filter view by checking status
        all_views = list(filter(lambda x: (x.attrib['checked']=='true') == element.checked, all_views))
        if len(all_views)==0:
            logger.debug("Status is not satisfied.")

    if element.location is not None:
        # filter view by location
        pass
    
    if len(all_views) == 0:
        return False
    else:
        element.populate_info(all_views[0])
    return True