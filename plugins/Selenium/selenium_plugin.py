from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import os
from typing import Annotated
from enum import Enum
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from pydantic import BaseModel
import time
import logging


class WebActionType(Enum):
    CLICK = "click"
    TYPE_TEXT = "type_text"
    TYPE_ENTER = "type_enter"
    WAIT = "wait"
    NONE = "none"

class WebAction(BaseModel):
    action: WebActionType
    target: str
    content: str
    termination_message: str


class SeleniumPlugin():

    def __init__(self, ai_endpoint) -> None:
        self.logger = logging.getLogger('plugins.Selenium.selenium_plugin')

        # Configure webdriver
        options = webdriver.EdgeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-logging']) # Suppress console logs
        self.driver = webdriver.ChromiumEdge(
            options=options
        )
        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )
        self.ai_client = AzureOpenAI(
            azure_endpoint=ai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-08-01-preview"
        )


    @kernel_function(
        description="Open a web page to perform an action.",
        name="OpenWebPage"
    )
    def open_web_page(self, url: Annotated[str, "the url of the web page to open"]) -> str:
        self.logger.info(f"Navigating to {url}")
        self.driver.get(url)
        return "Web page opened. You may now perform actions."


    @kernel_function(
        description="Perform an action on the current web page.",
        name="PerformWebAction"
    )
    def perform_web_action(self, action_description: Annotated[str, "The action to achieve on the web page."]) -> str:
        done = False
        max_attempts = 15
        current_attempt = 0
        while not done:
            current_attempt += 1
            if current_attempt > max_attempts:
                return "Max attempts reached. Could not complete the action."
            time.sleep(0.2)
            action = self.get_web_action(action_description)
            self.logger.info(f"Performing action: {action}")
            message = self.execute_web_action(action)
            if action.action == WebActionType.NONE:
                done = True
                message = action.termination_message
        return message
    

    def get_web_action(self, objective: Annotated[str, "the objective to achieve on the web page"]) -> WebAction:
        # Get screenshot of the web page
        screenshot_b64 = self.capture_current_page()
        image_url = f"data:image/png;base64,{screenshot_b64}"

        # Use AI to determine the next action
        response = self.ai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                { "role": "system",
                    "content":
                            """Determine the next action to perform on the pictured web page for achieving the objective.
                            If there is a cookie banner obstructing the web page (and only if), close this first before continuing with the objective.
                            If the objective has been achieved, the action may be none. In that case, provide a termination message that includes any information
                            that needs to be retrieved from the web page. If the website appears to be loading, use the wait action. In addition to the action,
                            provide the target element. For type actions, include the content to type."""
                            },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": f"Objective: {objective}" 
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ] } 
            ],
            response_format=WebAction
        )
        result = response.choices[0].message.parsed
        return result
    

    def capture_current_page(self) -> str:
        body_element = self.driver.find_element(By.TAG_NAME, "body")
        screenshot_b64 = body_element.screenshot_as_base64
        return screenshot_b64


    def execute_web_action(self, action: WebAction) -> str:
        if action.action == WebActionType.CLICK:
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            anchors = self.driver.find_elements(By.TAG_NAME, "a")
            elements = buttons + anchors
            element = self.match_clickable(elements, action.target)
            element.click()
            return f"Clicked the element: {element}"
        elif action.action == WebActionType.TYPE_TEXT:
            elements = self.driver.find_elements(By.TAG_NAME, "input")
            element = self.match_input(elements, action.target)
            element.send_keys(action.content)
            return f"Typed '{action.content}' in the input field: {element}"
        elif action.action == WebActionType.TYPE_ENTER:
            elements = self.driver.find_elements(By.TAG_NAME, "input")
            element = self.match_input(elements, action.target)
            element.send_keys(Keys.RETURN)
            return f"Pressed Enter in the input field: {element}"
        elif action.action == WebActionType.WAIT:
            time.sleep(3)
            return "Waiting..."
        elif action.action == WebActionType.NONE:
            return action.termination_message
        else:
            raise ValueError(f"Unsupported action type: {action.action}")
        

    def match_clickable(self, elements: list[WebElement], target: Annotated[str, "the element to find"]) -> WebElement:
        elements_dicts = [{"idx": i, "text": element.text} for i, element in enumerate(elements) if element.is_displayed()]
        idx = self.match_element_idx(elements_dicts, target)
        return elements[idx]
    

    def match_input(self, elements: list[WebElement], target: Annotated[str, "the element to find"]) -> WebElement:
        elements_dicts = [{"idx": i, "name": element.get_attribute("name"), "placeholder": element.get_attribute("placeholder")} for i, element in enumerate(elements) if element.is_displayed()]
        idx = self.match_element_idx(elements_dicts, target)
        return elements[idx]
    

    def match_element_idx(self, elements: list[dict], target: Annotated[str, "the element to find"]) -> WebElement:
        class ResponseFormat(BaseModel): selected_element_idx: int
        response = self.ai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Select the most relevant element from the list that best matches the following description: {target}"},
                {"role": "user", "content": f"Elements: {elements}"}
            ],
            response_format=ResponseFormat
        )
        idx = response.choices[0].message.parsed.selected_element_idx
        return idx