from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from dotenv import load_dotenv
import os
import asyncio
import logging
from colorama import Fore, Style
from plugins.Web.web_plugin import WebPlugin
from plugins.Selenium.selenium_plugin import SeleniumPlugin

async def main():
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger('plugins.Selenium.selenium_plugin').setLevel(logging.INFO)
    logging.debug("Starting application")

    # Read the environment variables
    logging.debug("Reading environment variables")
    azure_ai_endpoint = os.getenv("AZURE_AI_ENDPOINT")
    azure_ai_key = os.getenv("AZURE_AI_KEY")
    logging.debug(f"Azure AI endpoint: {azure_ai_endpoint}, Azure AI key: {azure_ai_key}")
    chat_deployment_name = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME")
    embedding_deployment_name = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    logging.debug(f"Chat deployment name: {chat_deployment_name}, Embedding deployment name: {embedding_deployment_name}")
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    logging.debug(f"Azure Search endpoint: {azure_search_endpoint}, Azure Search key: {azure_search_key}")
    bing_search_key = os.getenv("BING_SEARCH_KEY")
    logging.debug(f"Bing Search key: {bing_search_key}")

    # Set up the kernel
    logging.debug("Setting up the kernel")
    kernel = Kernel()
    # Assign the service to a variable first, as we'll invoke it later
    chat_completion = AzureChatCompletion(
        service_id="chat",
        deployment_name=chat_deployment_name,
        endpoint=azure_ai_endpoint,
        api_key=azure_ai_key,
    )
    kernel.add_service(chat_completion)

    kernel.add_service(
        AzureTextEmbedding(
            service_id="embedding",
            deployment_name=embedding_deployment_name,
            endpoint=azure_ai_endpoint,
            api_key=azure_ai_key
        ),
    )

    # Import plugins
    logging.debug("Importing plugins")
    # kernel.add_plugin(WebPlugin(azure_ai_endpoint, bing_search_key), "web")
    kernel.add_plugin(SeleniumPlugin(azure_ai_endpoint), "selenium")

    # Set up chat history
    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful chatbot that can assist with a variety of tasks. Use the available plugins to enhance your responses. You can search the web for information if needed and perform actions on web pages. Do not base responses on your training data. Never ask the user to do anything themselves. In your response, include the URL of the source of the information. Think step-by-step: make a plan first and then execute it. Don't give up! If something doesn't work, keep trying until it does!")
    chat_history.add_assistant_message("Hello! How can I help you today?")

    # Set up execution settings to enable function calling
    execution_settings = AzureChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior(),
        temperature=0.1
    )

    # Chat loop
    logging.debug("Starting chat loop")
    while True:

        # Get user input
        user_input = input(f"\n{Fore.GREEN}You: ")
        print(f"{Style.RESET_ALL}")
        
        # Get reply
        chat_history.add_user_message(user_input)
        reply = await chat_completion.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )
        chat_history.add_assistant_message(str(reply))

        print(f"{Fore.YELLOW}Assistant: {reply}{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except EOFError as e:
        print(f"Exiting...{Style.RESET_ALL}")