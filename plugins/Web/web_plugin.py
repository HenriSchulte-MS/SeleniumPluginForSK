from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from pydantic import BaseModel, create_model
from azure.identity import AzureCliCredential, get_bearer_token_provider
import logging
import requests
from openai import AzureOpenAI

class WebPlugin:

    def __init__(self, ai_endpoint: str, search_key: str):
        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )
        self.ai_client = AzureOpenAI(
            azure_endpoint=ai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-08-01-preview"
        )
        self.search_key = search_key


    @kernel_function(
        description="Extract particular information from a web site based based on its source code. Use this function if the information is not available in the search results.",
        name="ExtractDataFromSite"
    )
    async def extract_data_from_site(self, url: Annotated[str, "URL of the website to extract data from"], fields_to_extract: Annotated[list[str], "The items of data to extract"]) -> Annotated[BaseModel, "The extracted data"]:
        
        # Create Pydantic model
        structure_dict = {field: (str, ...) for field in fields_to_extract}
        data_model = create_model('DataModel', **structure_dict)

        # Define class for the response
        class ParseResult(BaseModel):
            extracted_data: list[data_model] # type: ignore
            next_url: str

        extracted_data = []
        step = 0
        max_steps = 3  # Limit the number of steps to prevent infinite loops

        # Iterate until a phone number is found
        while len(extracted_data) == 0:
            step += 1
            if step > max_steps:
                return "Too many steps needed to extract requested data."

            logging.info(f"Step {step}: Fetching {url}...")

            # Get the source code from a url
            response = requests.get(url)
            source_code = response.text

            # Cap the source code at 300k characters
            if len(source_code) > 300_000:
                source_code = source_code[:300_000]

            # Parse the source code
            completion = self.ai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You will be given the source code of a website. Find the information specified below. If you cannot find the information, find the absolute URL to a location where that information may be found."},
                    {"role": "user", "content": f"The information to extract: {fields_to_extract}\n\nSource code:\n{source_code}"},
                ],
                response_format=ParseResult,
            )
            result = completion.choices[0].message.parsed
            if len(result.extracted_data) > 0:
                extracted_data = result.extracted_data
            else:
                if result.next_url != url:
                    url = result.next_url   
                else:
                    return "The requested data was not found."      

        return extracted_data
    

    @kernel_function(
        description="Perform a web search with a query to find relevant URLs, snippets, and deep links.",
        name="WebSearch"
    )
    def perform_web_search(self, query: Annotated[str, "Search query"]) -> Annotated[list[str], "Top 5 search results"]:
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.search_key}
        params = {"q": query}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        # Return top 5 search results
        results = []
        for result in search_results["webPages"]["value"][:5]:
            name = result["name"]
            url = result["url"]
            snippet = result["snippet"]
            deep_links = result.get("deepLinks", [])
            results.append({"name": name, "url": url, "snippet": snippet, "deep_links": deep_links})

        return results