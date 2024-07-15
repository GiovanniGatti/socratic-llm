import abc
import argparse
import re
from typing import Optional, Tuple, Dict, List

import httpx
import openai
from ollama import Client, ResponseError
from openai import OpenAI
from pydantic import ValidationError
from pydantic_core import from_json

from data import Evaluation


def escape_template(str_template: str) -> (str, set[str]):
    """
    Escapes '{' and '}' characters for string formatting. Ignores input keys with the format "{[a-zA-Z_]+}".

    This function is useful because "{...}" is a protected case in string formatting which conflicts with
    the JSON specification.

    :return: a Tuple with the escaped string and the input keys from the template
    """
    # Find all keys
    pattern = re.compile(r'{[a-zA-Z0-9_]+}')
    keys = pattern.findall(str_template)

    # Replace keys with a temporary placeholder
    placeholders = []
    protected_string = str_template
    for i, key in enumerate(keys):
        placeholder = f'__SPECIAL_CASE_{i}__'
        placeholders.append((placeholder, key))
        protected_string = protected_string.replace(key, placeholder)

    # escape '{' and '}'
    escaped_template = protected_string.replace('{', '{{').replace('}', '}}')

    # put back placeholders
    final_template = escaped_template
    for placeholder, key in placeholders:
        final_template = final_template.replace(placeholder, key)

    return final_template


class ClientLLM(abc.ABC):

    @abc.abstractmethod
    def chat(
            self, messages: List[Dict[str, str]], temperature: float = 0.2, seed: Optional[int] = 0
    ) -> str:
        ...

    @abc.abstractmethod
    def healthcheck(self) -> None:
        """
        :raises: ValueError if the Client cannot connect to provider or model is not available
        """
        pass


class OpenAIClient(ClientLLM):

    def __init__(self, openai_api_key: str, model: str):
        self._model = model
        self._client = OpenAI(api_key=openai_api_key)

    def chat(
            self, messages: List[Dict[str, str]], temperature: float = 0.2, seed: Optional[int] = 0
    ) -> str:
        chat_completion = self._client.chat.completions.create(
            messages=messages,
            model=self._model,
            temperature=temperature,
            seed=seed
        )
        return chat_completion.choices[0].message.content

    def healthcheck(self) -> None:
        try:
            models = self._client.models.list()
        except openai.AuthenticationError as e:
            raise ValueError("Unable to authenticate at OpenAI. Check if key is valid.", e)

        available_models = [m.id for m in models]
        if self._model not in available_models:
            raise ValueError(f"Invalid model. Expected one of {available_models}")


class OllamaClient(ClientLLM):

    def __init__(self, ollama_address: str, model: str):
        self._client = Client(host=ollama_address)
        self._model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, seed: Optional[int] = 0) -> str:
        chat_completion = self._client.chat(
            messages=messages,
            model=self._model,
            options=dict(
                temperature=temperature,
                seed=seed,
            )
        )
        return chat_completion["message"]["content"]

    def healthcheck(self) -> None:
        try:
            models = self._client.list()
        except httpx.ConnectError as e:
            raise ValueError("Unable to connect to Ollama server. Check server's address.", e)

        available_models = [m["name"] for m in models["models"]]

        if self._model not in available_models:
            try:
                print(f" === Pulling {self._model} from OllamaHub === ")
                self._client.pull(self._model)
            except ResponseError as e:
                raise ValueError("Model is unavailable. Unable to pull it.", e)


class JudgeLLM(argparse.Action):

    def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: List[str],
            option_string: List[str] = None
    ) -> None:
        if len(values) != 3:
            raise ValueError(f"Expected the model provider, the provider's information access, "
                             f"and the model name to use, but found {values}")

        provider, access_info, model = values

        if provider not in ("openai", "ollama"):
            raise ValueError(f"Only \"openai\" or \"ollama\" are the accepted providers. Found {provider}")

        client_llm: ClientLLM
        if provider == "ollama":
            client_llm = OllamaClient(ollama_address=access_info, model=model)
        else:
            client_llm = OpenAIClient(openai_api_key=access_info, model=model)

        client_llm.healthcheck()

        setattr(namespace, self.dest, client_llm)


def safe_eval(client: ClientLLM, content: str, max_retry: int = 3) -> Tuple[str, Optional[str], Optional[Evaluation]]:
    """
    Performs a call to the LLM and parses its assessments. Errors are managed to avoid failures.

    :param client: the client LLM
    :param content: the eval raw message
    :param max_retry: max number of retries to fix the input
    :return: A tuple containing the LLM's raw message, the error message (if applicable) and the evaluation result (if
    parsing successful)
    """

    content = client.chat([{"role": "user", "content": content}, ])

    i = 0
    error: Optional[Exception] = None
    while i < max_retry:
        try:
            deserialized = from_json(content, allow_partial=True)
        except ValueError as e:
            error = e
            i += 1
            continue

        try:
            evaluation = Evaluation.model_validate(deserialized)
            return content, None, evaluation
        except ValidationError as e:
            error = e
            i += 1
            continue

    return content, str(error), None
