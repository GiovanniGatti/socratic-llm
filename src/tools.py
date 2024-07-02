import re
from typing import Optional, Tuple

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


def safe_eval(client: OpenAI, content: str, max_retry: int = 3) -> Tuple[str, Optional[str], Optional[Evaluation]]:
    """
    Performs a call to gpt-4o and parses its assessments. Errors are managed to avoid failures.

    :param client: the OpenAI client
    :param content: the eval raw message
    :param max_retry: max number of retries to fix the input
    :return: A tuple containing GPT-4o raw message, the error message (if applicable) and the evaluation result (if
    parsing successful)
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": content}, ],
        model="gpt-4o",
        temperature=0.2,
        seed=0
    )
    content = chat_completion.choices[0].message.content

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
