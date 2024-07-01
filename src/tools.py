import re


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
