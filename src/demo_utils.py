#!/usr/bin/env python3

###
# Environment loading
###
import dotenv
import os
g_dotenv_loaded = False
def getenv(variable: str) -> str:
    global g_dotenv_loaded
    if not g_dotenv_loaded:
        g_dotenv_loaded = True
        dotenv.load_dotenv()
    value = os.getenv(variable)
    return value


def main():
    key_value = getenv("GEMINI_API_KEY")
    if key_value:
        print("Have value for GEMINI_API_KEY.")
    else:
        print("Do not have value for GEMINI_API_KEY.")

    return

if __name__ == "__main__":
    main()

