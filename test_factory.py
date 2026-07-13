import os
from llm_factory import LLMFactory

if __name__ == "__main__":
# Test 1: Verify local configuration loading
    try:
        llm = LLMFactory().get_llm()
        response = llm.invoke("Say the 'Iam ready' and nothing else")
        print(f"Response content:\n {response.content.strip('\n')}")
    except Exception as e:
        print(f"Execution failed {str(e)}")
