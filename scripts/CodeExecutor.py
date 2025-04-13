from typing import Dict, Any

class CodeExecutor:
    """Handles code generation and validation"""
    
    @staticmethod
    def execute_code(code_str: str, context: Dict[str, Any]) -> Any:
        """Safely execute generated code"""
        try:
            exec_globals = context.copy()
            exec(code_str, exec_globals)
            return {k: v for k, v in exec_globals.items() if not k.startswith('__')}
        except Exception as e:
            print(f"Code execution failed: {str(e)}")
            return None
