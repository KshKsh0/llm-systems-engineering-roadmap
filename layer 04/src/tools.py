import ast
import operator

class ToolBox:
    def __init__(self):
        # A safe dictionary of mathematical operators
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow
        }

    def safe_calculator(self, expression: str) -> str:
        """Safely evaluates a mathematical string without using Python's risky eval()"""
        try:
            node = ast.parse(expression, mode='eval').body
            result = self._eval_node(node)
            return str(result)
        except Exception as e:
            return f"Error executing calculation: {str(e)}"

    def _eval_node(self, node):
        """Helper for safe AST evaluation"""
        if isinstance(node, ast.Num): # <number>
            return node.n
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
        raise TypeError(node)