from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Lexer
class Lexer:
    def __init__(self, source):
        self.tokens = []
        self.tokenize(source)

    def tokenize(self, source):
        token_specification = [
            ('NUMBER',   r'-?\d+(\.\d+)?'),  # Integer or float
            ('PRINT',    r'PRINT'),          # PRINT keyword
            ('INSERT',     r'INSERT'),           # INSERT keyword
            ('ADD',      r'ADD'),            # ADD keyword
            ('SUB',      r'SUB'),            # SUB keyword
            ('MUL',      r'MUL'),            # MUL keyword
            ('DIV',      r'DIV'),            # DIV keyword
            ('EXIT',     r'EXIT'),           # EXIT keyword
            ('IF',       r'IF'),             # IF keyword
            ('ELSE',     r'ELSE'),           # ELSE keyword
            ('ENDIF',    r'ENDIF'),          # ENDIF keyword
            ('LET',      r'LET'),            # Variable declaration
            ('IDENT',    r'[a-zA-Z_][a-zA-Z0-9_]*'),  # Identifiers
            ('ASSIGN',   r'='),              # Assignment operator
            ('GT',       r'>'),              # Greater than
            ('LT',       r'<'),              # Less than
            ('EQ',       r'=='),             # Equals
            ('STRING',   r'"[^"]*"'),        # String literal
            ('SKIP',     r'[ \t]+'),         # Skip spaces and tabs
            ('COMMENT',  r'//.*'),           # Skip comments
            ('NEWLINE',  r'\n'),             # Line endings
            ('MISMATCH', r'.'),              # Any other character
        ]
        
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
        for mo in re.finditer(tok_regex, source):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'NUMBER':
                value = int(value) if '.' not in value else float(value)
            elif kind == 'STRING':
                value = value[1:-1]
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'Unexpected token "{value}"')
            self.tokens.append((kind, value))
        return self.tokens

# AST Nodes
class ASTNode:
    pass

class PushNode(ASTNode):
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"type": "PushNode", "value": self.value}

class AddNode(ASTNode):
    def to_dict(self):
        return {"type": "AddNode"}

class SubNode(ASTNode):
    def to_dict(self):
        return {"type": "SubNode"}

class MulNode(ASTNode):
    def to_dict(self):
        return {"type": "MulNode"}

class PrintNode(ASTNode):
    def __init__(self, message):
        self.message = message

    def to_dict(self):
        return {"type": "PrintNode", "message": self.message}

class HaltNode(ASTNode):
    def to_dict(self):
        return {"type": "HaltNode"}

class DivNode(ASTNode):
    def to_dict(self):
        return {"type": "DivNode"}

class IfNode(ASTNode):
    def __init__(self, condition, true_block, false_block=None):
        self.condition = condition
        self.true_block = true_block
        self.false_block = false_block

    def to_dict(self):
        return {
            "type": "IfNode",
            "condition": self.condition.to_dict(),
            "true_block": [node.to_dict() for node in self.true_block],
            "false_block": [node.to_dict() for node in self.false_block] if self.false_block else None
        }

class ComparisonNode(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def to_dict(self):
        return {
            "type": "ComparisonNode",
            "left": self.left.to_dict(),
            "operator": self.operator,
            "right": self.right.to_dict()
        }

class VariableNode(ASTNode):
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"type": "VariableNode", "name": self.name}

class AssignmentNode(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_dict(self):
        return {
            "type": "AssignmentNode",
            "name": self.name,
            "value": self.value.to_dict()
        }

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_index = 0
        self.variables = {}  # Track declared variables

    def parse(self):
        nodes = []
        while self.current_token_index < len(self.tokens):
            token_type, token_value = self.tokens[self.current_token_index]
            if token_type == 'INSERT':
                self.current_token_index += 1
                value = self.tokens[self.current_token_index][1]
                nodes.append(PushNode(value))
            elif token_type == 'ADD':
                nodes.append(AddNode())
            elif token_type == 'SUB':
                nodes.append(SubNode())
            elif token_type == 'MUL':
                nodes.append(MulNode())
            elif token_type == 'DIV':
                nodes.append(DivNode())
            elif token_type == 'PRINT':
                self.current_token_index += 1
                message = self.tokens[self.current_token_index][1]
                nodes.append(PrintNode(message))
            elif token_type == 'EXIT':
                nodes.append(HaltNode())
            elif token_type == 'IF':
                nodes.append(self.parse_if_statement())
            elif token_type == 'LET':
                nodes.append(self.parse_variable_declaration())
            elif token_type == 'IDENT':
                nodes.append(self.parse_variable_assignment())
            self.current_token_index += 1
        return nodes

    # Add parse_condition method
    def parse_condition(self):
        # Get first operand
        left_token_type, left_value = self.tokens[self.current_token_index]
        if left_token_type == 'INSERT':
            self.current_token_index += 1
            left_value = self.tokens[self.current_token_index][1]
            left = PushNode(left_value)
        else:
            raise SyntaxError(f"Expected INSERT, got {left_token_type}")
        
        self.current_token_index += 1
        
        # Get comparison operator
        op_token_type, op_value = self.tokens[self.current_token_index]
        if op_token_type not in ['GT', 'LT', 'EQ']:
            raise SyntaxError(f"Expected comparison operator, got {op_token_type}")
            
        self.current_token_index += 1
        
        # Get second operand
        right_token_type, right_value = self.tokens[self.current_token_index]
        if right_token_type == 'INSERT':
            self.current_token_index += 1
            right_value = self.tokens[self.current_token_index][1]
            right = PushNode(right_value)
        else:
            raise SyntaxError(f"Expected INSERT, got {right_token_type}")
            
        return ComparisonNode(left, op_value, right)

    # Add parse_statement method
    def parse_statement(self):
        token_type, token_value = self.tokens[self.current_token_index]
        
        if token_type == 'INSERT':
            self.current_token_index += 1
            value = self.tokens[self.current_token_index][1]
            self.current_token_index += 1
            return PushNode(value)
        elif token_type == 'PRINT':
            self.current_token_index += 1
            message = self.tokens[self.current_token_index][1]
            self.current_token_index += 1
            return PrintNode(message)
        elif token_type == 'ADD':
            self.current_token_index += 1
            return AddNode()
        elif token_type == 'SUB':
            self.current_token_index += 1
            return SubNode()
        elif token_type == 'MUL':
            self.current_token_index += 1
            return MulNode()
        elif token_type == 'DIV':
            self.current_token_index += 1
            return DivNode()
        elif token_type == 'EXIT':
            self.current_token_index += 1
            return HaltNode()
        else:
            raise SyntaxError(f"Unexpected token type: {token_type}")

    # Update parse_if_statement to handle NEWLINE tokens
    def parse_if_statement(self):
        self.current_token_index += 1  # Skip IF token
        condition = self.parse_condition()
        true_block = []
        false_block = []
        
        # Skip any NEWLINE after condition
        while self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == 'NEWLINE':
            self.current_token_index += 1
        
        while self.current_token_index < len(self.tokens):
            token_type, _ = self.tokens[self.current_token_index]
            if token_type == 'ELSE':
                self.current_token_index += 1
                # Skip any NEWLINE after ELSE
                while self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == 'NEWLINE':
                    self.current_token_index += 1
                break
            elif token_type == 'ENDIF':
                self.current_token_index += 1
                return IfNode(condition, true_block)
            elif token_type == 'NEWLINE':
                self.current_token_index += 1
                continue
            
            true_block.append(self.parse_statement())
        
        while self.current_token_index < len(self.tokens):
            token_type, _ = self.tokens[self.current_token_index]
            if token_type == 'ENDIF':
                self.current_token_index += 1
                break
            elif token_type == 'NEWLINE':
                self.current_token_index += 1
                continue
                
            false_block.append(self.parse_statement())
        
        return IfNode(condition, true_block, false_block)

# Type Checking
class TypeChecker:
    def __init__(self, ast):
        self.ast = ast
        self.errors = []

    def check(self):
        stack_depth = 0
        has_halt = False

        for node in self.ast:
            if isinstance(node, HaltNode):
                has_halt = True
            elif isinstance(node, PushNode):
                if not (isinstance(node.value, int) or isinstance(node.value, float)):
                    self.errors.append("INSERT value must be an integer.")
                stack_depth += 1
            elif isinstance(node, AddNode):
                if stack_depth < 2:
                    self.errors.append("ADD requires at least two values on the stack.")
                else:
                    stack_depth -= 1
            elif isinstance(node, SubNode):
                if stack_depth < 2:
                    self.errors.append("SUB requires at least two values on the stack.")
                else:
                    stack_depth -= 1
            elif isinstance(node, MulNode):
                if stack_depth < 2:
                    self.errors.append("MUL requires at least two values on the stack.")
                else:
                    stack_depth -= 1
            elif isinstance(node, DivNode):
                if stack_depth < 2:
                    self.errors.append("DIV requires at least two values on the stack.")
                else:
                    stack_depth -= 1
            elif isinstance(node, PrintNode):
                if not isinstance(node.message, str):
                    self.errors.append("PRINT message must be a string.")
            else:
                self.errors.append(f"Unknown node type: {type(node).__name__}")

        if not has_halt:
            self.errors.append("Program must end with EXIT statement")
        return self.errors

# Code Generation
class CodeGenerator:
    def __init__(self, ast):
        self.ast = ast
        self.code = []
        self.accumulator = 0  # For simulation
        self.registers = {'eax': 0, 'ebx': 0}
        self.output = ""  # Captured output during simulation
        self.variables = {}
        self.label_count = 0

    def generate(self):
        self.code.append(".data")
        self.code.append('hello db "Hello, World",0')
        self.code.append(".code")
        self.code.append("main PROC")
        for node in self.ast:
            if type(node) == PushNode:
                self.code.append(f"    INSERT {node.value}")
            elif type(node) == AddNode:
                self.code.append("    pop eax")
                self.code.append("    pop ebx")
                self.code.append("    add eax, ebx")
                self.code.append("    INSERT eax")
            elif type(node) == SubNode:
                self.code.append("    pop eax")
                self.code.append("    pop ebx")
                self.code.append("    sub eax, ebx")
                self.code.append("    INSERT eax")
            elif type(node) == MulNode:
                self.code.append("    pop eax")
                self.code.append("    pop ebx")
                self.code.append("    imul ebx")
                self.code.append("    INSERT eax")
            elif type(node) == DivNode:
                self.code.append("    pop eax")
                self.code.append("    pop ebx")
                self.code.append("    cdq")  # Sign-extend eax into edx for division
                self.code.append("    idiv ebx")
                self.code.append("    INSERT eax")
            elif type(node) == PrintNode:
                self.code.append("    mov eax, 4")
                self.code.append("    mov ebx, 1")
                self.code.append("    lea ecx, hello")
                self.code.append("    mov edx, 13")
                self.code.append("    int 80h")  # For Windows, replace this with a proper print call
            elif type(node) == HaltNode:
                self.code.append("    mov eax, 1")
                self.code.append("    mov ebx, 0")
                self.code.append("    int 80h")  # For Windows, this would also be different
            elif isinstance(node, IfNode):
                self.generate_if(node)
            elif isinstance(node, AssignmentNode):
                self.generate_assignment(node)
            elif isinstance(node, VariableNode):
                self.generate_variable_access(node)

        self.code.append("main ENDP")
        self.code.append("END main")
        return '\n'.join(self.code)

    def generate_if(self, node):
        label_else = f"L_else_{self.label_count}"
        label_end = f"L_end_{self.label_count}"
        self.label_count += 1

        # Generate condition code
        self.generate_condition(node.condition)
        self.code.append("    pop eax")
        self.code.append("    test eax, eax")
        self.code.append(f"    jz {label_else}")

        # Generate true block
        for stmt in node.true_block:
            self.generate_statement(stmt)
        self.code.append(f"    jmp {label_end}")

        # Generate else block
        self.code.append(f"{label_else}:")
        if node.false_block:
            for stmt in node.false_block:
                self.generate_statement(stmt)

        self.code.append(f"{label_end}:")


    def simulate(self):
        stack = []
        for node in self.ast:
            try:
                if isinstance(node, PushNode):
                    stack.append(node.value)
                elif isinstance(node, AddNode):
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        result = a + b
                        stack.append(result)
                        self.accumulator = result
                        self.registers['eax'] = result
                    else:
                        self.output += "Error: Not enough values on the stack for ADD\n"
                elif isinstance(node, SubNode):
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        result = a - b
                        stack.append(result)
                        self.accumulator = result
                        self.registers['eax'] = result
                    else:
                        self.output += "Error: Not enough values on the stack for SUB\n"
                elif isinstance(node, MulNode):
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        result = a * b
                        stack.append(result)
                        self.accumulator = result
                        self.registers['eax'] = result
                    else:
                        self.output += "Runtime Error: Stack underflow in MUL operation\n"
                elif isinstance(node, DivNode):
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        if b == 0:
                            self.output += "Runtime Error: Division by zero\n"
                            break
                        result = a / b
                        stack.append(result)
                        self.accumulator = result
                        self.registers['eax'] = result
                    else:
                        self.output += "Runtime Error: Stack underflow in DIV operation\n"
                elif isinstance(node, PrintNode):
                    message = node.message
                    # If there's a value on the stack, append it to the message
                    if stack and message.lower().startswith("final result"):
                        current_value = stack[-1]  # Get last value without popping
                        # Format number to handle both int and float
                        if isinstance(current_value, float):
                            formatted_value = f"{current_value:.6f}"
                        else:
                            formatted_value = str(current_value)
                        self.output += f"{message} {formatted_value}\n"
                    else:
                        self.output += f"{message}\n"
                elif isinstance(node, HaltNode):
                    break
            except Exception as e:
                if "division by zero" in str(e).lower():
                    self.output += "Runtime Error: Division by zero\n"
                else:
                    self.output += f"Runtime Error: {str(e)}\n"
                break

        # Add final state to output
        if stack:
            self.output += f"\nFinal Stack: {stack}\n"
        self.output += f"Final Register State:\n"
        self.output += f"Accumulator: {self.accumulator}\n"
        self.output += f"Registers: {self.registers}\n"
        
        return {
            'accumulator': self.accumulator,
            'registers': self.registers,
            'stack': stack,
            'output': self.output.strip()
        }

@app.route('/tokenize', methods=['POST'])
def tokenize():
    try:
        source_code = request.json.get('source', '')
        lexer = Lexer(source_code)
        tokens = lexer.tokens
        return jsonify(tokens=tokens)
    except RuntimeError as e:
        return jsonify(error=str(e)), 400

@app.route('/parse', methods=['POST'])
def parse():
    try:
        tokens = request.json.get('tokens', [])
        print("Received tokens for parsing:", tokens)  # Debugging line
        parser = Parser(tokens)
        ast = parser.parse()
        print("Generated AST:", ast)  # Debugging line
        return jsonify(ast=[node.to_dict() for node in ast])  # Convert nodes to dicts
    except Exception as e:
        print(f"Error during parsing: {str(e)}")  # Log the error
        return jsonify(error="An error occurred during parsing"), 500

@app.route('/typecheck', methods=['POST'])
def typecheck():
    try:
        ast_data = request.json.get('ast', [])
        # Create the AST node instances from the received data
        ast = []
        for node_data in ast_data:
            if node_data['type'] == "PushNode":
                ast.append(PushNode(node_data['value']))
            elif node_data['type'] == "AddNode":
                ast.append(AddNode())
            elif node_data['type'] == "SubNode":
                ast.append(SubNode())
            elif node_data['type'] == "MulNode":
                ast.append(MulNode())
            elif node_data['type'] == "DivNode":
                ast.append(DivNode())
            elif node_data['type'] == "PrintNode":
                ast.append(PrintNode(node_data['message']))
            elif node_data['type'] == "HaltNode":
                ast.append(HaltNode())
            else:
                raise ValueError(f"Unknown node type: {node_data['type']}")

        type_checker = TypeChecker(ast)
        errors = type_checker.check()
        return jsonify(errors=errors)
    except Exception as e:
        print(f"Error during type checking: {str(e)}")  # Log the error
        return jsonify(error="An error occurred during type checking"), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        ast_data = request.json.get('ast', [])
        # Create the AST node instances from the received data
        ast = []
        for node_data in ast_data:
            if node_data['type'] == "PushNode":
                ast.append(PushNode(node_data['value']))
            elif node_data['type'] == "AddNode":
                ast.append(AddNode())
            elif node_data['type'] == "SubNode":  
                ast.append(SubNode())
            elif node_data['type'] == "MulNode":
                ast.append(MulNode())
            elif node_data['type'] == "DivNode":
                ast.append(DivNode())
            elif node_data['type'] == "PrintNode":
                ast.append(PrintNode(node_data['message']))
            elif node_data['type'] == "HaltNode":
                ast.append(HaltNode())
            else:
                raise ValueError(f"Unknown node type: {node_data['type']}")

        code_generator = CodeGenerator(ast)
        nasm_code = code_generator.generate()
        simulation_result = code_generator.simulate()
        
        if "Runtime Error:" in simulation_result['output']:
            return jsonify(error=simulation_result['output']), 400
            
        return jsonify(nasm_code=nasm_code, simulation=simulation_result)
    except Exception as e:
        return jsonify(error=f"Runtime Error: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=True)
