import re

# Lexer
class Lexer:
    def __init__(self, source):
        self.tokens = []
        self.tokenize(source)

    def tokenize(self, source):
        token_specification = [
            ('NUMBER',   r'\d+'),         # Integer
            ('PRINT',    r'PRINT'),       # Print keyword
            ('PUSH',     r'PUSH'),        # Push keyword
            ('ADD',      r'ADD'),         # Add keyword
            ('HALT',     r'HALT'),        # Halt keyword
            ('STRING',   r'"[^"]*"'),     # String literals
            ('SKIP',     r'[ \t]+'),      # Skip whitespace
            ('NEWLINE',  r'\n'),          # Line endings
            ('MISMATCH', r'.'),           # Any other character
        ]
        
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
        for mo in re.finditer(tok_regex, source):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'NUMBER':
                value = int(value)
            elif kind == 'STRING':
                value = value[1:-1]  # Strip quotes
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'{value} unexpected')
            self.tokens.append((kind, value))

# AST Nodes
class ASTNode:
    pass

class PushNode(ASTNode):
    def __init__(self, value):
        self.value = value

class AddNode(ASTNode):
    pass

class PrintNode(ASTNode):
    def __init__(self, message):
        self.message = message

class HaltNode(ASTNode):
    pass

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_index = 0

    def parse(self):
        nodes = []
        while self.current_token_index < len(self.tokens):
            token_type, token_value = self.tokens[self.current_token_index]
            if token_type == 'PUSH':
                self.current_token_index += 1
                value = self.tokens[self.current_token_index][1]
                nodes.append(PushNode(value))
            elif token_type == 'ADD':
                nodes.append(AddNode())
            elif token_type == 'PRINT':
                self.current_token_index += 1
                message = self.tokens[self.current_token_index][1]
                nodes.append(PrintNode(message))
            elif token_type == 'HALT':
                nodes.append(HaltNode())
            self.current_token_index += 1
        return nodes

# Symbol Table
class SymbolTable:
    def __init__(self):
        self.symbols = {}

    def define(self, name, value):
        self.symbols[name] = value

    def lookup(self, name):
        return self.symbols.get(name)

# Type Checking
class TypeChecker:
    def __init__(self, ast):
        self.ast = ast
        self.errors = []

    def check(self):
        for node in self.ast:
            if isinstance(node, PushNode):
                if not isinstance(node.value, int):
                    self.errors.append("PUSH value must be an integer.")
            elif isinstance(node, AddNode):
                # Placeholder for future type checks
                pass
        return self.errors

# Code Generation
class CodeGenerator:
    def __init__(self, ast):
        self.ast = ast
        self.code = []

    def generate(self):
        self.code.append("section .data")
        self.code.append('hello db "Hello, World",0')
        self.code.append("section .text")
        self.code.append("global _start")
        self.code.append("_start:")

        for node in self.ast:
            if isinstance(node, PushNode):
                self.code.append(f"    push {node.value}")
            elif isinstance(node, AddNode):
                self.code.append("    pop rax")
                self.code.append("    pop rbx")
                self.code.append("    add rax, rbx")
                self.code.append("    push rax")
            elif isinstance(node, PrintNode):
                self.code.append("    mov rax, 1")
                self.code.append("    mov rdi, 1")
                self.code.append("    mov rsi, hello")
                self.code.append("    mov rdx, 13")
                self.code.append("    syscall")
            elif isinstance(node, HaltNode):
                self.code.append("    mov rax, 60")
                self.code.append("    xor rdi, rdi")
                self.code.append("    syscall")

        return '\n'.join(self.code)

# Main Execution
if __name__ == "__main__":
    source_code = """
    PUSH 10
    PUSH 7
    ADD
    PRINT "Hello, World"
    HALT
    """
    
    lexer = Lexer(source_code)
    print("Tokens:", lexer.tokens)

    parser = Parser(lexer.tokens)
    ast = parser.parse()
    print("AST:", ast)

    type_checker = TypeChecker(ast)
    errors = type_checker.check()
    if errors:
        print("Type Errors:", errors)
    else:
        code_generator = CodeGenerator(ast)
        nasm_code = code_generator.generate()
        print("Generated NASM Code:\n", nasm_code)
