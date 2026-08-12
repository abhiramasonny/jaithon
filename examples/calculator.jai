enum Token {
    Number(value: float),
    Name(text: str),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
    Assign,
    End,
}

fn token_text(t: Token) -> str {
    return match t {
        Token.Number(v) => v.to_str(), Token.Name(s) => s,
        Token.Plus => "+", Token.Minus => "-", Token.Star => "*", Token.Slash => "/",
        Token.Percent => "%", Token.Caret => "**", Token.LParen => "(", Token.RParen => ")",
        Token.Comma => ",", Token.Assign => "=", Token.End => "end of input",
    }
}

fn tokenize(source: str) -> list[Token] {
    var tokens = []
    var i = 0
    let n = source.len()

    while i < n {
        let ch = source[i]

        if ch == " " or ch == "\t" {
            i += 1
        } elif ch.is_digit() or ch == "." {
            var j = i
            while j < n and (source[j].is_digit() or source[j] == ".") { j += 1 }
            let text = source[i:j]
            let value = text.to_float()
            if value == null { throw ParseError(f"bad number literal: {text}") }
            tokens.push(Token.Number(value ?? 0.0))
            i = j
        } elif ch.is_alpha() or ch == "_" {
            var j = i
            while j < n and (source[j].is_alnum() or source[j] == "_") { j += 1 }
            tokens.push(Token.Name(source[i:j]))
            i = j
        } elif ch == "*" and i + 1 < n and source[i + 1] == "*" {
            tokens.push(Token.Caret)
            i += 2
        } else {
            let simple = match ch {
                "+" => Token.Plus, "-" => Token.Minus, "*" => Token.Star, "/" => Token.Slash,
                "%" => Token.Percent, "(" => Token.LParen, ")" => Token.RParen, "," => Token.Comma,
                "=" => Token.Assign, _ => Token.End,
            }
            if simple is Token.End { throw ParseError(f"unexpected character '{ch}' at column {i + 1}") }
            tokens.push(simple)
            i += 1
        }
    }

    tokens.push(Token.End)
    return tokens
}

enum Expr {
    Literal(value: float),
    Variable(name: str),
    Unary(op: str, operand: Expr),
    Binary(op: str, left: Expr, right: Expr),
    Call(name: str, args: list[Expr]),
    Assignment(name: str, value: Expr),
}

fn infix_power(t: Token) -> int {
    return match t { Token.Plus | Token.Minus => 10, Token.Star | Token.Slash | Token.Percent => 20, Token.Caret => 30, _ => 0 }
}

class Parser {
    var tokens: list[Token]
    var pos: int

    fn init(self, tokens: list[Token]) {
        self.tokens = tokens
        self.pos = 0
    }

    fn peek(self) -> Token { return self.tokens[self.pos] }
    fn advance(self) -> Token {
        let t = self.tokens[self.pos]
        self.pos += 1
        return t
    }

    fn expect(self, want: Token) -> void {
        let got = self.advance()
        if token_text(got) != token_text(want) {
            throw ParseError(f"expected {token_text(want)}, found {token_text(got)}")
        }
    }

    pub fn parse(self) -> Expr {
        let e = self.expression(0)
        self.expect(Token.End)
        return e
    }

    fn expression(self, min_power: int) -> Expr {
        var left = self.prefix()

        loop {
            let power = infix_power(self.peek())
            if power == 0 or power < min_power { break }

            let op = self.advance()
            let next_min = if op is Token.Caret { power } else { power + 1 }
            let right = self.expression(next_min)
            left = Expr.Binary(token_text(op), left, right)
        }

        return left
    }

    fn prefix(self) -> Expr {
        let t = self.advance()
        return match t { Token.Number(v) => Expr.Literal(v), Token.Minus => Expr.Unary("-", self.expression(25)), Token.Plus => self.expression(25), Token.LParen => self.grouping(), Token.Name(name) => self.name_or_call(name), _ => throw ParseError(f"unexpected {token_text(t)}"), }
    }

    fn grouping(self) -> Expr {
        let inner = self.expression(0)
        self.expect(Token.RParen)
        return inner
    }

    fn name_or_call(self, name: str) -> Expr {
        if self.peek() is Token.LParen {
            self.advance()
            var args = []
            if not self.peek() is Token.RParen {
                args.push(self.expression(0))
                while self.peek() is Token.Comma {
                    self.advance()
                    args.push(self.expression(0))
                }
            }
            self.expect(Token.RParen)
            return Expr.Call(name, args)
        }
        if self.peek() is Token.Assign {
            self.advance()
            return Expr.Assignment(name, self.expression(0))
        }
        return Expr.Variable(name)
    }
}

import std.math

class Interpreter {
    var vars: dict[str, float]

    fn init(self) {
        self.vars = {"pi": math.PI, "e": math.E}
    }

    pub fn eval(self, e: Expr) -> float { return match e { Expr.Literal(v) => v, Expr.Variable(n) => self.lookup(n), Expr.Unary(op, operand) => match op { "-" => -self.eval(operand), _ => self.eval(operand), }, Expr.Binary(op, left, right) => self.binary(op, self.eval(left), self.eval(right)), Expr.Call(name, args) => self.call(name, args.map(|a| self.eval(a))), Expr.Assignment(name, value) => self.assign(name, self.eval(value)), } }

    fn lookup(self, name: str) -> float {
        let v = self.vars.get(name)
        if v == null { throw NameError(f"undefined variable '{name}'") }
        return v ?? 0.0
    }

    fn assign(self, name: str, value: float) -> float {
        self.vars[name] = value
        return value
    }

    fn binary(self, op: str, a: float, b: float) -> float { return match op { "+" => a + b, "-" => a - b, "*" => a * b, "/" => self.divide(a, b), "%" => self.modulo(a, b), "**" => a ** b, _ => throw RuntimeError(f"unknown operator '{op}'"), } }

    fn divide(self, a: float, b: float) -> float {
        if b == 0.0 { throw DivisionByZeroError("division by zero") }
        return a / b
    }

    fn modulo(self, a: float, b: float) -> float {
        if b == 0.0 { throw DivisionByZeroError("modulo by zero") }
        return math.fmod(a, b)
    }

    fn call(self, name: str, args: list[float]) -> float {
        let unary = { "sqrt": math.sqrt, "abs": math.fabs, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.ln, "log2": math.log2, "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil }
        let f = unary.get(name)
        if f != null { if args.len() != 1 { throw TypeError(f"{name} takes 1 argument, got {args.len()}") }
        return (f ?? math.fabs)(args[0]) }
        if name == "min" or name == "max" { if args.len() == 0 { throw TypeError(f"{name} needs at least one argument") }
        return if name == "min" { args.min() ?? 0.0 } else { args.max() ?? 0.0 } }
        throw NameError(f"unknown function '{name}'")
    }
}

fn show(value: float) -> str {
    if math.is_finite(value) and value == math.floor(value) and math.fabs(value) < 1e15 { return int(value).to_str() }
    return value.to_str()
}

fn main() -> int {
    let interp = Interpreter()
    print("Jaithon calculator. Blank line to quit.")

    loop {
        let line = input("> ").strip()
        if line.len() == 0 { break }

        try {
            let tokens = tokenize(line)
            let tree = Parser(tokens).parse()
            print(show(interp.eval(tree)))
        } catch e: ParseError | NameError | TypeError | DivisionByZeroError | RuntimeError {
            print(f"error: {e.message}")
        }
    }

    return 0
}
