class Visitor:
    def visit_list(self, elements):
        for element in elements:
            element.visit(self)

    def visit_symbol(self, symbol):
        pass

    def visit_with_children(self, with_children):
        pass

    def visit_super_scripted(self, super_scripted):
        pass

    def visit_sub_scripted(self, sub_scripted):
        pass

    def visit_fraction(self, fraction):
        pass

    def visit_math_choice(self, math_choice):
        pass


class AllVisitor(Visitor):
    def __init__(self, visitor):
        self.visitor = visitor

    def visit_symbol(self, symbol):
        self.visitor.visit_symbol(symbol)

    def visit_with_children(self, with_children):
        self.visitor.visit_with_children(with_children)
        self.visit_list(with_children.children)

    def visit_super_scripted(self, super_scripted):
        self.visitor.visit_super_scripted(super_scripted)
        super_scripted.element.visit(self)
        self.visit_list(super_scripted.elements)

    def visit_sub_scripted(self, sub_scripted):
        self.visitor.visit_sub_scripted(sub_scripted)
        sub_scripted.element.visit(self)
        self.visit_list(sub_scripted.elements)

    def visit_fraction(self, fraction):
        self.visitor.visit_fraction(fraction)
        self.visit_list(fraction.up)
        self.visit_list(fraction.down)

    def visit_math_choice(self, math_choice):
        self.visitor.visit_math_choice(math_choice)
        self.visit_list(math_choice.display)
        self.visit_list(math_choice.text)
        self.visit_list(math_choice.script)
        self.visit_list(math_choice.script_script)


class Symbol:
    def __init__(self, symbol):
        self.symbol = symbol

    def visit(self, visitor):
        visitor.visit_symbol(self)


class WithChildren:
    def __init__(self, header, children):
        self.header = header
        self.children = children

    def visit(self, visitor):
        visitor.visit_with_children(self)


class SuperScripted:
    def __init__(self, element, elements):
        self.element = element
        self.elements = elements

    def visit(self, visitor):
        visitor.visit_super_scripted(self)


class SubScripted:
    def __init__(self, element, elements):
        self.element = element
        self.elements = elements

    def visit(self, visitor):
        visitor.visit_sub_scripted(self)


class Fraction:
    def __init__(self, header, up, down):
        self.header = header
        self.up = up
        self.down = down

    def visit(self, visitor):
        visitor.visit_fraction(self)


class MathChoice:
    def __init__(self, header, display, text, script, script_script):
        self.header = header
        self.display = display
        self.text = text
        self.script = script
        self.script_script = script_script

    def visit(self, visitor):
        visitor.visit_math_choice(self)


def iter_elements(lines):
    lines = iter(lines)
    line0 = next(lines)

    def iter_while_first_equals():
        nonlocal line1
        token = line1[0]
        yield line1[1:]
        for line1 in lines:
            if line1[0] == token:
                yield line1[1:]
            else:
                break
        else:
            line1 = None

    def assert_iter_while_first_equals(token):
        assert line1.startswith(token)
        return iter_while_first_equals()

    while True:
        line1 = next(lines, None)

        if line1 and line1.startswith(b'\\\\'):
            element = Fraction(
                line0,
                list(iter_elements(iter_while_first_equals())),
                list(iter_elements(assert_iter_while_first_equals(b'/')))
            )
        elif not line1 or line1.startswith((b'^', b'_', b'\\', b'/')):
            element = Symbol(line0)
        elif line1.startswith(b'.'):
            element = WithChildren(line0, list(iter_elements(iter_while_first_equals())))
        elif line1.startswith(b'D'):
            element = MathChoice(
                line0,
                list(iter_elements(iter_while_first_equals())),
                list(iter_elements(assert_iter_while_first_equals(b'T'))),
                list(iter_elements(assert_iter_while_first_equals(b'S'))),
                list(iter_elements(assert_iter_while_first_equals(b's')))
            )
        else:
            raise Exception(line1)
        if line1 and line1.startswith(b'^'):
            element = SuperScripted(element, list(iter_elements(iter_while_first_equals())))
        if line1 and line1.startswith(b'_'):
            element = SubScripted(element, list(iter_elements(iter_while_first_equals())))
        yield element
        if not line1:
            return
        line0 = line1


def iter_tree_lines(lines):
    tree = []
    denominator = []
    for line in lines:
        if line == b'this will be denominator of:\n':
            denominator = [b'/' + denominator_line for denominator_line in tree]
            tree = []
        elif line == b'\n':
            yield tree + denominator
            tree = []
            denominator = []
        else:
            if line.endswith(b' \n') and not line.endswith(b'  \n'):
                assert next(lines) == b'\n'
                line += b'\n'
            tree.append(line)
