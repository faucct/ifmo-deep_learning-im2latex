class Symbol:
    def __init__(self, symbol):
        self.symbol = symbol


class WithChildren:
    def __init__(self, header, children):
        self.header = header
        self.children = children


class SuperScripted:
    def __init__(self, element, elements):
        self.element = element
        self.elements = elements


class SubScripted:
    def __init__(self, element, elements):
        self.element = element
        self.elements = elements


class Fraction:
    def __init__(self, header, up, down):
        self.header = header
        self.up = up
        self.down = down


class MathChoice:
    def __init__(self, header, display, text, script, script_script):
        self.header = header
        self.display = display
        self.text = text
        self.script = script
        self.script_script = script_script


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
            tree.append(line)
