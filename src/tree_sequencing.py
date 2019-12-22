from src.tree_parser import *


class SequencingVisitor(Visitor):
    def visit_list(self, elements):
        for element in elements:
            yield from element.visit(self)
        yield b'list_end'

    def visit_symbol(self, symbol):
        yield b'symbol:' + symbol.symbol

    def visit_with_children(self, with_children):
        yield b'with_children:' + with_children.header
        yield from self.visit_list(with_children.children)

    def visit_super_scripted(self, super_scripted):
        yield b'super_scripted'
        yield from super_scripted.element.visit(self)
        yield from self.visit_list(super_scripted.elements)

    def visit_sub_scripted(self, sub_scripted):
        yield b'sub_scripted'
        yield from sub_scripted.element.visit(self)
        yield from self.visit_list(sub_scripted.elements)

    def visit_fraction(self, fraction):
        yield b'fraction:' + fraction.header
        yield from self.visit_list(fraction.up)
        yield from self.visit_list(fraction.down)

    def visit_math_choice(self, math_choice):
        raise NotImplementedError()


def elements_to_sequence(elements):
    return list(SequencingVisitor().visit_list(elements))


def elements_from_sequence(sequence):
    sequence = iter(sequence)
    token = next(sequence)

    def element():
        nonlocal token
        if token == b'super_scripted':
            token = next(sequence)
            return SuperScripted(element(), list(elements_from_sequence(sequence)))
        if token == b'sub_scripted':
            token = next(sequence)
            return SubScripted(element(), list(elements_from_sequence(sequence)))
        kind, header = token.split(b':', 1)
        if kind == b'fraction':
            return Fraction(header, list(elements_from_sequence(sequence)), list(elements_from_sequence(sequence)))
        if kind == b'symbol':
            return Symbol(header)
        if kind == b'with_children':
            return WithChildren(header, list(elements_from_sequence(sequence)))
        else:
            raise Exception(token)

    while token != b'list_end':
        yield element()
        token = next(sequence)
